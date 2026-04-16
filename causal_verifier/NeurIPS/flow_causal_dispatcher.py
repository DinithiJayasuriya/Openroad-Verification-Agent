"""flow_causal_dispatcher.py — Executes FlowControllerDecision actions.

Mirrors dispatcher.py in structure. Handles controller actions:
  re_generate      — call LLM with constraint prompt + repair hint → update state
  re_retrieve_edge — re-run RAG for a specific metric edge, then re_generate
  commit_best      — mark state.committed = True
  stop_fail        — mark state as failed

Also owns the five bootstrap steps:
  bootstrap_flow_decompose  — task → ActionGraph  (TaskDecomposer)
  bootstrap_flow_extract    — ActionGraph → MultiActionChains  (FlowMultiChainExtractor)
                              also populates state.all_edges + state.edge_apis
  bootstrap_flow_generate   — MultiActionChains → code  (LLM)
  bootstrap_static_verify   — code → VerifierSnapshot  (OpenROADStaticVerifier L1-L3)
  bootstrap_flow_l4a_verify — code → L4aSnapshot  (FlowL4aVerifier A1-A4)
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

import numpy as np

try:
    from .flow_causal_state        import FlowCausalAgentState, L4aSnapshot
    from .flow_causal_controller   import FlowControllerDecision
    from .flow_task_decomposer     import TaskDecomposer
    from .flow_multi_chain_extractor import FlowMultiChainExtractor
    from .flow_l4a_verifier        import FlowL4aVerifier
    from .causal_state             import VerifierSnapshot
except ImportError:
    from flow_causal_state         import FlowCausalAgentState, L4aSnapshot    # type: ignore
    from flow_causal_controller    import FlowControllerDecision                # type: ignore
    from flow_task_decomposer      import TaskDecomposer                        # type: ignore
    from flow_multi_chain_extractor import FlowMultiChainExtractor              # type: ignore
    from flow_l4a_verifier         import FlowL4aVerifier                      # type: ignore
    from causal_state              import VerifierSnapshot                      # type: ignore


# ── Generation system prompt ──────────────────────────────────────────────────

_GENERATION_SYSTEM_PROMPT = """\
You are an expert OpenROAD Python API programmer.
Generate a complete Python script that runs inside the OpenROAD interactive shell.

RULES:
1. `design` (openroad.Design) is pre-available — do NOT import or re-create it.
2. Follow the FLOW TASK CONSTRAINT BLOCK below EXACTLY:
   - Acquire each tool using the stated getter.
   - Call each tool's methods in the stated order.
   - Implement the sandwich structure if specified.
3. Each tool must be acquired independently from `design` — do not share variables across tools.
4. Output ONLY the Python code. No explanation, no markdown fences.
"""


# ── FlowCausalDispatcher ──────────────────────────────────────────────────────

def _rt_matches(rt: str, type_name: str) -> bool:
    """Word-boundary match — prevents 'dbTechLayer' matching 'dbTechLayerCutClassRule'."""
    return bool(re.search(
        r'(?<![A-Za-z0-9_])' + re.escape(type_name) + r'(?![A-Za-z0-9_])', rt
    ))


def _row_hit(row: dict, score: float) -> dict:
    return {
        "function_name": str(row.get("Function Name:", "")).strip(),
        "description":   str(row.get("Description:",  "")).strip(),
        "return_type":   str(row.get("Return Type:",  "")).strip(),
        "score":         round(score, 3),
    }


def _rag_requery(
    src: str, tgt: str, hint: str,
    embed_model, metadata: list, embeddings,
    top_k: int = 3, threshold: float = 0.25,
) -> Optional[dict]:
    """Re-retrieve RAG for a specific (src, tgt) edge with an enriched hint query.

    Pass 1: structural filter (return-type word-boundary + src name in fn name).
    Pass 2: enriched semantic query.
    """
    from sentence_transformers.util import cos_sim

    src_short = src.replace("odb.", "").replace("openroad.", "")
    tgt_short = tgt.replace("odb.", "").replace("openroad.", "")

    # Pass 1 — strict: return type matches AND source in function name
    strict = []
    for row in metadata:
        rt = str(row.get("Return Type:", "")).strip()
        fn = str(row.get("Function Name:", "")).strip()
        if (_rt_matches(rt, tgt_short) or _rt_matches(rt, tgt)) and \
           (src_short.lower() in fn.lower() or src.lower() in fn.lower()):
            strict.append(_row_hit(row, 1.0))
    if strict:
        return strict[0]

    # Pass 1 — loose: return type matches only
    loose = []
    for row in metadata:
        rt = str(row.get("Return Type:", "")).strip()
        if _rt_matches(rt, tgt_short) or _rt_matches(rt, tgt):
            loose.append(_row_hit(row, 0.9))
    if loose:
        return loose[0]

    # Pass 2 — semantic query enriched with hint
    query  = f"{src} {tgt} {src_short} {tgt_short} method {hint[:60]}"
    q_emb  = embed_model.encode(query, convert_to_tensor=True)
    scores = cos_sim(q_emb, embeddings).cpu().numpy().flatten()
    best_i = int(np.argmax(scores))
    if scores[best_i] >= threshold:
        return _row_hit(metadata[best_i], float(scores[best_i]))
    return None


class FlowCausalDispatcher:
    """Executes bootstrap steps and controller decisions for flow tasks.

    Parameters
    ----------
    api_key      : OpenAI API key (for decomposer + generator).
    rag_api_path : Path to RAGAPIs.csv (for FlowMultiChainExtractor + re_retrieve_edge).
    model        : LLM model (default gpt-4.1-mini).
    """

    def __init__(self, api_key: str, rag_api_path: str, model: str = "gpt-4.1-mini"):
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        self._api_key     = api_key
        self._rag_api_path = rag_api_path
        self._model       = model
        self._decomposer  = TaskDecomposer(openai_key=api_key, model=model)
        self._extractor   = FlowMultiChainExtractor(rag_api_path)
        self._verifier    = FlowL4aVerifier()

        # ── RAG retrieval data (for re_retrieve_edge) ─────────────────────────
        print("  [FlowDispatcher] Loading embedding model...", flush=True)
        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        rag_df    = pd.read_csv(rag_api_path)
        metadata  = []
        documents = []
        for _, row in rag_df.iterrows():
            desc = str(row.get("Description:", "")).strip()
            if not desc or desc.lower() == "nan":
                continue
            documents.append(f"OpenROAD Python API Description:{desc}")
            metadata.append(row.to_dict())
        self._metadata  = metadata
        self._embeddings = self._embed_model.encode(
            documents, convert_to_tensor=True, show_progress_bar=False
        )
        print(f"  [FlowDispatcher] RAG ready ({len(metadata)} entries).", flush=True)

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def bootstrap_flow_decompose(self, state: FlowCausalAgentState) -> None:
        """Bootstrap step 1: task → ActionGraph via TaskDecomposer."""
        print("  [boot/flow_decompose] ...", flush=True)
        graph = self._decomposer.decompose(state.task)
        if graph is None:
            state.add_bootstrap_obs(
                "flow_decompose",
                "[FAILED] TaskDecomposer returned None — task may not be a flow task",
            )
            print("  [boot/flow_decompose] FAILED", flush=True)
            return

        state.action_graph = graph
        summary = graph.ordering_summary()
        state.add_bootstrap_obs(
            "flow_decompose",
            f"OK: {summary}  sandwich={graph.sandwich}",
        )
        print(f"  [boot/flow_decompose] {summary}", flush=True)

    def bootstrap_flow_extract(self, state: FlowCausalAgentState) -> None:
        """Bootstrap step 2: ActionGraph → MultiActionChains.

        Also populates state.all_edges + state.edge_apis from metric chains
        so the controller can target specific edges with re_retrieve_edge.
        """
        if state.action_graph is None:
            state.add_bootstrap_obs("flow_extract", "[SKIP] no action_graph")
            return

        print("  [boot/flow_extract] ...", flush=True)
        multi = self._extractor.extract(state.action_graph)
        state.multi_chains = multi

        # Populate metric chain edges for re_retrieve_edge
        all_edges: List[Tuple[str, str]] = []
        edge_apis = []
        for chain in multi.metric_chains:
            path = chain.chain_types
            for i in range(len(path) - 1):
                all_edges.append((path[i], path[i + 1]))
                # Attach the NodeAPIEntry as a dict (or None if unknown)
                api = chain.node_apis[i] if i < len(chain.node_apis) else None
                if api is not None and not api.method_name.startswith("<UNKNOWN"):
                    edge_apis.append({
                        "function_name": f"{api.source_type}.{api.method_name}(",
                        "description":   api.description,
                        "return_type":   api.target_type,
                        "score":         1.0,
                    })
                else:
                    edge_apis.append(None)

        state.all_edges = all_edges
        state.edge_apis = edge_apis

        state.add_bootstrap_obs("flow_extract", multi.summary())
        print(f"  [boot/flow_extract] {multi.summary()}  "
              f"({len(all_edges)} metric edges)", flush=True)

    def bootstrap_flow_generate(self, state: FlowCausalAgentState) -> None:
        """Bootstrap step 3: generate code from constraint prompt."""
        if state.multi_chains is None:
            state.add_bootstrap_obs("flow_generate", "[SKIP] no multi_chains")
            return

        print("  [boot/flow_generate] ...", flush=True)
        code = self._generate(state.task, state.multi_chains.to_full_constraint_prompt())
        if not code:
            state.add_bootstrap_obs("flow_generate", "[FAILED] LLM returned empty")
            return

        state.current_code = code
        state.code_history.append(code)
        state.add_bootstrap_obs(
            "flow_generate",
            f"OK: {len(code)} chars",
            detail=code,
        )
        print(f"  [boot/flow_generate] {len(code)} chars", flush=True)

    def bootstrap_static_verify(self, state: FlowCausalAgentState) -> None:
        """Bootstrap step 4 (between generate and L4a): L1-L3 static verification.

        Runs OpenROADStaticVerifier (syntax, DB hierarchy, API signatures) on
        the generated code and records result as state.static_result.
        Does NOT gate the pipeline — only records so controller can see it.
        """
        if not state.current_code:
            state.add_bootstrap_obs("static_verify", "[SKIP] no code")
            return

        # Import lazily (requires src_1_reflector on sys.path)
        try:
            from verifier import OpenROADStaticVerifier
        except ImportError:
            state.add_bootstrap_obs("static_verify", "[SKIP] verifier not importable")
            return

        print("  [boot/static_verify] ...", flush=True)
        sv = OpenROADStaticVerifier(self._rag_api_path)
        vr = sv.verify(state.task, state.current_code, rag_context="")

        snap = VerifierSnapshot(
            passed       = vr.passed,
            layer_failed = vr.layer_failed,
            issues       = vr.issues or [],
            feedback     = vr.feedback or "",
            api_diffs    = getattr(vr, "api_diffs", None) or [],
        )
        state.static_result = snap

        status = "PASS" if snap.passed else f"FAIL(L{snap.layer_failed})"
        state.add_bootstrap_obs(
            "static_verify",
            status,
            detail=snap.feedback,
        )
        print(f"  [boot/static_verify] {status}", flush=True)

    def bootstrap_flow_l4a_verify(self, state: FlowCausalAgentState) -> None:
        """Bootstrap step 4: L4a verification on generated code."""
        if not state.current_code or state.multi_chains is None:
            state.add_bootstrap_obs("flow_l4a_verify", "[SKIP] no code or chains")
            return

        print("  [boot/flow_l4a_verify] ...", flush=True)
        result = self._verifier.verify(state.current_code, state.multi_chains)
        snap   = L4aSnapshot.from_result(result)
        state.l4a_result = snap
        state.maybe_update_best()

        status = "PASS" if snap.passed else f"FAIL checks={snap.issue_checks}"
        state.add_bootstrap_obs(
            "flow_l4a_verify",
            status,
            detail=snap.feedback,
        )
        print(f"  [boot/flow_l4a_verify] {status}", flush=True)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch(
        self,
        state:    FlowCausalAgentState,
        decision: FlowControllerDecision,
    ) -> None:
        """Execute the controller's decision. Updates state in-place."""

        # Write lesson before executing
        if decision.updated_lesson:
            state.add_lesson(decision.updated_lesson)

        action = decision.next_action

        if action == "re_generate":
            self._do_re_generate(state, decision.repair_hint)

        elif action == "re_retrieve_edge":
            self._do_re_retrieve_edge(state, decision)

        elif action == "commit_best":
            self._do_commit(state)

        elif action == "stop_fail":
            state.add_observation(
                "stop_fail",
                "STOP: budget exhausted, no passing code",
            )
            print("  [dispatcher] stop_fail", flush=True)

        else:
            # Unknown action — fall back to re_generate
            print(f"  [dispatcher] unknown action '{action}' — re_generating", flush=True)
            self._do_re_generate(state, decision.repair_hint)

    # ── Action implementations ────────────────────────────────────────────────

    def _do_re_generate(
        self, state: FlowCausalAgentState, repair_hint: str, _no_budget: bool = False
    ) -> None:
        """Re-generate code with repair hint injected.

        Parameters
        ----------
        _no_budget : If True, do not call add_observation (the caller handles
                     budget accounting).  Used internally by _do_re_retrieve_edge
                     so the combined action costs exactly 1 budget unit.
        """
        if state.multi_chains is None:
            if not _no_budget:
                state.add_observation("re_generate", "[SKIP] no multi_chains")
            return

        print(f"  [dispatcher/re_generate] ...", flush=True)

        constraint_prompt = state.multi_chains.to_full_constraint_prompt()
        code = self._generate(state.task, constraint_prompt, repair_hint)

        if not code:
            if not _no_budget:
                state.add_observation("re_generate", "[FAILED] LLM returned empty")
            return

        state.current_code = code
        state.code_history.append(code)

        # Re-run both static (L1-L3) and L4a checks
        self._re_static_verify(state)

        result = self._verifier.verify(code, state.multi_chains)
        snap   = L4aSnapshot.from_result(result)
        state.l4a_result = snap
        state.maybe_update_best()

        status = "PASS" if snap.passed else f"FAIL checks={snap.issue_checks}"
        if not _no_budget:
            state.add_observation("re_generate", status, detail=code)
        print(f"  [dispatcher/re_generate] {status}", flush=True)

    def _re_static_verify(self, state: FlowCausalAgentState) -> None:
        """Silently re-run L1-L3 static verifier and update state.static_result."""
        try:
            from verifier import OpenROADStaticVerifier
        except ImportError:
            return
        sv = OpenROADStaticVerifier(self._rag_api_path)
        vr = sv.verify(state.task, state.current_code, rag_context="")
        state.static_result = VerifierSnapshot(
            passed       = vr.passed,
            layer_failed = vr.layer_failed,
            issues       = vr.issues or [],
            feedback     = vr.feedback or "",
            api_diffs    = getattr(vr, "api_diffs", None) or [],
        )

    def _do_commit(self, state: FlowCausalAgentState) -> None:
        code = state.best_code or state.current_code
        state.committed      = True
        state.committed_code = code
        snap = state.best_snap
        status = "PASS" if (snap and snap.passed) else "BEST_FAIL"
        state.add_observation("commit_best", status)
        print(f"  [dispatcher/commit_best] {status}", flush=True)

    # ── re_retrieve_edge ──────────────────────────────────────────────────────

    def _do_re_retrieve_edge(
        self, state: FlowCausalAgentState, decision: FlowControllerDecision
    ) -> None:
        """Re-retrieve RAG for a specific metric chain edge, then re-generate.

        The controller sets decision.target_edge as "<src_short> → <tgt_short>"
        or a numeric index string.  If the edge cannot be parsed from
        state.all_edges, falls back directly to re_generate.
        """
        target = getattr(decision, "target_edge", "")
        edge_idx = self._parse_target_edge(target, state.all_edges)

        if edge_idx < 0 or not state.all_edges:
            print(f"  [dispatcher/re_retrieve_edge] cannot resolve edge '{target}' "
                  f"— falling back to re_generate", flush=True)
            self._do_re_generate(state, decision.repair_hint)
            return

        src, tgt = state.all_edges[edge_idx]
        print(f"  [dispatcher/re_retrieve_edge] edge [{src}] → [{tgt}]", flush=True)

        hit = _rag_requery(
            src, tgt,
            hint=decision.repair_hint,
            embed_model=self._embed_model,
            metadata=self._metadata,
            embeddings=self._embeddings,
        )

        if hit:
            method = hit["function_name"].split("(")[0].split(".")[-1].strip()
            print(f"  [dispatcher/re_retrieve_edge] new hit: {hit['function_name']} "
                  f"(score={hit['score']})", flush=True)
            # Update the cached edge API so the constraint prompt stays consistent
            while len(state.edge_apis) <= edge_idx:
                state.edge_apis.append(None)
            state.edge_apis[edge_idx] = hit
            # Inject the new API into the repair hint
            hint = (
                f"For {src.split('.')[-1]}→{tgt.split('.')[-1]}: "
                f"use {method}() — {hit['description']}. "
                + decision.repair_hint
            )
        else:
            print(f"  [dispatcher/re_retrieve_edge] no RAG hit — "
                  f"keeping original hint", flush=True)
            hint = decision.repair_hint

        # re_retrieve_edge + re_generate cost ONE budget unit together
        self._do_re_generate(state, hint, _no_budget=True)

        l4a_status = "PASS" if (state.l4a_result and state.l4a_result.passed) \
            else f"FAIL checks={state.l4a_result.issue_checks if state.l4a_result else '?'}"
        state.add_observation(
            "re_retrieve_edge",
            (f"edge=[{src.split('.')[-1]}→{tgt.split('.')[-1]}] "
             f"hit={'YES' if hit else 'NO'} | {l4a_status}"),
        )

    @staticmethod
    def _parse_target_edge(target: str, all_edges: List[Tuple[str, str]]) -> int:
        """Return the index in all_edges for the edge described by *target*.

        Accepts:
          • Numeric string  "2"
          • Short name pair "Design → dbBlock" or "Design→dbBlock"
        Returns -1 if not found.
        """
        if not target or not all_edges:
            return -1
        target = target.strip()
        # Numeric index
        if target.isdigit():
            idx = int(target)
            return idx if idx < len(all_edges) else -1
        # Name-pair match (case-insensitive partial)
        parts = re.split(r"\s*[→>-]+\s*", target)
        if len(parts) >= 2:
            a, b = parts[0].lower().strip(), parts[-1].lower().strip()
            for i, (src, tgt) in enumerate(all_edges):
                if a in src.lower() and b in tgt.lower():
                    return i
        return -1

    # ── LLM code generation ───────────────────────────────────────────────────

    def _generate(
        self,
        task:              str,
        constraint_prompt: str,
        repair_hint:       str = "",
    ) -> str:
        user_parts = [
            f"Task: {task}",
            "",
            "FLOW TASK CONSTRAINT BLOCK (follow exactly):",
            constraint_prompt,
        ]
        if repair_hint:
            user_parts += [
                "",
                "REPAIR HINT — fix these issues from the previous attempt:",
                repair_hint,
            ]
        user_parts.append("\nOutput the complete Python script:")

        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
                {"role": "user",   "content": "\n".join(user_parts)},
            ],
            "temperature": 0,
            "max_tokens":  900,
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                text = body["choices"][0]["message"]["content"].strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = "\n".join(
                        l for l in text.splitlines()
                        if not l.strip().startswith("```")
                    ).strip()
                return text
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 10 * (2 ** attempt)
                    print(f"    [flow-generate rate-limit] waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    [flow-generate HTTP {e.code}]", flush=True)
                    return ""
            except Exception as exc:
                print(f"    [flow-generate error] {exc}", flush=True)
                return ""
        return ""
