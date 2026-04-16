"""dispatcher.py — Executes controller decisions for the causal agent loop.

Each action mutates CausalAgentState and returns a one-line observation string
that gets recorded in state.observations.

Actions
-------
re_generate
    Re-run the code generator with the same chain + edge_apis, but append the
    controller's repair_hint to the generation prompt so the LLM knows exactly
    what to fix. Then re-runs the causal verifier.

re_retrieve_edge
    Re-run targeted RAG retrieval for a specific chain edge (identified by
    controller.target_edge). Updates state.edge_apis for that edge with the
    new hit (or marks it None if nothing found). Then calls re_generate
    automatically so both sub-steps cost only 1 budget unit together.

commit_best
    Commits state.best_code (or state.current_code if no best) and sets
    state.committed = True, ending the loop.

stop_fail
    Same as commit_best but logs the failure reason explicitly.
"""

import re
from typing import Optional


def _rt_matches(rt: str, type_name: str) -> bool:
    """Word-boundary match — prevents 'dbTechLayer' matching 'dbTechLayerCutClassRule'."""
    return bool(re.search(r'(?<![A-Za-z0-9_])' + re.escape(type_name) + r'(?![A-Za-z0-9_])', rt))

import numpy as np
from sentence_transformers.util import cos_sim

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from causal_state    import CausalAgentState, VerifierSnapshot
from causal_verifier import CausalVerifier
from controller      import ControllerDecision


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers (shared with run_causal_agent bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(messages, api_key, model, max_tokens=600) -> str:
    """Thin OpenAI wrapper (used by non-cached paths)."""
    import json, time, urllib.request, urllib.error
    payload = json.dumps({
        "model": model, "messages": messages,
        "temperature": 0, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    wait = 10
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    [dispatcher rate-limit] waiting {wait}s...", flush=True)
                time.sleep(wait); wait = min(wait * 2, 120)
            else:
                return ""
        except Exception:
            return ""
    return ""


def _call_openai_with_usage(messages, api_key, model,
                             max_tokens=600) -> tuple:
    """OpenAI call that also returns (content, cached_tokens, total_input_tokens).

    cached_tokens > 0 means OpenAI hit the prefix cache (charged at 50%).
    Returns ("", 0, 0) on any error.
    """
    import json, time, urllib.request, urllib.error
    payload = json.dumps({
        "model": model, "messages": messages,
        "temperature": 0, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    wait = 10
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
            content = body["choices"][0]["message"]["content"].strip()
            usage   = body.get("usage", {})
            cached  = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            total   = usage.get("prompt_tokens", 0)
            return content, cached, total
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    [generator rate-limit] waiting {wait}s...", flush=True)
                time.sleep(wait); wait = min(wait * 2, 120)
            else:
                return "", 0, 0
        except Exception:
            return "", 0, 0
    return "", 0, 0


_TCL_GENERATION_SYSTEM_PROMPT = """\
You are an expert OpenROAD Python API programmer.
Generate a Python script that runs inside the OpenROAD interactive Python shell.

The required operation is NOT available as a Python API method.
You MUST use design.evalTclString("...") to call the equivalent Tcl command.

Rules you MUST follow:
1. `design` (openroad.Design) and `tech` (openroad.Tech) are already available.
2. For operations the Python API cannot do, use design.evalTclString("<tcl_command>").
3. Still use the Python API for object acquisition (getBlock, findInst, getNets, etc.).
4. After any lookup that can return None, immediately check and print a clear message.
5. Prints all results so they are visible in the shell output.
6. Write FLAT procedural code — no functions, no classes, no `global`.

Common OpenROAD Tcl commands for operations missing from the Python API:
  - Create instance:   create_cell <inst_name> <master_name>
  - Delete instance:   delete_cell <inst_name>
  - Connect pin:       connect_pin {<cell>/<pin>} <net_name>
  - Disconnect pin:    disconnect_pin {<cell>/<pin>}
  - Set net type:      set_net_type -net <net_name> <type>
  - Place instance:    place_cell -ref <master> -inst <inst_name> -origin {x y}
  - Rename net:        rename_net <old_name> <new_name>

Example of using evalTclString:
  design.evalTclString("create_cell clk_buf_0 BUF_X4")
  design.evalTclString("connect_pin {clk_buf_0/A} clk")

Output format — use EXACTLY these two labeled blocks:
[Diagnosis]: Explain which Python method was missing and which Tcl command replaces it.
[Code]:
<Python script — no markdown fences>
"""

_GENERATION_SYSTEM_PROMPT = """\
You are an expert OpenROAD Python API programmer.
Generate a Python script that runs inside the OpenROAD interactive Python shell.

Rules you MUST follow:
1. `design` (openroad.Design) and `tech` (openroad.Tech) are already available.
2. Acquire objects in the exact order shown in the chain — do not skip steps.
3. ALL methods are INSTANCE methods — call them on objects, never on the class.
4. Write FLAT procedural code — no functions, no classes, no `global`.
5. Prints all results so they are visible in the shell output.
6. After any lookup that can return None (findInst, findNet, findBTerm, findITerm, etc.), \
immediately check if the result is None and print a clear message, then skip the remaining steps \
using an if/else block (e.g. `if inst is None: print("Instance 'X' not found") else: ...`). \
Never call exit() or sys.exit() — they kill the OpenROAD shell. Never call methods on an unchecked lookup result.
7. If you choose an API method that differs from the provided RAG suggestions or the Causal Chain \
structure, you MUST provide a [Diagnosis]. Explain the technical reason for the deviation \
(e.g., Cardinality mismatch, Hierarchy error in RAG docs, or Object-type conflict). \
Your goal is to provide 'Correct and Executable' code, even if it contradicts the provided hints.

Output format — use EXACTLY these two labeled blocks:
[Diagnosis]: (Optional) If you are deviating from the suggested Causal Chain or RAG API, explain \
why (e.g., "Singular vs Plural mismatch" or "Hierarchy error in RAG"). Write "None" if following exactly.
[Code]:
<Python script — no markdown fences>
"""


def _parse_generation_output(raw: str):
    """Parse [Diagnosis]: ... [Code]: ... output format from the generator LLM.

    Returns (code, diagnosis). Falls back to (raw_stripped, "") when the
    labeled format is absent so old-style plain-code responses still work.
    """
    import re as _re
    diag_match = _re.search(
        r'\[Diagnosis\]\s*:\s*(.*?)(?=\[Code\])',
        raw, _re.DOTALL | _re.IGNORECASE,
    )
    code_match = _re.search(
        r'\[Code\]\s*:\s*(.*)',
        raw, _re.DOTALL | _re.IGNORECASE,
    )

    if code_match:
        code = code_match.group(1).strip()
        diag = diag_match.group(1).strip() if diag_match else ""
        diag = "" if diag.lower() in ("none", "n/a", "-") else diag
    else:
        # Fallback: treat entire response as code
        code = raw.strip()
        diag = ""

    # Strip markdown fences from code block
    if code.startswith("```"):
        code = "\n".join(
            ln for ln in code.splitlines()
            if not ln.strip().startswith("```")
        ).strip()

    return code, diag


def _build_chain_context(chain: list, edge_apis: list,
                         paths: list = None, all_edges: list = None,
                         action_node: str = "") -> str:
    """Build acquisition context for the generator. Delegates to multi-path version
    from run_causal_agent when paths/all_edges are available."""
    # Import here to avoid circular imports
    try:
        from run_causal_agent import _build_generation_context
        return _build_generation_context(
            chain, edge_apis,
            paths=paths or None,
            all_edges=all_edges or None,
            action_node=action_node,
        )
    except ImportError:
        pass
    # Fallback: simple linear context
    lines = ["Causal acquisition chain (follow this order exactly):"]
    edges = list(all_edges) if all_edges else list(zip(chain[:-1], chain[1:]))
    for i, (src, tgt) in enumerate(edges, 1):
        api = edge_apis[i - 1] if i - 1 < len(edge_apis) else None
        if api:
            raw_fn = api["function_name"]
            fn     = (raw_fn[:-1].split(".")[-1].strip() if raw_fn.endswith("(")
                      else raw_fn.split("(")[0].split(".")[-1].strip())
            params = api["parameters"].strip()
            if params.lower() == "nan":
                params = ""
            call   = f"{fn}({params})" if params else f"{fn}()"
            lines.append(f"  Step {i}: {src} -> {tgt}  |  method: {call}  |  {api['description']}")
        else:
            lines.append(f"  Step {i}: {src} -> {tgt}  |  method: UNKNOWN (use best judgment)")
    return "\n".join(lines)


def _generate_code(state: CausalAgentState, api_key: str, model: str,
                   repair_hint: str = "") -> str:
    chain_ctx = _build_chain_context(state.chain, state.edge_apis)
    user_msg  = (
        f"Task: {state.task}\n\n"
        f"{chain_ctx}\n\n"
    )
    # ── Inject lessons first (prohibitive memory from controller) ─────────────
    if state.lessons:
        lessons_str = "\n".join(f"  - {l}" for l in state.lessons)
        user_msg += (
            f"CRITICAL PROHIBITIONS — from previous failed attempts, "
            f"you MUST NOT violate these:\n{lessons_str}\n\n"
        )

    if repair_hint:
        prev_attempts = sum(1 for o in state.observations if o.action == "re_generate")
        if prev_attempts >= 2:
            # Escalate: be very explicit after repeated failures on the same issue
            user_msg += (
                f"CRITICAL — previous {prev_attempts} attempts all failed with the same error.\n"
                f"You MUST fix this exactly:\n{repair_hint}\n"
                f"Do NOT use any method not explicitly listed above. "
                f"Use only the method name shown in the chain context.\n\n"
            )
        else:
            user_msg += (
                f"IMPORTANT — fix from previous attempt:\n{repair_hint}\n\n"
            )
    user_msg += (
        f"Write the complete Python script that:\n"
        f"  1. Acquires the objects above in the exact order shown.\n"
        f"  2. Then fully implements the task — queries, modifications, prints, etc.\n"
        f"  3. Prints all results so they are visible in the shell output."
    )
    text = _call_openai(
        messages=[
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        api_key=api_key, model=model,
    )
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.splitlines()
            if not ln.strip().startswith("```")
        ).strip()
    return text


def _run_verifier(state: CausalAgentState,
                  static_verifier,
                  causal_verifier: CausalVerifier,
                  llm_verifier=None) -> VerifierSnapshot:
    """Re-run static + causal verifiers. If both pass, run LLM semantic verifier.

    The LLM result is stored in state.llm_result.
    The returned snapshot is the tightest failing result, or PASS (with LLM
    confidence baked in) when all layers agree.
    """
    from verifier import OpenROADStaticVerifier, VerifierResult

    sv     = static_verifier.verify(state.task, state.current_code, rag_context="")
    s_snap = VerifierSnapshot(
        passed=sv.passed, layer_failed=sv.layer_failed,
        issues=sv.issues, feedback=sv.feedback,
        confidence=1.0 if sv.passed else 0.0,
    )
    c_snap = causal_verifier.verify(state.current_code, state.chain, state.edge_apis,
                                    all_edges=state.all_edges or None)

    # ── Static or causal failed — skip LLM ────────────────────────────────────
    if not s_snap.passed and not c_snap.passed:
        state.llm_result = None
        return s_snap if s_snap.layer_failed <= c_snap.layer_failed else c_snap
    elif not s_snap.passed:
        state.llm_result = None
        return s_snap
    elif not c_snap.passed:
        state.llm_result = None
        return c_snap

    # ── Both static + causal passed — now run LLM semantic check ──────────────
    if llm_verifier is not None:
        chain_ctx = _build_chain_context(state.chain, state.edge_apis)
        lv_snap   = llm_verifier.verify(state.task, state.current_code, chain_ctx)
        state.llm_result = lv_snap
        if not lv_snap.passed:
            # LLM found a semantic issue — surface it (layer 5)
            print(f"  [verifier] LLM FAIL(L5) conf={lv_snap.confidence:.2f} "
                  f"issues={lv_snap.issues[:1]}", flush=True)
            return lv_snap
        print(f"  [verifier] LLM PASS conf={lv_snap.confidence:.2f}", flush=True)
    else:
        state.llm_result = None

    # ── Full PASS — merge causal warnings ──────────────────────────────────────
    return VerifierSnapshot(
        passed=True, layer_failed=0,
        issues=c_snap.issues, feedback=c_snap.feedback,
        confidence=(state.llm_result.confidence if state.llm_result else 1.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAG re-retrieval helper
# ─────────────────────────────────────────────────────────────────────────────

def _rag_requery(src: str, tgt: str, hint: str,
                 embed_model, metadata: list, embeddings,
                 top_k: int = 3, threshold: float = 0.25) -> Optional[dict]:
    """
    Re-retrieve RAG for a specific edge with an enriched query that includes
    the controller's repair_hint (e.g. the hallucinated method name) as context.
    """
    src_short = src.replace("odb.", "").replace("openroad.", "")
    tgt_short = tgt.replace("odb.", "").replace("openroad.", "")

    # Pass 1: structural filter with word-boundary matching
    strict = []
    for row in metadata:
        rt = str(row.get("Return Type:", "")).strip()
        fn = str(row.get("Function Name:", "")).strip()
        if (_rt_matches(rt, tgt_short) or _rt_matches(rt, tgt)) and \
           (src_short.lower() in fn.lower() or src.lower() in fn.lower()):
            strict.append(_row_hit(row, 1.0))
    if strict:
        return strict[0]

    loose = []
    for row in metadata:
        rt = str(row.get("Return Type:", "")).strip()
        if _rt_matches(rt, tgt_short) or _rt_matches(rt, tgt):
            loose.append(_row_hit(row, 0.9))
    if loose:
        return loose[0]

    # Pass 2: enriched semantic query using hint context
    query  = f"{src} {tgt} {src_short} {tgt_short} method {hint[:60]}"
    q_emb  = embed_model.encode(query, convert_to_tensor=True)
    scores = cos_sim(q_emb, embeddings).cpu().numpy().flatten()
    best_i = int(np.argmax(scores))
    if scores[best_i] >= threshold:
        return _row_hit(metadata[best_i], float(scores[best_i]))
    return None


def _row_hit(row: dict, score: float) -> dict:
    return {
        "description":   str(row.get("Description:", "")).strip(),
        "function_name": str(row.get("Function Name:", "")).strip(),
        "parameters":    str(row.get("Parameters:", "")).strip(),
        "return_type":   str(row.get("Return Type:", "")).strip(),
        "score":         round(score, 3),
    }


def _rag_freequery(query: str, embed_model, metadata: list, embeddings,
                   threshold: float = 0.25) -> Optional[dict]:
    """Pure semantic RAG search using a free-text query string.

    Used when the controller wants to retrieve an API for an edge that is NOT
    in the current causal chain (target_edge parse failed but rag_query is set).
    """
    q_emb  = embed_model.encode(query, convert_to_tensor=True)
    scores = cos_sim(q_emb, embeddings).cpu().numpy().flatten()
    best_i = int(np.argmax(scores))
    if scores[best_i] >= threshold:
        return _row_hit(metadata[best_i], float(scores[best_i]))
    return None


def _parse_target_edge(target_edge: str, chain: list, all_edges: list = None) -> int:
    """
    Given a 'src -> tgt' string, return the index i in all_edges (or chain edges)
    such that all_edges[i] == (src, tgt). Returns -1 if not found.
    """
    parts = [p.strip() for p in target_edge.split("->")]
    if len(parts) != 2:
        return -1
    src, tgt = parts
    edges = list(all_edges) if all_edges else list(zip(chain[:-1], chain[1:]))
    for i, (s, t) in enumerate(edges):
        if (src.lower() in s.lower() or s.lower() in src.lower()) and \
           (tgt.lower() in t.lower() or t.lower() in tgt.lower()):
            return i
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# CausalDispatcher
# ─────────────────────────────────────────────────────────────────────────────

class CausalDispatcher:
    """
    Executes controller decisions.

    Parameters
    ----------
    api_key         : OpenAI key
    model           : generation model
    embed_model     : SentenceTransformer instance
    metadata        : list of RAG row dicts
    embeddings      : encoded embeddings tensor
    static_verifier : OpenROADStaticVerifier instance
    causal_verifier : CausalVerifier instance
    code_pieces     : list of {description, code} dicts from RAGCodePiece.csv
                      (used to inject concrete examples when re-retrieving)
    """

    def __init__(self, api_key: str, model: str,
                 embed_model, metadata: list, embeddings,
                 static_verifier, causal_verifier: CausalVerifier,
                 code_pieces: Optional[list] = None,
                 llm_verifier=None):
        self.api_key         = api_key
        self.model           = model
        self.embed_model     = embed_model
        self.metadata        = metadata
        self.embeddings      = embeddings
        self.static_verifier = static_verifier
        self.causal_verifier = causal_verifier
        self.code_pieces     = code_pieces or []
        self.llm_verifier    = llm_verifier   # CausalLLMVerifier or None

        # Multi-turn conversation state for prefix caching (reset per case)
        # Structure: [system, user(task+chain_ctx)] is the stable prefix;
        # each re_generate appends [asst(prev_code), user(repair_hint)] on top.
        self._messages: list = []

    def reset_conversation(self, state: CausalAgentState) -> None:
        """Build stable [system, user(task+chain_ctx)] prefix for prefix caching.

        Call once after bootstrap and again after re_retrieve_edge updates
        edge_apis (chain context changes → old prefix is stale).
        Discards conversation history when called mid-episode.
        """
        chain_ctx  = _build_chain_context(state.chain, state.edge_apis,
                                          paths=state.paths or None,
                                          all_edges=state.all_edges or None,
                                          action_node=state.action_node)
        first_user = (
            f"Task: {state.task}\n\n"
            f"{chain_ctx}\n\n"
            f"Write the complete Python script that:\n"
            f"  1. Acquires the objects above in the exact order shown.\n"
            f"  2. Then fully implements the task — queries, modifications, prints, etc.\n"
            f"  3. Prints all results so they are visible in the shell output.\n\n"
            f"Remember: use the [Diagnosis] + [Code] output format."
        )
        self._messages = [
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            {"role": "user",   "content": first_user},
        ]

    def _call_generator(self, messages: list) -> tuple:
        """Call OpenAI for code generation with cache-hit logging.

        Returns (code_str, diagnosis_str, cached_tokens, total_input_tokens).
        Parses [Diagnosis] + [Code] labeled blocks; falls back to plain code.
        """
        raw, cached, total = _call_openai_with_usage(
            messages, self.api_key, self.model, max_tokens=800,
        )
        if cached > 0:
            print(f"    [PrefixCache] HIT {cached}/{total} tokens cached", flush=True)
        else:
            print(f"    [PrefixCache] MISS  ({total} input tokens)", flush=True)
        code, diagnosis = _parse_generation_output(raw)
        return code, diagnosis, cached, total

    def execute(self, decision: ControllerDecision,
                state: CausalAgentState) -> str:
        """Execute decision. Returns one-line observation string."""
        action = decision.next_action

        if action == "re_generate":
            return self._do_re_generate(decision, state)
        elif action == "re_retrieve_edge":
            return self._do_re_retrieve_edge(decision, state)
        elif action == "re_generate_tcl":
            return self._do_re_generate_tcl(decision, state)
        elif action == "re_extract_chain":
            return self._do_re_extract_chain(decision, state)
        elif action == "commit_best":
            return self._do_commit(decision, state, failed=False)
        elif action == "stop_fail":
            return self._do_commit(decision, state, failed=True)
        else:
            return f"[unknown action: {action}] — no-op"

    # ── re_generate ───────────────────────────────────────────────────────────

    def _do_re_generate(self, decision: ControllerDecision,
                         state: CausalAgentState) -> str:
        print(f"  [dispatch/re_generate] hint: {decision.repair_hint[:100]}", flush=True)

        # ── Lazy init: build stable prefix on first call ───────────────────────
        if not self._messages:
            self.reset_conversation(state)

        # ── Build the repair user turn ─────────────────────────────────────────
        # Lessons go here — NOT in the stable prefix (lessons change across
        # attempts; putting them in the prefix would bust the cache every step).
        repair_parts = []
        if state.lessons:
            lessons_str = "\n".join(f"  - {l}" for l in state.lessons)
            repair_parts.append(
                f"CRITICAL PROHIBITIONS — do NOT violate:\n{lessons_str}"
            )
        if decision.repair_hint:
            prev_re_generates = sum(
                1 for o in state.observations if o.action == "re_generate"
            )
            if prev_re_generates >= 2:
                repair_parts.append(
                    f"CRITICAL — {prev_re_generates} previous attempts failed "
                    f"with the same error. You MUST fix exactly:\n"
                    f"{decision.repair_hint}\n"
                    f"Use ONLY methods listed in the chain context above."
                )
            else:
                repair_parts.append(
                    f"Fix from previous attempt:\n{decision.repair_hint}"
                )
        repair_parts.append(
            "Output your response using the [Diagnosis] + [Code] format:\n"
            "[Diagnosis]: Explain any deviation from the suggested APIs (or 'None').\n"
            "[Code]:\n<corrected Python script — no markdown fences>"
        )
        repair_user = "\n\n".join(repair_parts)

        # ── Multi-turn messages: stable prefix + prior history + new turn ──────
        prev_code = state.current_code
        msgs = list(self._messages)
        if prev_code:
            msgs.append({"role": "assistant",
                         "content": f"```python\n{prev_code}\n```"})
        msgs.append({"role": "user", "content": repair_user})

        code, diagnosis, cached, _ = self._call_generator(msgs)
        if not code:
            return "re_generate: LLM returned empty code"

        if diagnosis:
            print(f"  [dispatch/re_generate] LLM Diagnosis: {diagnosis[:150]}", flush=True)

        # ── Grow conversation history for next call ────────────────────────────
        if prev_code:
            self._messages.append({"role": "assistant",
                                   "content": f"```python\n{prev_code}\n```"})
        self._messages.append({"role": "user", "content": repair_user})

        state.current_code  = code
        state.llm_diagnosis = diagnosis
        state.code_history.append(code)

        snap = _run_verifier(state, self.static_verifier, self.causal_verifier,
                             self.llm_verifier)
        state.static_result = snap
        state.maybe_update_best()

        status = "PASS" if snap.passed else f"FAIL(L{snap.layer_failed})"
        print(f"  [dispatch/re_generate] {status}  issues={snap.issues[:1]}", flush=True)
        return f"re_generate: {status}  issues={snap.issues[:1]}"

    # ── re_generate_tcl ───────────────────────────────────────────────────────

    def _do_re_generate_tcl(self, decision: ControllerDecision,
                             state: CausalAgentState) -> str:
        """Re-generate using design.evalTclString() fallback for missing Python methods.

        The controller puts the missing method name in decision.repair_hint.
        This builds a fresh prompt (separate system prompt) that tells the LLM
        to use the equivalent Tcl command via evalTclString().
        """
        bad_method = decision.repair_hint.strip()
        print(
            f"  [dispatch/re_generate_tcl] missing Python method='{bad_method}()' "
            f"— prompting for evalTclString fallback",
            flush=True,
        )

        chain_ctx = _build_chain_context(
            state.chain, state.edge_apis,
            paths=state.paths or None,
            all_edges=state.all_edges or None,
            action_node=state.action_node,
        )

        user_msg = (
            f"Task: {state.task}\n\n"
            f"{chain_ctx}\n\n"
            f"CRITICAL: The OpenROAD Python API does NOT have '{bad_method}()' "
            f"on any object type. RAG retrieval could not find a Python replacement.\n"
            f"You MUST use design.evalTclString('<tcl_command>') to accomplish "
            f"the part of the task that requires '{bad_method}'.\n\n"
        )

        if state.lessons:
            lessons_str = "\n".join(f"  - {l}" for l in state.lessons)
            user_msg += f"CRITICAL PROHIBITIONS — do NOT violate:\n{lessons_str}\n\n"

        user_msg += (
            f"Rewrite the complete Python script to:\n"
            f"  1. Acquire objects via the Python API (getBlock, findInst, getNets, etc.) "
            f"in the order shown in the chain.\n"
            f"  2. Replace the '{bad_method}()' call with the equivalent "
            f"design.evalTclString(...) call.\n"
            f"  3. Print all results so they are visible in the shell output.\n\n"
            f"Remember: use the [Diagnosis] + [Code] output format."
        )

        msgs = [
            {"role": "system", "content": _TCL_GENERATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        code, diagnosis, cached, _ = self._call_generator(msgs)
        if not code:
            return "re_generate_tcl: LLM returned empty code"

        if diagnosis:
            print(f"  [dispatch/re_generate_tcl] LLM Diagnosis: {diagnosis[:150]}",
                  flush=True)

        state.current_code  = code
        state.llm_diagnosis = diagnosis
        state.code_history.append(code)

        # Reset conversation so any subsequent re_generate starts fresh
        self.reset_conversation(state)

        snap = _run_verifier(state, self.static_verifier, self.causal_verifier,
                             self.llm_verifier)
        state.static_result = snap
        state.maybe_update_best()

        status = "PASS" if snap.passed else f"FAIL(L{snap.layer_failed})"
        print(f"  [dispatch/re_generate_tcl] {status}  issues={snap.issues[:1]}",
              flush=True)
        return f"re_generate_tcl: {status}  issues={snap.issues[:1]}"

    # ── re_extract_chain ──────────────────────────────────────────────────────

    def _find_replacement_node(self, bad_node: str,
                                state: CausalAgentState) -> tuple:
        """Find the correct replacement for bad_node using two sources:

        1. state.lessons — the controller already accumulated lessons like
           "use net.getWire() instead of getSegs()" — mine these for the
           method name, then RAG-look up its return type.
        2. RAG search on the parent node — query "parent_short getWire wire
           routing segments" to find a high-confidence method whose return
           type contains 'Wire' or routing-related names.

        Returns (replacement_type, method_name) e.g. ("odb.dbWire", "getWire")
        or ("", "") if nothing found.
        """
        import re as _re

        # Find parent of bad_node in current chain edges
        parent_type = ""
        all_edges = state.all_edges or list(zip(state.chain[:-1], state.chain[1:]))
        for src, tgt in all_edges:
            if bad_node in tgt.split(".")[-1] or tgt.split(".")[-1] in bad_node:
                parent_type = src
                break
        parent_short = parent_type.split(".")[-1]  # e.g. "dbNet"

        # ── Source 1: scan accumulated lessons for method hints ────────────
        # Lessons look like: "use net.getWire() to access wire segments"
        candidate_methods = []
        for lesson in state.lessons:
            # Extract method names mentioned alongside the parent type
            if parent_short.lower() in lesson.lower() or bad_node.lower() in lesson.lower():
                for m in _re.findall(r'\.(\w+)\(\)', lesson):
                    candidate_methods.append(m)
        # De-duplicate, preserve order
        seen = set()
        candidate_methods = [m for m in candidate_methods
                              if not (m in seen or seen.add(m))]

        # ── Source 2: RAG lookup for each candidate method ─────────────────
        for method in candidate_methods:
            # Search RAG for an entry whose function_name contains both
            # parent_short and method, to find its return type
            for row in self.metadata:
                fn = str(row.get("Function Name:", "")).strip()
                rt = str(row.get("Return Type:", "")).strip()
                if (parent_short.lower() in fn.lower()
                        and method.lower() in fn.lower()
                        and rt and rt.lower() != "nan"):
                    # Found a RAG entry: e.g. odb.dbNet.getWire → return odb.dbWire
                    # Normalise to qualified name
                    if "." not in rt:
                        rt = f"odb.{rt}"
                    print(
                        f"  [dispatch/re_extract_chain] replacement found via lesson: "
                        f"{parent_short}.{method}() → {rt}",
                        flush=True,
                    )
                    return rt, method

        # ── Source 3: semantic RAG query on parent + "wire routing segments" ─
        if parent_short:
            query = f"{parent_short} wire routing segments method return type"
            hit = _rag_freequery(
                query,
                embed_model=self.embed_model,
                metadata=self.metadata,
                embeddings=self.embeddings,
                threshold=0.40,
            )
            if hit:
                fn  = hit["function_name"]
                rt  = hit["return_type"]
                method_name = fn.split("(")[0].split(".")[-1].strip()
                if rt and rt.lower() != "nan" and "." in fn and parent_short.lower() in fn.lower():
                    if "." not in rt:
                        rt = f"odb.{rt}"
                    print(
                        f"  [dispatch/re_extract_chain] replacement found via RAG: "
                        f"{fn} → {rt}",
                        flush=True,
                    )
                    return rt, method_name

        return "", ""

    def _do_re_extract_chain(self, decision: ControllerDecision,
                              state: CausalAgentState) -> str:
        """Re-extract the causal chain after detecting a bad intermediate node.

        Steps:
          1. Discover the correct replacement node via lessons + RAG.
          2. Build a positive correction hint (not just "X doesn't exist" but
             "replace X with Y via method()").
          3. Re-run bootstrap_causal_extract with that hint.
          4. Re-run bootstrap_causal_rag    → new edge_apis.
          5. Reset conversation prefix      → chain context updated.
          6. Re-generate + re-verify.
        controller.repair_hint holds the bad node name (e.g. "dbSeg").
        """
        bad_node = decision.repair_hint.strip()
        print(
            f"  [dispatch/re_extract_chain] bad node='{bad_node}' — "
            f"discovering replacement node before re-extracting chain",
            flush=True,
        )

        try:
            from run_causal_agent import bootstrap_causal_extract, bootstrap_causal_rag
        except ImportError as e:
            print(f"  [dispatch/re_extract_chain] import error: {e} — falling back to re_generate",
                  flush=True)
            return self._do_re_generate(decision, state)

        # ── Discover the positive replacement for bad_node ─────────────────
        replacement_type, replacement_method = self._find_replacement_node(bad_node, state)

        # ── Build correction hint ──────────────────────────────────────────
        if bad_node and replacement_type:
            hint_suffix = (
                f"\n\n[CORRECTION] The type '{bad_node}' does NOT exist in the "
                f"OpenROAD Python API. "
                f"The correct intermediate object is '{replacement_type}', "
                f"accessed via {replacement_method}(). "
                f"Update the acquisition path to use "
                f"'... -> {replacement_type}' instead of '... -> odb.{bad_node}'."
            )
            print(
                f"  [dispatch/re_extract_chain] hint: replace '{bad_node}' "
                f"with '{replacement_type}' via {replacement_method}()",
                flush=True,
            )
        elif bad_node:
            hint_suffix = (
                f"\n\n[CORRECTION] The type '{bad_node}' does NOT exist in the "
                f"OpenROAD Python API. Do NOT include '{bad_node}' in any acquisition "
                f"path. Find the correct intermediate object that actually exists."
            )
            print(
                f"  [dispatch/re_extract_chain] no replacement found — "
                f"using negative-only hint",
                flush=True,
            )
        else:
            hint_suffix = (
                f"\n\n[CORRECTION] The previous causal chain contained a non-existent "
                f"type. Re-derive the chain using only real OpenROAD Python types."
            )

        original_task = state.task
        state.task    = original_task + hint_suffix

        # ── Re-extract chain ───────────────────────────────────────────────
        old_chain = list(state.chain)
        bootstrap_causal_extract(state, self.api_key, self.model)
        state.task = original_task   # restore clean task before generation

        if not state.chain or state.chain == old_chain:
            print(
                f"  [dispatch/re_extract_chain] chain unchanged after re-extraction "
                f"— falling back to re_generate",
                flush=True,
            )
            decision.repair_hint = (
                f"Chain re-extraction did not improve the path. "
                f"Do NOT use '{bad_node}'. "
                + (f"Use '{replacement_type}' via {replacement_method}() instead. "
                   if replacement_type else
                   "Use only methods from the known-methods list in the verifier.")
            )
            return self._do_re_generate(decision, state)

        new_chain_str = " -> ".join(state.chain)
        print(f"  [dispatch/re_extract_chain] new chain: {new_chain_str}", flush=True)

        # ── Re-run RAG with new chain ──────────────────────────────────────
        bootstrap_causal_rag(state, self.embed_model, self.metadata, self.embeddings)

        # Reset conversation so next re_generate uses updated chain context
        self.reset_conversation(state)

        # Re-generate with fresh chain
        decision.repair_hint = (
            f"The causal chain has been rewritten — '{bad_node}' was removed"
            + (f" and replaced with '{replacement_type}' via {replacement_method}()"
               if replacement_type else "")
            + f". Use ONLY the new acquisition chain shown above."
        )
        gen_obs = self._do_re_generate(decision, state)
        return f"re_extract_chain: new_chain=[{new_chain_str}] | {gen_obs}"

    # ── re_retrieve_edge ──────────────────────────────────────────────────────

    def _do_re_retrieve_edge(self, decision: ControllerDecision,
                              state: CausalAgentState) -> str:
        edge_idx = _parse_target_edge(decision.target_edge, state.chain, state.all_edges or None)
        if edge_idx < 0:
            # Edge not in current chain — try free-text RAG query if provided
            if decision.rag_query.strip():
                print(f"  [dispatch/re_retrieve_edge] edge not in chain — "
                      f"free-text query: '{decision.rag_query[:80]}'", flush=True)
                hit = _rag_freequery(
                    decision.rag_query,
                    embed_model=self.embed_model,
                    metadata=self.metadata,
                    embeddings=self.embeddings,
                )
                if hit:
                    method = hit["function_name"].split("(")[0].split(".")[-1].strip()
                    print(f"  [dispatch/re_retrieve_edge] free-query hit: "
                          f"{hit['function_name']} (score={hit['score']})", flush=True)
                    decision.repair_hint = (
                        f"RAG found for '{decision.rag_query[:60]}': "
                        f"use {method}() — {hit['description']}. "
                        + decision.repair_hint
                    )
                else:
                    print(f"  [dispatch/re_retrieve_edge] free-query no hit — "
                          f"falling back to re_generate", flush=True)
                return self._do_re_generate(decision, state)
            print(f"  [dispatch/re_retrieve_edge] could not parse target_edge "
                  f"'{decision.target_edge}' and no rag_query — falling back to re_generate",
                  flush=True)
            return self._do_re_generate(decision, state)

        edges = state.all_edges if state.all_edges else list(zip(state.chain[:-1], state.chain[1:]))
        src, tgt = edges[edge_idx]
        print(f"  [dispatch/re_retrieve_edge] edge [{src}] -> [{tgt}]", flush=True)

        # Build enriched hint from api_diffs if available
        api_diff = None
        if state.static_result and state.static_result.api_diffs:
            for d in state.static_result.api_diffs:
                if (src.lower() in d.src_type.lower() or d.src_type.lower() in src.lower()) and \
                   (tgt.lower() in d.tgt_type.lower() or d.tgt_type.lower() in tgt.lower()):
                    api_diff = d
                    break

        if api_diff:
            hallucinated = ", ".join(api_diff.code_methods)
            rag_says     = api_diff.rag_method
            hint_prefix  = (
                f"API hallucination on {src.split('.')[-1]}→{tgt.split('.')[-1]}: "
                f"code used [{hallucinated}] but RAG says '{rag_says}'. "
                f"Do NOT use {hallucinated}. "
            )
            print(f"  [dispatch/re_retrieve_edge] diff: code=[{hallucinated}] "
                  f"rag='{rag_says}'", flush=True)
        else:
            hint_prefix = ""

        hit = _rag_requery(
            src, tgt,
            hint=hint_prefix + decision.repair_hint,
            embed_model=self.embed_model,
            metadata=self.metadata,
            embeddings=self.embeddings,
        )

        if hit:
            method = hit["function_name"].split("(")[0].split(".")[-1].strip()
            print(f"  [dispatch/re_retrieve_edge] new hit: {hit['function_name']} "
                  f"(score={hit['score']})", flush=True)
            decision.repair_hint = (
                hint_prefix +
                f"For {src.split('.')[-1]}→{tgt.split('.')[-1]}: "
                f"use {method}() — {hit['description']}"
            )
            # Ensure edge_apis list is long enough
            while len(state.edge_apis) <= edge_idx:
                state.edge_apis.append(None)
            state.edge_apis[edge_idx] = hit
            # Chain context changed → rebuild stable prefix so next re_generate
            # caches the updated chain description, not the stale one.
            self.reset_conversation(state)
            rag_obs = f"RAG updated edge[{edge_idx}]: {method}"
        else:
            print(f"  [dispatch/re_retrieve_edge] no RAG hit — injecting code examples",
                  flush=True)
            # Inject relevant code examples from RAGCodePiece.csv as fallback
            examples = self._find_code_examples(src, tgt, api_diff)
            decision.repair_hint = hint_prefix
            if examples:
                decision.repair_hint += f"\nCode examples for reference:\n{examples}"
            rag_obs = f"RAG no hit for edge[{edge_idx}]"

        # Always re-generate after retrieval
        gen_obs = self._do_re_generate(decision, state)
        return f"{rag_obs} | {gen_obs}"

    def _find_code_examples(self, src: str, tgt: str, api_diff) -> str:
        """Find relevant code snippets from RAGCodePiece.csv for the given edge."""
        if not self.code_pieces:
            return ""
        src_short = src.split(".")[-1].lower()
        tgt_short = tgt.split(".")[-1].lower()

        # Keywords to match: type names + hallucinated method (if known)
        keywords = {src_short, tgt_short}
        if api_diff and api_diff.code_methods:
            keywords.update(m.lower() for m in api_diff.code_methods)
        if api_diff and api_diff.rag_method and api_diff.rag_method != "<no RAG hit>":
            keywords.add(api_diff.rag_method.lower())

        matches = []
        for piece in self.code_pieces:
            desc = str(piece.get("description", "")).lower()
            code = str(piece.get("code", ""))
            code_lower = code.lower()
            if any(kw in desc or kw in code_lower for kw in keywords):
                matches.append(f"# Example ({piece.get('description', '')[:60]})\n{code[:300]}")
            if len(matches) >= 2:
                break

        return "\n\n".join(matches)

    # ── commit / stop ─────────────────────────────────────────────────────────

    def _do_commit(self, decision: ControllerDecision,
                    state: CausalAgentState, failed: bool) -> str:
        code = state.best_code or state.current_code
        state.committed_code = code
        state.committed      = True
        tag = "stop_fail" if failed else "commit_best"
        print(f"  [dispatch/{tag}] committing "
              f"{'best' if state.best_code else 'current'} code "
              f"(score={state.best_score:.3f})", flush=True)
        return f"{tag}: committed code (score={state.best_score:.3f})"
