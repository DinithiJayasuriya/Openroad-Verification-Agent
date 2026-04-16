"""run_causal_agent.py — Causal agentic orchestrator.

Architecture (mirrors src_1_reflector/run_agent.py):

  Bootstrap (always runs, free — no budget cost):
    1. causal_extract   — task text → causal chain (GPT-4.1-mini)
    2. causal_rag       — per-edge structural + semantic RAG retrieval
    3. causal_generate  — build skeleton + GPT code completion
    4. static_verify    — static verifier on generated code

  Controller loop (TBD — placeholder stub below):
    • Reads full CausalAgentState, decides next action, executes it.
    • Each action costs 1 unit of budget.

  Execution (once, after commit):
    • Best committed code is run in a live OpenROAD shell.

Usage:
  conda activate prompt
  python causal_verifier/run_causal_agent.py \\
      --dataset      src_1_reflector/dataset4.xlsx \\
      --ragApi       RAGData/RAGAPIs.csv \\
      --ragCodePiece RAGData/RAGCodePiece.csv \\
      --openaiKey    sk-... \\
      --openroadPath OpenROAD/build/src/openroad \\
      --runDir       src_1_reflector \\
      --numCases     5 \\
      --budget       6
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple


def _clean_output(text: str) -> str:
    """Strip ANSI escape sequences and VS Code shell-integration sequences from PTY output."""
    # VS Code OSC 633 sequences: \x1b]633;...\x07
    text = re.sub(r'\x1b\][^\x07]*\x07', '', text)
    # Generic ANSI escape sequences: ESC [ ... m  and others
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    # Strip stray control chars (SOH \x01, STX \x02, etc.) but keep \r\n
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ── path setup ─────────────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAUSAL_DIR  = os.path.dirname(os.path.abspath(__file__))
# causal_verifier/ must come first so its controller/dispatcher shadow src_1_reflector's
sys.path.insert(0, os.path.join(_ROOT, "src_1_reflector"))
sys.path.insert(0, os.path.join(_ROOT, "src_1_agentic"))
sys.path.insert(0, _CAUSAL_DIR)

from util import (
    runOpenROADShell, sendCommandOpenROAD, processCodeString,
    clearQueue, readOpenROADOutput,
)
from verifier import OpenROADStaticVerifier, VerifierResult

from causal_state        import CausalAgentState, VerifierSnapshot
from causal_verifier     import CausalVerifier
from controller          import CausalController
from dispatcher          import CausalDispatcher, _parse_generation_output
from llm_verifier        import CausalLLMVerifier
from structured_rag_gate import StructuredRAGGate


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Chain extraction
# ─────────────────────────────────────────────────────────────────────────────

CHAIN_SYSTEM_PROMPT = """\
You are an OpenROAD Python API expert.

Given a task, identify all INDEPENDENT terminal objects needed to complete it.
For each terminal object, provide the shortest valid acquisition path from openroad.Design.
Then state the Action Node that combines them.

RULES:
- Entry point is always openroad.Design (variable `design` is pre-available).
- To reach odb.dbBlock: design.getBlock() — NO dbChip step in the Python API.
- To reach odb.dbDatabase: design.getDb().
- Physical pin shapes of block ports are odb.dbBPin — NOT odb.dbPin, NOT odb.dbBPins.
  Access via: odb.dbBTerm -> odb.dbBPin  (method: bterm.getBPins()).
  odb.dbBPin has getPlacementStatus() returning a string ('FIRM','PLACED', etc.) and getLocation().
  odb.dbBPin does NOT have getName(). Port names come from bterm.getName(), NOT from the bpin.
- Each Path ends at a DISTINCT terminal object required for the action.
- If two objects come from the same parent (e.g. both from dbBlock), list them as SEPARATE paths.
- Use exact OpenROAD Python type names.
- LEAF-ACTION RULE: If the task requires calling a specific method ON the leaf object
  (e.g. checking a property, reading a flag, calling a named function on it),
  extend the path by ONE more edge using bracket notation: -> [methodName].
  This tells the retriever to look up that exact method in the API database.
  Only add a leaf-action edge when the task names the operation explicitly
  (e.g. "check direction", "call isOutputSignal", "call setPlacementStatus").
  Do NOT add leaf-action edges for generic iteration or printing.

OUTPUT FORMAT (no markdown, no JSON):
Path 1 (<TerminalType>): openroad.Design -> ... -> <TerminalType>
Path 2 (<TerminalType>): openroad.Design -> ... -> <TerminalType>
Action Node: <action>(<TerminalType1>, <TerminalType2>)

EXAMPLES:

Task: "Find instance named u_cpu"
Path 1 (dbInst): openroad.Design -> odb.dbBlock -> odb.dbInst
Action Node: findInst(dbInst)

Task: "Get all nets in the design"
Path 1 (dbNet): openroad.Design -> odb.dbBlock -> odb.dbNet
Action Node: getNets(dbNet)

Task: "Connect port 'req_val' to net 'req_msg[17]'"
Path 1 (dbBTerm): openroad.Design -> odb.dbBlock -> odb.dbBTerm
Path 2 (dbNet): openroad.Design -> odb.dbBlock -> odb.dbNet
Action Node: connect(dbBTerm, dbNet)

Task: "Swap master of instance '_486_' to 'INV_X4'"
Path 1 (dbInst): openroad.Design -> odb.dbBlock -> odb.dbInst
Path 2 (dbMaster): openroad.Design -> odb.dbDatabase -> odb.dbMaster
Action Node: swapMaster(dbInst, dbMaster)

Task: "Find which net is connected to pin A of instance inv_1"
Path 1 (dbNet): openroad.Design -> odb.dbBlock -> odb.dbInst -> odb.dbITerm -> odb.dbNet
Action Node: getNet(dbITerm)

Task: "Run global placement"
Path 1 (Replace): openroad.Design -> gpl.Replace
Action Node: doInitialPlace(Replace)

Task: "Iterate all instances and count ITerms with direction OUTPUT; find the instance with the most output pins"
Path 1 (dbITerm): openroad.Design -> odb.dbBlock -> odb.dbInst -> odb.dbITerm -> [isOutputSignal]
Action Node: countOutputPins(dbInst, dbITerm)

Task: "Print the placement status of each instance"
Path 1 (dbInst): openroad.Design -> odb.dbBlock -> odb.dbInst -> [getPlacementStatus]
Action Node: printPlacementStatus(dbInst)

Task: "Find block ports whose first pins are firmly placed"
Path 1 (dbBPin): openroad.Design -> odb.dbBlock -> odb.dbBTerm -> odb.dbBPin -> [[getPlacementStatus]]
Action Node: checkFirmPlacement(dbBTerm, dbBPin)

Task: "Find block ports whose first pin names start with 'clk' and confirm they are firmly placed"
Path 1 (dbBTerm): openroad.Design -> odb.dbBlock -> odb.dbBTerm
Path 2 (dbBPin): openroad.Design -> odb.dbBlock -> odb.dbBTerm -> odb.dbBPin -> [[getPlacementStatus]]
Action Node: findFirmClkPorts(dbBTerm, dbBPin)
"""


def _parse_multi_path_chain(text: str):
    """Parse multi-path chain extraction output.

    Returns (paths, action_node) where paths is a list of type-name lists.
    Falls back to JSON-array parsing for backward compatibility.
    """
    paths = []
    action_node = ""

    for line in text.splitlines():
        line = line.strip()
        # Match "Path N (TypeName): A -> B -> C"
        pm = re.match(r'Path\s+\d+\s*\([^)]*\)\s*:\s*(.+)', line, re.IGNORECASE)
        if pm:
            types = [t.strip() for t in pm.group(1).split('->') if t.strip()]
            if len(types) >= 2:
                paths.append(types)
        # Match "Action Node: ..."
        am = re.match(r'Action\s+Node\s*:\s*(.+)', line, re.IGNORECASE)
        if am:
            action_node = am.group(1).strip()

    if paths:
        return paths, action_node

    # Fallback: try JSON array (old format)
    try:
        chain = json.loads(text)
        if isinstance(chain, list) and all(isinstance(t, str) for t in chain):
            return [chain], ""
    except Exception:
        pass
    return [], ""


def _call_openai(messages: list, api_key: str, model: str,
                 temperature: float = 0, max_tokens: int = 200) -> str:
    """Low-level OpenAI call. Returns message content string or ''."""
    payload = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (2 ** attempt)
                print(f"    [rate limit] waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    [HTTP {e.code}] {e.reason}", flush=True)
                return ""
        except Exception as exc:
            print(f"    [openai error] {exc}", flush=True)
            return ""
    return ""


def bootstrap_causal_extract(state: CausalAgentState, api_key: str, model: str) -> None:
    """Bootstrap step 1: extract causal paths from task. Updates state.paths/chain/all_edges."""
    text = _call_openai(
        messages=[
            {"role": "system", "content": CHAIN_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Task: {state.task}"},
        ],
        api_key=api_key, model=model, max_tokens=250,
    )

    paths, action_node = _parse_multi_path_chain(text)

    if paths:
        state.paths       = paths
        state.action_node = action_node
        state.chain       = paths[0]   # primary path for backward compat

        # Build all_edges: unique (src, tgt) pairs across all paths, preserving order
        seen  = set()
        edges = []
        for path in paths:
            for src, tgt in zip(path[:-1], path[1:]):
                if (src, tgt) not in seen:
                    seen.add((src, tgt))
                    edges.append((src, tgt))
        state.all_edges = edges

        path_strs = " | ".join(" -> ".join(f"[{t}]" for t in p) for p in paths)
        result    = path_strs + (f"  Action: {action_node}" if action_node else "")
    else:
        state.chain = []
        state.paths = []
        state.all_edges = []
        result = f"[FAILED: {text[:80]}]"

    state.add_bootstrap_obs("causal_extract", result)
    print(f"  [boot/causal_extract] {result}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b — Node Gate: validate every type in the extracted chain
# ─────────────────────────────────────────────────────────────────────────────

_MAX_NODE_REWRITES = 2


def _rebuild_state_paths(state: CausalAgentState, paths: list, action_node: str) -> None:
    """Update state.paths/chain/all_edges from a new paths list."""
    state.paths       = paths
    state.action_node = action_node
    state.chain       = paths[0] if paths else []
    seen, edges = set(), []
    for path in paths:
        for src, tgt in zip(path[:-1], path[1:]):
            if (src, tgt) not in seen:
                seen.add((src, tgt))
                edges.append((src, tgt))
    state.all_edges = edges


def bootstrap_node_gate(state: CausalAgentState,
                        gate: StructuredRAGGate,
                        api_key: str, model: str) -> None:
    """Bootstrap step 1b: validate every type node against the structured RAG.

    For each type in the extracted causal chain:
      VALID        → proceed as-is
      RAG_MISS     → real OpenROAD type missing from RAG; gate auto-appends a
                     placeholder row to the structured CSV
      HALLUCINATION→ LLM invented this type; re-extract with corrective feedback
                     (up to _MAX_NODE_REWRITES attempts)

    Updates state.paths / chain / all_edges in place.
    """
    if gate is None or not state.paths:
        return

    for attempt in range(_MAX_NODE_REWRITES + 1):
        report = gate.validate(state.paths, task=state.task)

        rag_miss_types = report.rag_miss_types()
        if rag_miss_types:
            print(f"  [node_gate] RAG_MISS types (appended to CSV): {rag_miss_types}",
                  flush=True)

        if not report.had_hallucinations:
            status = "ALL_VALID" if not rag_miss_types else f"VALID+RAG_MISS({len(rag_miss_types)})"
            state.add_bootstrap_obs("node_gate", status)
            print(f"  [node_gate] {status}", flush=True)
            return

        halluc = report.hallucinated_types()
        print(f"  [node_gate] HALLUCINATION(s) detected (attempt {attempt+1}/"
              f"{_MAX_NODE_REWRITES+1}): {[h[0] for h in halluc]}", flush=True)

        if attempt >= _MAX_NODE_REWRITES:
            # Exhausted rewrites — log and continue with best effort
            status = f"HALLUCINATION_UNFIXED({[h[0] for h in halluc]})"
            state.add_bootstrap_obs("node_gate", status)
            print(f"  [node_gate] {status} — proceeding anyway", flush=True)
            return

        # Re-extract with corrective feedback
        user_content = (
            f"CORRECTION REQUIRED:\n{report.rewrite_feedback}\n\n"
            f"Re-extract the causal chain fixing the invalid types.\n\n"
            f"Task: {state.task}"
        )
        raw = _call_openai(
            messages=[
                {"role": "system", "content": CHAIN_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            api_key=api_key, model=model, max_tokens=300,
        )
        new_paths, new_action = _parse_multi_path_chain(raw)
        if new_paths:
            _rebuild_state_paths(state, new_paths, new_action)
            path_str = " | ".join(" -> ".join(p) for p in new_paths)
            print(f"  [node_gate] rewritten chain: {path_str}", flush=True)
        else:
            print(f"  [node_gate] re-extraction empty — keeping current chain", flush=True)
            break


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Node-specific RAG (type-restricted + semantic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_hit(row: dict, score: float) -> dict:
    return {
        "description":   str(row.get("Description:", "")).strip(),
        "function_name": str(row.get("Function Name:", "")).strip(),
        "parameters":    str(row.get("Parameters:", "")).strip(),
        "return_type":   str(row.get("Return Type:", "")).strip(),
        "score":         round(score, 3),
    }


def _rt_matches(rt: str, type_name: str) -> bool:
    """
    Word-boundary match for a type name in a return-type string.
    Prevents 'dbTechLayer' from matching 'dbTechLayerCutClassRule' as a substring.
    """
    return bool(re.search(r'(?<![A-Za-z0-9_])' + re.escape(type_name) + r'(?![A-Za-z0-9_])', rt))


_LEAF_THRESHOLD = 0.65   # raised from 0.40 — reject low-confidence leaf hits


def _rag_query_for_edge(src: str, tgt: str,
                        embed_model, metadata: list, embeddings,
                        top_k: int = 3, threshold: float = 0.40,
                        api_key: str = "", model: str = "",
                        task: str = "") -> Optional[dict]:
    """Two-pass retrieval for one chain edge. Returns best hit dict or None.

    Leaf-action edges have tgt in [method] bracket notation, e.g. [isOutputSignal].
    For these, retrieval switches to name-based lookup on src type instead of
    return-type-based lookup.

    When api_key/model/task are provided and a leaf-action edge gets no confident
    hit (score < _LEAF_THRESHOLD), calls _rag_miss_requery() to ask the LLM for
    better search phrases and retries. Returns a RAG-MISS sentinel dict if the
    requery also fails.
    """
    # ── Leaf-action edge: tgt is [methodName] ──────────────────────────────────
    if tgt.startswith("[") and tgt.endswith("]"):
        method_name   = tgt[1:-1]
        src_short     = src.replace("odb.", "").replace("openroad.", "")
        leaf_threshold = _LEAF_THRESHOLD  # tighter than normal edges

        # Pass 1: exact name match on the source type's method
        for row in metadata:
            fn = str(row.get("Function Name:", "")).strip()
            if src_short.lower() in fn.lower() and method_name.lower() in fn.lower():
                return _row_to_hit(row, 1.0)

        # Pass 1b: type-scoped semantic — restrict to rows whose function name
        # contains src_short before searching the full corpus.  Prevents an
        # unrelated entry winning when the LLM hallucinated the method name.
        query  = f"{src_short} {method_name} {src}"
        q_emb  = embed_model.encode(query, convert_to_tensor=True)
        scoped_indices = [
            i for i, row in enumerate(metadata)
            if src_short.lower() in str(row.get("Function Name:", "")).lower()
        ]
        if scoped_indices:
            scoped_embs   = embeddings[scoped_indices]
            scoped_scores = cos_sim(q_emb, scoped_embs).cpu().numpy().flatten()
            best_j        = int(np.argmax(scoped_scores))
            if scoped_scores[best_j] >= leaf_threshold:
                return _row_to_hit(metadata[scoped_indices[best_j]],
                                   float(scoped_scores[best_j]))
        else:
            # src_type has ZERO entries in the API db — it is an invented type.
            # Global fallback would find an unrelated hit. Skip it and go straight
            # to LLM requery (or MISS sentinel).
            print(f"  [boot/causal_rag] [{src}] -> [{tgt}]  src type '{src_short}' not in API db — skipping global fallback", flush=True)
            if api_key and model:
                return _rag_miss_requery(src, method_name, task,
                                         api_key, model,
                                         embed_model, metadata, embeddings)
            return None

        # Pass 2: global semantic fallback — only reached when type IS known but
        # no scoped hit was confident enough.
        scores = cos_sim(q_emb, embeddings).cpu().numpy().flatten()
        best_i = int(np.argmax(scores))
        if scores[best_i] >= leaf_threshold:
            return _row_to_hit(metadata[best_i], float(scores[best_i]))

        # ── RAG MISS — escalate to LLM requery if credentials provided ─────────
        if api_key and model:
            print(f"  [boot/causal_rag] [{src}] -> [{tgt}]  MISS (max={scores[best_i]:.2f} < {leaf_threshold}) — LLM requery...", flush=True)
            return _rag_miss_requery(src, method_name, task,
                                     api_key, model,
                                     embed_model, metadata, embeddings)
        return None

    # ── Normal acquisition edge: tgt is a type name ────────────────────────────
    src_short = src.replace("odb.", "").replace("openroad.", "")
    tgt_short = tgt.replace("odb.", "").replace("openroad.", "")

    # Pass 1a: strict — return_type matches tgt (word-boundary) AND function_name contains src
    strict = []
    for row in metadata:
        rt = str(row.get("Return Type:", "")).strip()
        fn = str(row.get("Function Name:", "")).strip()
        if (_rt_matches(rt, tgt_short) or _rt_matches(rt, tgt)) and \
           (src_short.lower() in fn.lower() or src.lower() in fn.lower()):
            strict.append(_row_to_hit(row, 1.0))
    if strict:
        return strict[0]

    # Pass 1b: loose — return_type matches tgt (word-boundary) only
    loose = []
    for row in metadata:
        rt = str(row.get("Return Type:", "")).strip()
        if _rt_matches(rt, tgt_short) or _rt_matches(rt, tgt):
            loose.append(_row_to_hit(row, 0.9))
    if loose:
        return loose[0]

    # Pass 2: semantic fallback
    query  = f"{src} get {tgt} {src_short} {tgt_short} method"
    q_emb  = embed_model.encode(query, convert_to_tensor=True)
    scores = cos_sim(q_emb, embeddings).cpu().numpy().flatten()
    best_i = int(np.argmax(scores))
    if scores[best_i] >= threshold:
        return _row_to_hit(metadata[best_i], float(scores[best_i]))
    return None


def _rag_miss_requery(src: str, method_name: str, task: str,
                      api_key: str, model: str,
                      embed_model, metadata: list, embeddings,
                      requery_threshold: float = 0.55) -> dict:
    """Called when a leaf-action edge has no confident RAG hit (score < _LEAF_THRESHOLD).

    Asks the LLM to reformulate the operation into 2-3 concrete API search phrases,
    then re-queries RAG with each phrase (type-scoped first, global fallback).

    Returns the best hit dict if any phrase scores >= requery_threshold,
    otherwise returns a RAG-MISS sentinel: {"rag_miss": True, "method_name": ..., "src_type": ...}
    """
    src_short = src.replace("odb.", "").replace("openroad.", "")
    prompt = (
        f"You are an OpenROAD Python API expert.\n"
        f"A semantic search for method '[[{method_name}]]' on type '{src}' "
        f"found no match in the API database.\n"
        f"Task context: {task[:300]}\n\n"
        f"Give 2-3 short, specific API method names or short descriptions that could "
        f"help find the real API for this operation on {src}. "
        f"Focus on what the method actually DOES (e.g. 'getPlacementStatus', "
        f"'check placement firmly placed', 'getBPins placement'). "
        f"Output one phrase per line, nothing else."
    )
    raw = _call_openai(
        messages=[
            {"role": "system", "content": "You are an OpenROAD API expert. Output only concise search phrases, one per line."},
            {"role": "user",   "content": prompt},
        ],
        api_key=api_key, model=model, max_tokens=80,
    )
    phrases = [p.strip().lstrip("-•* ") for p in raw.splitlines() if p.strip()]
    if not phrases:
        print(f"  [rag_miss_requery] LLM returned no phrases — injecting MISS sentinel", flush=True)
        return {"rag_miss": True, "method_name": method_name, "src_type": src}

    print(f"  [rag_miss_requery] LLM phrases: {phrases}", flush=True)

    best_hit   = None
    best_score = 0.0

    # Pre-compute scoped indices once — if empty, the src type is invented;
    # global fallback would only find unrelated hits so skip it entirely.
    scoped_indices = [
        i for i, row in enumerate(metadata)
        if src_short.lower() in str(row.get("Function Name:", "")).lower()
    ]
    type_known = len(scoped_indices) > 0

    for phrase in phrases[:3]:
        q_emb = embed_model.encode(phrase, convert_to_tensor=True)

        # Type-scoped search
        if scoped_indices:
            scoped_embs   = embeddings[scoped_indices]
            scoped_scores = cos_sim(q_emb, scoped_embs).cpu().numpy().flatten()
            best_j        = int(np.argmax(scoped_scores))
            if scoped_scores[best_j] >= requery_threshold and scoped_scores[best_j] > best_score:
                best_hit   = _row_to_hit(metadata[scoped_indices[best_j]],
                                         float(scoped_scores[best_j]))
                best_score = float(scoped_scores[best_j])

        # Global fallback — only when type IS known in the API db.
        # If scoped_indices is empty the type is invented; global search
        # will find an unrelated entry and poison generation context.
        if type_known:
            scores = cos_sim(q_emb, embeddings).cpu().numpy().flatten()
            best_i = int(np.argmax(scores))
            if scores[best_i] >= requery_threshold and scores[best_i] > best_score:
                best_hit   = _row_to_hit(metadata[best_i], float(scores[best_i]))
                best_score = float(scores[best_i])

    if best_hit:
        print(f"  [rag_miss_requery] found: {best_hit['function_name']}  (score={best_score:.2f})", flush=True)
        return best_hit

    print(f"  [rag_miss_requery] no hit above {requery_threshold} — injecting MISS sentinel", flush=True)
    return {"rag_miss": True, "method_name": method_name, "src_type": src}


# ── Type-restricted edge lookup (structured CSV) ──────────────────────────────

# Confidence thresholds for edge enforcement
_CONF_MANDATORY  = 0.85   # verifier mandates this method; repair if code deviates
_CONF_SUGGESTED  = 0.70   # soft hint; generator uses it but verifier won't hard-fail
# Below _CONF_SUGGESTED → MISS sentinel; generator must not guess


def _confidence_status(score: float) -> str:
    """Map a RAG confidence score to an enforcement tier."""
    if score >= _CONF_MANDATORY:
        return "MANDATORY"
    elif score >= _CONF_SUGGESTED:
        return "SUGGESTED"
    else:
        return "MISS"


def _structured_row_to_hit(row: dict, score: float) -> dict:
    """Convert a RAGAPIs_structured.csv row dict to a hit dict."""
    return {
        "description":   str(row.get("Description", "")).strip(),
        "function_name": str(row.get("Method Name",  "")).strip(),
        "parameters":    str(row.get("Parameters",   "")).strip(),
        "return_type":   str(row.get("Return Type",  "")).strip(),
        "receiver_type": str(row.get("Receiver Type","")).strip(),
        "score":         round(score, 3),
        "status":        _confidence_status(score),
        "source":        "structured",
    }


def _rag_query_for_edge_typed(src: str, tgt: str,
                               structured_df,          # pd.DataFrame from RAGAPIs_structured.csv
                               embed_model,
                               task: str = "") -> Optional[dict]:
    """Type-restricted edge lookup using RAGAPIs_structured.csv.

    Principle: a method is only 'correct' if it belongs to the confirmed
    Receiver Type.  All lookups are first filtered to rows where
    Receiver Type == src before any matching is attempted.

    Normal edge  src → tgt_type :
        Filter by Receiver Type == src.
        Find row where Return Type matches tgt_type (word-boundary).
        Fallback: semantic search within the filtered subset.

    Leaf edge    src → [methodName] :
        Filter by Receiver Type == src.
        Find row where Method Name == methodName (exact, then fuzzy).
        Fallback: semantic search within the filtered subset.

    Returns:
        hit dict  — found; "source": "structured"
        None      — type present in RAG but no matching method/return found
        {"rag_miss": True, ...}  — src type has NO rows in structured RAG at all
    """
    import pandas as pd
    from sentence_transformers.util import cos_sim as _cos_sim

    src_short = src.rsplit(".", 1)[-1]

    # ── Filter structured CSV to Receiver Type == src ──────────────────────
    def _type_matches(v: str) -> bool:
        v = v.strip()
        return v == src or v == src_short or v.rsplit(".", 1)[-1] == src_short

    mask     = structured_df["Receiver Type"].astype(str).apply(_type_matches)
    type_df  = structured_df[mask]

    if type_df.empty:
        # src type has no rows at all → likely a RAG miss at the node level
        return {"rag_miss": True, "method_name": "?", "src_type": src,
                "reason": f"No rows for Receiver Type '{src}' in structured RAG"}

    # ── Leaf-action edge  [methodName] ────────────────────────────────────
    if tgt.startswith("[") and tgt.endswith("]"):
        method = tgt[1:-1]

        # Pass 1: exact Method Name match (case-insensitive)
        exact = type_df[type_df["Method Name"].astype(str).str.strip().str.lower()
                        == method.lower()]
        if not exact.empty:
            return _structured_row_to_hit(exact.iloc[0].to_dict(), 1.0)

        # Pass 2: Method Name contains method string
        fuzzy = type_df[type_df["Method Name"].astype(str).str.lower()
                        .str.contains(method.lower(), na=False, regex=False)]
        if not fuzzy.empty:
            return _structured_row_to_hit(fuzzy.iloc[0].to_dict(), 0.85)

        # Pass 3: semantic within type-restricted subset (threshold = _CONF_SUGGESTED)
        if embed_model is not None and len(type_df) > 0:
            query      = f"{src_short} {method}"
            q_emb      = embed_model.encode(query, convert_to_tensor=True)
            descs      = type_df["Description"].astype(str).tolist()
            d_embs     = embed_model.encode(descs, convert_to_tensor=True,
                                            show_progress_bar=False)
            scores_t   = _cos_sim(q_emb, d_embs).cpu().numpy().flatten()
            best_j     = int(scores_t.argmax())
            best_score = float(scores_t[best_j])
            if best_score >= _CONF_SUGGESTED:
                row = type_df.iloc[best_j].to_dict()
                return _structured_row_to_hit(row, best_score)
            # Below threshold → MISS (do not guess from low-confidence hit)

        # No confident match within this type → MISS sentinel
        return {"rag_miss": True, "method_name": method, "src_type": src,
                "status": "MISS",
                "reason": f"Method '{method}' not found on Receiver Type '{src}'"}

    # ── Normal acquisition edge  src → tgt_type ───────────────────────────
    tgt_short = tgt.rsplit(".", 1)[-1]

    # Pass 1: exact Return Type word-boundary match
    rt_exact = type_df[type_df["Return Type"].astype(str).apply(
        lambda v: _rt_matches(v, tgt_short) or _rt_matches(v, tgt)
    )]
    if not rt_exact.empty:
        return _structured_row_to_hit(rt_exact.iloc[0].to_dict(), 1.0)

    # Pass 2: semantic search within type-restricted subset (threshold = _CONF_SUGGESTED)
    if embed_model is not None and len(type_df) > 0:
        query  = f"{src_short} get {tgt_short} {tgt}"
        q_emb  = embed_model.encode(query, convert_to_tensor=True)
        descs  = type_df["Description"].astype(str).tolist()
        d_embs = embed_model.encode(descs, convert_to_tensor=True,
                                    show_progress_bar=False)
        scores_t = _cos_sim(q_emb, d_embs).cpu().numpy().flatten()
        best_j   = int(scores_t.argmax())
        best_score = float(scores_t[best_j])
        if best_score >= _CONF_SUGGESTED:
            row = type_df.iloc[best_j].to_dict()
            return _structured_row_to_hit(row, best_score)
        # Below threshold → MISS

    # No confident match within this type → MISS sentinel
    return {"rag_miss": True, "method_name": "?", "src_type": src,
            "status": "MISS",
            "reason": f"No method returning '{tgt_short}' found on Receiver Type '{src}' "
                      f"(below confidence threshold {_CONF_SUGGESTED})"}


def bootstrap_causal_rag(state: CausalAgentState,
                         embed_model, metadata: list, embeddings,
                         api_key: str = "", model: str = "",
                         structured_df=None) -> None:
    """Bootstrap step 2: per-edge RAG retrieval over all_edges. Updates state.edge_apis.

    When structured_df (RAGAPIs_structured.csv as DataFrame) is provided, each
    edge is first looked up via _rag_query_for_edge_typed (type-restricted).
    Only if that returns None does it fall back to the original semantic search
    on the full RAGAPIs.csv metadata/embeddings.
    """
    edges = state.all_edges if state.all_edges else (
        list(zip(state.chain[:-1], state.chain[1:])) if len(state.chain) >= 2 else []
    )
    if not edges:
        state.edge_apis   = []
        state.api_summary = "(no edges)"
        state.add_bootstrap_obs("causal_rag", "skipped — no edges")
        return

    edge_apis = []
    api_parts = []
    for src, tgt in edges:
        hit    = None
        source = "unstructured"

        # ── Path A: type-restricted lookup on structured CSV (preferred) ───
        # The node gate already confirmed every type is valid, so all edge
        # lookups must stay within that type's scope.  A global semantic
        # fallback risks matching methods from a different receiver type
        # (e.g. findLayer for dbPin), which reintroduces exactly the
        # hallucinations the node gate removed.
        if structured_df is not None:
            hit = _rag_query_for_edge_typed(
                src, tgt, structured_df, embed_model, task=state.task,
            )
            if hit is not None and not hit.get("rag_miss"):
                source = "structured"
            else:
                # No method found within this type — RAG_MISS is the safe
                # outcome.  Do NOT fall back to the global corpus.
                # (hit is already the rag_miss sentinel or None; keep it.)
                source = "structured(miss)"

        # ── Path B: structured CSV not loaded → legacy global search ───────
        # Only used when RAGStructuredPath was not provided (backward compat).
        else:
            hit = _rag_query_for_edge(
                src, tgt, embed_model, metadata, embeddings,
                api_key=api_key, model=model, task=state.task,
            )
            if hit is not None and not hit.get("rag_miss"):
                source = "unstructured"

        edge_apis.append(hit)

        src_s = src.split(".")[-1]
        tgt_s = tgt.split(".")[-1] if not tgt.startswith("[") else tgt
        if hit is None:
            api_parts.append(f"{src_s}->{tgt_s}: NOT FOUND")
            print(f"  [boot/causal_rag] [{src}] -> [{tgt}]  Hit: [none]", flush=True)
        elif hit.get("rag_miss"):
            reason = hit.get("reason", hit.get("method_name", "?"))
            label  = f"RAG_MISS({hit['method_name']})"
            api_parts.append(f"{src_s}->{tgt_s}: {label}")
            print(f"  [boot/causal_rag] [{src}] -> [{tgt}]  "
                  f"EDGE INVALID — {reason}", flush=True)
        else:
            method = hit["function_name"].split("(")[0].split(".")[-1].strip()
            api_parts.append(f"{src_s}->{tgt_s}: {method}")
            print(f"  [boot/causal_rag] [{src}] -> [{tgt}]  "
                  f"Hit: {hit['function_name']}  "
                  f"(score={hit['score']}, src={source})", flush=True)

    state.edge_apis   = edge_apis
    state.api_summary = " | ".join(api_parts)
    state.add_bootstrap_obs("causal_rag", state.api_summary)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Skeleton builder + GPT code completion
# ─────────────────────────────────────────────────────────────────────────────

_VAR_NAMES = {
    "openroad.Design":  "design",
    "odb.dbDatabase":   "db",
    "odb.dbBlock":      "block",
    "odb.dbInst":       "inst",
    "odb.dbNet":        "net",
    "odb.dbBTerm":      "bterm",
    "odb.dbITerm":      "iterm",
    "odb.dbMaster":     "master",
    "odb.dbMTerm":      "mterm",
    "odb.dbTechLayer":  "layer",
    "odb.dbTech":       "tech",
    "odb.dbRow":        "row",
    "odb.dbSite":       "site",
    "odb.Rect":         "bbox",
    "odb.Point":        "pt",
    "openroad.Tech":    "tech",
    "openroad.Timing":  "timing",
    "gpl.Replace":      "placer",
    "ifp.InitFloorplan":"floorplan",
    "cts.TritonCTS":    "cts",
    "grt.GlobalRouter": "router",
    "ppl.IOPlacer":     "io_placer",
    "drt.TritonRoute":  "detailed_router",
}

def _var(type_name: str) -> str:
    return _VAR_NAMES.get(type_name, type_name.split(".")[-1].lower())


def _build_skeleton(chain: list, edge_apis: list) -> str:
    lines = ["# --- Causal acquisition chain (schema-verified, do not modify) ---"]
    prev_var = "design"   # always pre-available

    for (src, tgt), api in zip(zip(chain[:-1], chain[1:]), edge_apis):
        # ── Leaf-action edge: tgt is [methodName] — emit a hint comment, not an assignment
        if tgt.startswith("[") and tgt.endswith("]"):
            method_name = tgt[1:-1]
            if api and api.get("rag_miss"):
                # RAG-MISS sentinel — warn generator not to hallucinate
                lines.append(
                    f"# WARNING: '{method_name}' on {src} was NOT found in the API database. "
                    f"Do NOT call .{method_name}() — it does not exist. "
                    f"Instead, use only real documented methods for {src.split('.')[-1]}."
                )
            elif api:
                fn   = api["function_name"]
                desc = api.get("description", "")[:70]
                lines.append(f"# Leaf action: {prev_var}.{method_name}()  # {desc}")
            else:
                lines.append(f"# Leaf action: {prev_var}.{method_name}()  # TODO: verify signature")
            continue   # don't update prev_var — leaf action doesn't produce a new variable

        tgt_var = _var(tgt)
        if api is None or api.get("rag_miss"):
            lines.append(f"{tgt_var} = ???  # TODO: find method on {src} → {tgt}")
            prev_var = tgt_var
            continue

        fn     = api["function_name"]
        method = (fn[:-1].split(".")[-1].strip() if fn.endswith("(")
                  else fn.split("(")[0].split(".")[-1].strip())
        params = api["parameters"].strip()
        if params.lower() == "nan":
            params = ""

        # Convert type-descriptor params (e.g. "str(name)") to clear placeholders
        # so GPT fills in the actual value from the task, not a variable called `name`
        if params:
            placeholders = re.findall(r'\b(?:str|int|float|bool)\((\w+)\)', params)
            if placeholders:
                params = ", ".join(f'"<{p}>"' for p in placeholders)

        call = f"{prev_var}.{method}({params})"
        lines.append(f"{tgt_var} = {call}  # {src} -> {tgt}")
        prev_var = tgt_var

    lines += [
        "",
        "# --- Task logic (implement below) ---",
        "# ...",
    ]
    return "\n".join(lines)


GENERATION_SYSTEM_PROMPT = """\
You are an expert OpenROAD Python API programmer.
Generate a Python script that runs inside the OpenROAD interactive Python shell.

Rules you MUST follow:
1. `design` (openroad.Design) and `tech` (openroad.Tech) are already available — do NOT import or re-create them.
2. You will be given the mandatory object-acquisition chain and the exact API method for each step.
   You MUST acquire objects in that exact order using those exact methods.
3. ALL methods are INSTANCE methods — call them on objects, never on the class.
4. Write FLAT procedural code — no functions, no classes, no `global`.
5. If you choose an API method that differs from the provided RAG suggestions or the Causal Chain \
structure, you MUST provide a [Diagnosis]. Explain the technical reason for the deviation \
(e.g., Cardinality mismatch, Hierarchy error in RAG docs, or Object-type conflict). \
Your goal is to provide 'Correct and Executable' code, even if it contradicts the provided hints.

Output format — use EXACTLY these two labeled blocks:
[Diagnosis]: (Optional) If you are deviating from the suggested Causal Chain or RAG API, explain \
why (e.g., "Singular vs Plural mismatch" or "Hierarchy error in RAG"). Write "None" if following exactly.
[Code]:
<Python script — no markdown fences>
"""


def _get_edge_api(src: str, tgt: str, all_edges: list, edge_apis: list):
    """Look up the API hit for a specific (src, tgt) edge."""
    for i, (s, t) in enumerate(all_edges):
        if s == src and t == tgt and i < len(edge_apis):
            return edge_apis[i]
    return None


def _build_generation_context(chain: list, edge_apis: list,
                               paths: list = None, all_edges: list = None,
                               action_node: str = "") -> str:
    """Build the acquisition context shown to the LLM.

    When paths/all_edges are provided, shows each path independently so the
    model understands which objects come from which acquisition route.
    """
    effective_paths = paths if paths else [chain]
    effective_edges = all_edges if all_edges else list(zip(chain[:-1], chain[1:]))

    lines = ["Causal acquisition paths (acquire ALL objects in ALL paths):"]
    step = 1
    seen_edges = set()

    for p_idx, path in enumerate(effective_paths, 1):
        lines.append(f"\n  Path {p_idx}: {' -> '.join(path)}")
        for src, tgt in zip(path[:-1], path[1:]):
            api = _get_edge_api(src, tgt, effective_edges, edge_apis)
            key = (src, tgt)
            if key in seen_edges:
                lines.append(f"    Step {step}: {src.split('.')[-1]} -> {tgt.split('.')[-1]}"
                             f"  |  (already acquired above)")
            elif api and api.get("rag_miss"):
                # MISS sentinel: no confident API found within the confirmed type scope.
                # Generator must NOT guess — use evalTclString or report not found.
                src_s   = src.split(".")[-1]
                tgt_s   = tgt[1:-1] if (tgt.startswith("[") and tgt.endswith("]")) else tgt.split(".")[-1]
                reason  = api.get("reason", api.get("method_name", "unknown"))
                lines.append(
                    f"    Step {step}: {src_s} -> {tgt_s}"
                    f"  |  [MISS — confidence below threshold] {reason}. "
                    f"DO NOT guess a method name. "
                    f"Instead use: design.evalTclString('<tcl_command>') "
                    f"OR print('API for {tgt_s} on {src_s} not found — skipping this step')"
                )
            elif api:
                raw_fn  = api["function_name"]
                fn      = (raw_fn[:-1].split(".")[-1].strip() if raw_fn.endswith("(")
                           else raw_fn.split("(")[0].split(".")[-1].strip())
                params  = api["parameters"].strip()
                if params.lower() == "nan":
                    params = ""
                call    = f"{fn}({params})" if params else f"{fn}()"
                status  = api.get("status", "MANDATORY")
                if status == "MANDATORY":
                    enforcement = "[MANDATORY — use exactly this method; the verifier will reject alternatives]"
                else:
                    enforcement = "[SUGGESTED — preferred method; use if applicable]"
                lines.append(
                    f"    Step {step}: {src.split('.')[-1]} -> {tgt.split('.')[-1]}"
                    f"  |  method: {call} {enforcement}  |  {api['description']}"
                )
            else:
                lines.append(f"    Step {step}: {src.split('.')[-1]} -> {tgt.split('.')[-1]}"
                             f"  |  [MISS] No API found — use evalTclString or skip")
            seen_edges.add(key)
            step += 1

    if action_node:
        lines.append(f"\n  Action: {action_node}")
    return "\n".join(lines)


def bootstrap_generate(state: CausalAgentState, api_key: str, model: str) -> None:
    """Bootstrap step 3: generate code from chain + APIs. Updates state.current_code."""
    # still build the skeleton string for logging/state (not sent to LLM)
    state.skeleton = _build_skeleton(state.chain, state.edge_apis)

    chain_context = _build_generation_context(
        state.chain, state.edge_apis,
        paths=state.paths or None,
        all_edges=state.all_edges or None,
        action_node=state.action_node,
    )

    user_msg = (
        f"Task: {state.task}\n\n"
        f"{chain_context}\n\n"
        f"Write the complete Python script that:\n"
        f"  1. Acquires the objects above in the exact order shown.\n"
        f"  2. Then fully implements the task — queries, modifications, prints, etc.\n"
        f"     Do NOT stop after the acquisition lines. The task logic is mandatory.\n"
        f"  3. Prints all results so they are visible in the shell output.\n\n"
        f"Remember: use the [Diagnosis] + [Code] output format."
    )
    raw = _call_openai(
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        api_key=api_key, model=model, max_tokens=1200,
    )

    text, diagnosis = _parse_generation_output(raw)

    state.current_code  = text
    state.llm_diagnosis = diagnosis
    if text:
        state.code_history.append(text)

    preview = text[:120].replace("\n", "↵") if text else "(empty)"
    state.add_bootstrap_obs("causal_generate",
                             f"generated {len(text)} chars",
                             detail=text)
    print(f"  [boot/causal_generate] {len(text)} chars  preview: {preview}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 (bootstrap) — Static verifier
# ─────────────────────────────────────────────────────────────────────────────

def _vresult_to_snapshot(vr: VerifierResult) -> VerifierSnapshot:
    return VerifierSnapshot(
        passed       = vr.passed,
        layer_failed = vr.layer_failed,
        issues       = vr.issues,
        feedback     = vr.feedback,
        confidence   = 1.0 if vr.passed else 0.0,
    )


def bootstrap_static_verify(state: CausalAgentState,
                              static_verifier: OpenROADStaticVerifier,
                              causal_verifier: CausalVerifier) -> None:
    """
    Bootstrap step 4: run causal verifier (L1→L4) on generated code.

    Runs the static verifier first for its L1 syntax check and L3 API table
    (broader coverage). Then runs the causal verifier for chain-flow checks
    (L2) and RAG-coverage warnings (L4). The tighter of the two results wins.
    """
    if not state.current_code:
        state.static_result = None
        state.add_bootstrap_obs("causal_verify", "skipped — no code")
        return

    # Static verifier: broad L1+L2+L3 check
    sv    = static_verifier.verify(state.task, state.current_code, rag_context="")
    s_snap = _vresult_to_snapshot(sv)

    # Causal verifier: chain-flow L1+L2+L3+L4
    # Pass state.paths so L3b checks terminal methods for ALL paths, not just paths[0].
    c_snap = causal_verifier.verify(
        state.current_code, state.chain, state.edge_apis,
        paths=state.paths or None,
    )

    # Pick the tighter (lower layer = harder failure) result, or PASS only if both pass
    if not s_snap.passed and not c_snap.passed:
        # pick whichever found a lower layer failure (more fundamental)
        snap = s_snap if s_snap.layer_failed <= c_snap.layer_failed else c_snap
    elif not s_snap.passed:
        snap = s_snap
    elif not c_snap.passed:
        snap = c_snap
    else:
        # both pass — merge L4 warnings from causal verifier into issues
        snap = VerifierSnapshot(
            passed=True, layer_failed=0,
            issues=c_snap.issues,   # L4 warnings
            feedback=c_snap.feedback,
        )

    state.static_result = snap
    state.maybe_update_best()

    status  = "PASS" if snap.passed else f"FAIL(layer={snap.layer_failed})"
    summary = f"{status}  issues={snap.issues[:2]}"
    state.add_bootstrap_obs("causal_verify", summary, detail=snap.feedback)
    print(f"  [boot/causal_verify] {summary}", flush=True)
    if not snap.passed and snap.feedback:
        for line in snap.feedback.splitlines()[:5]:
            print(f"    {line}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chain rewrite: Block→Master→Inst anti-pattern for bulk traversal tasks
# ─────────────────────────────────────────────────────────────────────────────

_TRAVERSAL_RE = re.compile(
    r'\b(all\s+instances|every\s+instance|each\s+instance'
    r'|instances\s+(using|of|with|from)'
    r'|all\s+\w+\s+instances)\b'
    r'|every\b.{1,50}\binstance\b',
    re.IGNORECASE,
)

def _rewrite_traversal_chain(task: str, chain: list) -> list:
    """
    Pattern match against the prompt: if the task asks to operate on ALL
    instances of a given cell type, a Block→Master→Inst chain is wrong.
    findMaster() gives you the cell definition, not its instances.

    Correct pattern for bulk traversal:
      block.getInsts() → filter by inst.getMaster().getName() == cell_name

    Rewrite: remove odb.dbMaster so the chain becomes Block→Inst directly,
    which steers the generator toward getInsts()+filter instead of findMaster.
    """
    if not _TRAVERSAL_RE.search(task):
        return chain  # prompt does not suggest bulk traversal — no change

    try:
        block_idx  = chain.index("odb.dbBlock")
        master_idx = chain.index("odb.dbMaster")
        inst_idx   = chain.index("odb.dbInst")
    except ValueError:
        return chain  # Block→Master→Inst pattern not present — no change

    if not (block_idx < master_idx < inst_idx):
        return chain  # nodes out of expected order — leave as-is

    new_chain = [n for n in chain if n != "odb.dbMaster"]
    print(f"  [chain-rewrite] traversal keywords detected — removing dbMaster", flush=True)
    print(f"    before: {' -> '.join(chain)}", flush=True)
    print(f"    after : {' -> '.join(new_chain)}", flush=True)
    return new_chain


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap runner
# ─────────────────────────────────────────────────────────────────────────────

def run_bootstrap(state: CausalAgentState,
                  api_key: str, model: str,
                  embed_model, metadata: list, embeddings,
                  static_verifier: OpenROADStaticVerifier,
                  causal_verifier: CausalVerifier,
                  gate: Optional[StructuredRAGGate] = None,
                  structured_df=None) -> None:
    """Run all pipeline steps: extract → node_gate → rag → generate → causal_verify.

    gate / structured_df are optional. When provided:
      • node_gate validates every type node and rewrites on hallucination.
      • causal_rag uses type-restricted lookup from structured_df first, then
        falls back to full semantic search on metadata/embeddings.
    """
    print(f"\n[Pipeline] causal_extract → node_gate → causal_rag → generate → causal_verify",
          flush=True)
    bootstrap_causal_extract(state, api_key, model)
    # Pattern guard: rewrite Block→Master→Inst to Block→Inst for bulk traversal prompts
    if state.chain:
        state.chain = _rewrite_traversal_chain(state.task, state.chain)
    # Node gate: validate types, auto-fix hallucinations
    if gate is not None:
        bootstrap_node_gate(state, gate, api_key, model)
    bootstrap_causal_rag(state, embed_model, metadata, embeddings,
                         api_key=api_key, model=model,
                         structured_df=structured_df)
    bootstrap_generate(state, api_key, model)
    bootstrap_static_verify(state, static_verifier, causal_verifier)
    print(f"[Bootstrap] done — committing generated code for OpenROAD execution\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Controller loop (to be implemented later)
# ─────────────────────────────────────────────────────────────────────────────

# def run_controller_loop(state: CausalAgentState, **kwargs) -> None:
#     """
#     Controller loop.
#     The controller reads state.to_controller_string() and decides next action.
#     Possible actions (TBD):
#       - re_generate  : re-run GPT with skeleton + verifier feedback
#       - re_retrieve  : refine RAG for a specific chain edge
#       - extend_chain : extend the causal chain with a missing node
#       - fix_code     : targeted repair from static verifier feedback
#       - commit_best  : commit best checkpoint and exit
#       - stop_fail    : give up and commit best available
#     Each action costs 1 budget unit (state.add_observation).
#     """
#     pass


# ─────────────────────────────────────────────────────────────────────────────
# Case runner
# ─────────────────────────────────────────────────────────────────────────────

def run_case(task: str, state: CausalAgentState,
             api_key: str, model: str,
             embed_model, metadata: list, embeddings,
             static_verifier: OpenROADStaticVerifier,
             causal_verifier: CausalVerifier,
             code_pieces: Optional[list] = None,
             llm_verifier: Optional["CausalLLMVerifier"] = None,
             gate: Optional[StructuredRAGGate] = None,
             structured_df=None) -> CausalAgentState:
    """Bootstrap → controller loop → commit."""
    print(f"\n{'─'*60}", flush=True)
    print(f"TASK: {task}", flush=True)
    print(f"BUDGET: {state.max_budget} controller actions", flush=True)

    run_bootstrap(state, api_key, model, embed_model, metadata, embeddings,
                  static_verifier, causal_verifier,
                  gate=gate, structured_df=structured_df)

    # if bootstrap already passed static+causal — run LLM verifier before committing
    if state.static_result and state.static_result.passed:
        if llm_verifier is not None and state.current_code:
            from dispatcher import _build_chain_context
            chain_ctx = _build_chain_context(state.chain, state.edge_apis)
            lv_snap   = llm_verifier.verify(task, state.current_code, chain_ctx)
            state.llm_result = lv_snap
            if lv_snap.passed:
                print(f"  [loop] bootstrap PASS + LLM PASS (conf={lv_snap.confidence:.2f})"
                      f" — committing directly", flush=True)
                state.committed_code = state.best_code or state.current_code
                state.committed      = True
                return state
            else:
                print(f"  [loop] bootstrap static PASS but LLM FAIL(L5) "
                      f"conf={lv_snap.confidence:.2f} — entering controller loop",
                      flush=True)
                # Don't commit — let the controller fix it
        else:
            print(f"  [loop] bootstrap passed — committing directly", flush=True)
            state.committed_code = state.best_code or state.current_code
            state.committed      = True
            return state

    # ── controller loop ────────────────────────────────────────────────────────
    ctrl       = CausalController(api_key=api_key, model=model)
    dispatcher = CausalDispatcher(
        api_key=api_key, model=model,
        embed_model=embed_model, metadata=metadata, embeddings=embeddings,
        static_verifier=static_verifier, causal_verifier=causal_verifier,
        code_pieces=code_pieces,
        llm_verifier=llm_verifier,
    )
    # Build stable prefix from bootstrap chain+apis so all re_generate calls
    # share the same [system, user(task+chain_ctx)] cache key.
    dispatcher.reset_conversation(state)

    while not state.committed and state.budget_remaining > 0:
        decision = ctrl.decide(state)

        tag = "[fallback]" if decision.from_fallback else "[LLM]"
        print(f"\n  [controller {tag}] action={decision.next_action}  "
              f"budget_left={state.budget_remaining}", flush=True)
        print(f"    diagnosis:  {decision.diagnosis[:120]}", flush=True)
        if decision.target_edge:
            print(f"    target_edge:{decision.target_edge}", flush=True)
        if decision.repair_hint:
            print(f"    hint:       {decision.repair_hint[:150]}", flush=True)

        # write lesson before executing so it's visible on the next step
        if decision.updated_lesson:
            state.add_lesson(decision.updated_lesson)
            print(f"    lesson:     {decision.updated_lesson[:100]}", flush=True)

        observation = dispatcher.execute(decision, state)
        state.add_observation(decision.next_action, observation,
                              detail=decision.repair_hint)

        if state.committed:
            break

    # safety: auto-commit if budget exhausted
    if not state.committed:
        state.committed_code = state.best_code or state.current_code
        state.committed      = True
        print(f"  [loop] budget exhausted — committing best (score={state.best_score:.3f})",
              flush=True)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# OpenROAD execution (once per case, after commit)
# ─────────────────────────────────────────────────────────────────────────────

def execute_in_openroad(code: str, openroad_path: str, slave_fd: int,
                        output_queue: queue.Queue,
                        run_dir: str = "",
                        load_design_time: int = 5,
                        max_wait_time: int = 120,
                        command_flush_time: float = 0.1) -> Tuple[str, bool]:
    """Run committed code in a fresh OpenROAD shell. Returns (stdout, has_traceback)."""
    orig_dir = os.getcwd()
    openroad_path = os.path.abspath(openroad_path)
    if run_dir:
        os.chdir(run_dir)

    proc = None
    try:
        proc = runOpenROADShell(openroad_path, load_design_time, slave_fd, "")
        time.sleep(load_design_time)
        clearQueue(output_queue)

        cmd = processCodeString(code)
        while True:
            try:
                stdout, has_tb = sendCommandOpenROAD(
                    proc, cmd, output_queue, max_wait_time, command_flush_time,
                )
                break
            except RuntimeError:
                print("    [OpenROAD crashed] restarting...", flush=True)
                proc = runOpenROADShell(openroad_path, load_design_time, slave_fd, "")
                time.sleep(load_design_time)
                clearQueue(output_queue)

        proc.terminate()
        proc.wait()
        # Drain any residual PTY output the daemon thread may still be
        # pushing into the queue after process termination, so it cannot
        # contaminate the next case's output reading.
        time.sleep(0.3)
        clearQueue(output_queue)
        return stdout, has_tb
    finally:
        os.chdir(orig_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main eval loop
# ─────────────────────────────────────────────────────────────────────────────

def Run(args):
    # ── dataset ───────────────────────────────────────────────────────────────
    ext = os.path.splitext(args.dataset)[1].lower()
    df  = pd.read_excel(args.dataset) if ext in (".xlsx", ".xls") else pd.read_csv(args.dataset)
    col = args.promptColumn
    if col not in df.columns:
        # try common fallbacks
        for candidate in ("prompt", "Level One", "Sub-Prompts", "Complex Prompt"):
            if candidate in df.columns:
                col = candidate
                print(f"[dataset] --promptColumn not found; using '{col}'")
                break
        else:
            print(f"ERROR: column '{col}' not found. Available: {list(df.columns)}")
            sys.exit(1)
    prompts = df[col].dropna().tolist()
    if args.numCases > 0:
        prompts = prompts[:args.numCases]

    # ── RAG setup ─────────────────────────────────────────────────────────────
    print("\nLoading embedding model...", flush=True)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    rag_df   = pd.read_csv(args.ragApi)
    metadata = []
    documents = []
    for _, row in rag_df.iterrows():
        desc = str(row.get("Description:", "")).strip()
        if not desc or desc.lower() == "nan":
            continue
        documents.append(f"OpenROAD Python API Description:{desc}")
        metadata.append(row.to_dict())
    print(f"  [RAG] Encoding {len(documents)} API entries...", flush=True)
    embeddings = embed_model.encode(documents, convert_to_tensor=True, show_progress_bar=False)

    # ── verifiers ─────────────────────────────────────────────────────────────
    static_verifier = OpenROADStaticVerifier(args.ragApi)
    causal_ver      = CausalVerifier(metadata=metadata)
    llm_ver         = CausalLLMVerifier(api_key=args.openaiKey, model=args.model,
                                        fail_open=True)
    print("[Init] Static + causal + LLM semantic verifiers ready.", flush=True)

    # ── code examples (RAGCodePiece.csv) ──────────────────────────────────────
    code_pieces = []
    if args.ragCodePiece and os.path.isfile(args.ragCodePiece):
        cp_df = pd.read_csv(args.ragCodePiece)
        for _, row in cp_df.iterrows():
            desc = str(row.get("Description:", "")).strip()
            code = str(row.get("Code Piece:", "")).strip()
            if desc and code and desc.lower() != "nan" and code.lower() != "nan":
                code_pieces.append({"description": desc, "code": code})
        print(f"[Init] Loaded {len(code_pieces)} code examples from {args.ragCodePiece}",
              flush=True)
    else:
        print("[Init] --ragCodePiece not set — code examples disabled.", flush=True)

    # ── OpenROAD PTY (one per run) ────────────────────────────────────────────
    slave_fd     = None
    output_queue = None
    stop_event   = None
    if args.openroadPath:
        master_fd, slave_fd = os.openpty()
        output_queue        = queue.Queue()
        stop_event          = threading.Event()
        threading.Thread(
            target=readOpenROADOutput,
            args=(master_fd, output_queue, "STDOUT", stop_event),
            daemon=True,
        ).start()
        print("[Init] OpenROAD PTY ready.", flush=True)
    else:
        print("[Init] --openroadPath not set — OpenROAD execution will be skipped.", flush=True)

    print(f"\nLoaded  : {len(prompts)} prompts from {args.dataset}")
    print(f"Model   : {args.model}")
    print("=" * 70)

    passed  = 0
    results = []

    for i, task in enumerate(prompts, 1):
        task = str(task).strip()
        print(f"\n[{i}/{len(prompts)}] {task}")

        state = CausalAgentState(task=task, max_budget=args.budget if hasattr(args, 'budget') else 4)

        state = run_case(
            task, state,
            api_key=args.openaiKey, model=args.model,
            embed_model=embed_model, metadata=metadata, embeddings=embeddings,
            static_verifier=static_verifier, causal_verifier=causal_ver,
            code_pieces=code_pieces,
            llm_verifier=llm_ver,
        )

        # ── OpenROAD execution ────────────────────────────────────────────────
        ora_pass   = None
        ora_stdout = ""

        if args.openroadPath and state.committed_code:
            print(f"\n  [Execution] Running committed code in OpenROAD...", flush=True)
            try:
                ora_stdout, has_tb = execute_in_openroad(
                    state.committed_code, args.openroadPath, slave_fd, output_queue,
                    run_dir=args.runDir,
                    load_design_time=args.loadDesignTime,
                    max_wait_time=args.maxWaitTime,
                )
                ora_pass = not has_tb
            except Exception as exc:
                ora_stdout = f"[exec error] {exc}"
                ora_pass   = False

            status = "PASS" if ora_pass else "FAIL"
            print(f"  [Execution] {status}", flush=True)
            for line in _clean_output(ora_stdout).splitlines()[:10]:
                print(f"    {line}", flush=True)
            if ora_pass:
                passed += 1

        sv = state.static_result
        # Build controller action history: action → one-line result
        ctrl_history = " | ".join(
            f"{o.action}:{o.result[:50]}"
            for o in state.observations
            if not o.is_bootstrap
        )
        results.append({
            "prompt":           task,
            "chain":            " -> ".join(state.chain),
            "node_apis":        state.api_summary,
            "skeleton":         state.skeleton,
            "generated_code":   state.committed_code,
            "causal_verdict":   ("PASS" if sv.passed else f"FAIL(L{sv.layer_failed}): {'; '.join(sv.issues[:2])}") if sv else "N/A",
            "openroad_result":  ("PASS" if ora_pass else "FAIL") if ora_pass is not None else "SKIPPED",
            "openroad_output":  _clean_output(ora_stdout)[:1000],
            "ctrl_actions":     ctrl_history,
            "steps_used":       state.step,
            "budget_used":      state.budget_used,
        })

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if args.openroadPath:
        print(f"OpenROAD: {passed}/{len(results)} passed")
    if args.output:
        pd.DataFrame(results).to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Saved → {args.output}")
    print(f"Done: {len(results)} cases processed.")
    if stop_event:
        stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        required=True,  help=".xlsx or .csv with prompts")
    parser.add_argument("--promptColumn",   default="prompt", help="Column name for task prompts (default: 'prompt')")
    parser.add_argument("--ragApi",         required=True,  help="Path to RAGAPIs.csv")
    parser.add_argument("--ragCodePiece",   default="",     help="Path to RAGCodePiece.csv (optional, for code examples in re-retrieval)")
    parser.add_argument("--openaiKey",      required=True,  help="OpenAI API key")
    parser.add_argument("--model",          default="gpt-4.1-mini")
    parser.add_argument("--output",         default="",     help="Optional: save results to CSV")
    parser.add_argument("--numCases",       default=0,      type=int,   help="0 = all")
    parser.add_argument("--budget",         default=6,      type=int,   help="Controller action budget per case")
    parser.add_argument("--openroadPath",   default="",     help="Path to openroad binary")
    parser.add_argument("--runDir",         default="src_1_reflector", help="Working dir for OpenROAD")
    parser.add_argument("--loadDesignTime", default=5,      type=int)
    parser.add_argument("--maxWaitTime",    default=120,    type=int)
    args = parser.parse_args()
    Run(args)


if __name__ == "__main__":
    main()
