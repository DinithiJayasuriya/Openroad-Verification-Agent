"""extract_chains.py — Four-stage causal pipeline per prompt:

  Stage 1 — Chain Extraction  (GPT-4.1-mini)
      task text → mandatory object-acquisition chain
      e.g. "Find instance 486" → [dbDatabase, dbChip, dbBlock, dbInst]

  Stage 2 — Node-Specific RAG  (sentence-transformers + RAGAPIs.csv)
      one targeted query per chain edge instead of one big retrieval
      e.g. "Methods to get dbChip from dbDatabase" → db.getChip()

  Stage 3 — Skeleton Generation  (GPT-4.1-mini)
      the verified acquisition chain is locked in as a hard-coded skeleton;
      GPT only fills in the task logic below it — it cannot touch the chain.

  Stage 4 — OpenROAD Execution
      run the generated code in a live OpenROAD shell, record PASS/FAIL.

Usage:
  conda activate prompt
  python causal_verifier/extract_chains.py \
      --dataset      path/to/data.xlsx \
      --ragApi       RAGData/RAGAPIs.csv \
      --openaiKey    sk-... \
      --openroadPath OpenROAD/build/src/openroad

  # Also save to CSV
  python causal_verifier/extract_chains.py ... --output result/chains.csv

Requires: sentence_transformers  (available in 'prompt' conda env)
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

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# OpenROAD utilities (from existing pipeline)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src_1_agentic"))
from util import runOpenROADShell, sendCommandOpenROAD, processCodeString, clearQueue, readOpenROADOutput


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Chain extraction
# ─────────────────────────────────────────────────────────────────────────────

CHAIN_SYSTEM_PROMPT = """\
You are an OpenROAD Python API expert.

Given a task description, identify the mandatory sequence of OpenROAD Python object
types that must be acquired step-by-step to complete the task.

IMPORTANT — Python API rules (these differ from the C++ internal hierarchy):
- The entry point is always openroad.Design. The variable `design` is always
  pre-available in the shell — do NOT include any initialisation before it.
- To reach odb.dbBlock, call design.getBlock() directly. There is NO dbChip
  step in the Python API — that is a C++ internal not exposed in Python.
- To reach odb.dbDatabase, call design.getDb().
- Do NOT include dbChip in any chain — it does not exist in the Python API.
- Only include types that are NECESSARY transitions to reach the target object.
- Use exact OpenROAD Python type names.
- Output ONLY a JSON array of type name strings. No explanation, no markdown.

Examples:
  Task: "Find instance named u_cpu"
  Output: ["openroad.Design", "odb.dbBlock", "odb.dbInst"]

  Task: "Get all nets in the design"
  Output: ["openroad.Design", "odb.dbBlock", "odb.dbNet"]

  Task: "Find the top-level port clk"
  Output: ["openroad.Design", "odb.dbBlock", "odb.dbBTerm"]

  Task: "Find which net is connected to pin A of instance inv_1"
  Output: ["openroad.Design", "odb.dbBlock", "odb.dbInst", "odb.dbITerm", "odb.dbNet"]

  Task: "Find library cell BUF_X4"
  Output: ["openroad.Design", "odb.dbDatabase", "odb.dbMaster"]
"""


def extract_chain(task: str, api_key: str, model: str) -> list:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": CHAIN_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Task: {task}"},
        ],
        "temperature": 0,
        "max_tokens": 150,
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
            text  = body["choices"][0]["message"]["content"].strip()
            chain = json.loads(text)
            if isinstance(chain, list) and all(isinstance(t, str) for t in chain):
                return chain
            return []
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (2 ** attempt)
                print(f"    [rate limit] waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    [HTTP {e.code}] {e.reason}")
                return []
        except Exception as e:
            print(f"    [error] {e}")
            return []
    return []


def fmt_chain(chain: list) -> str:
    return " -> ".join(f"[{t}]" for t in chain)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Node-specific RAG
# ─────────────────────────────────────────────────────────────────────────────

def load_rag(rag_api_path: str, embed_model: SentenceTransformer):
    df = pd.read_csv(rag_api_path)
    documents, metadata = [], []
    for _, row in df.iterrows():
        desc = str(row.get("Description:", "")).strip()
        if not desc or desc.lower() == "nan":
            continue
        documents.append(f"OpenROAD Python API Description:{desc}")
        metadata.append(row.to_dict())
    print(f"  [RAG] Encoding {len(documents)} API entries...", flush=True)
    embeddings = embed_model.encode(documents, convert_to_tensor=True,
                                    show_progress_bar=False)
    return documents, metadata, embeddings


def _row_to_hit(row: dict, score: float) -> dict:
    return {
        "description":   str(row.get("Description:", "")).strip(),
        "function_name": str(row.get("Function Name:", "")).strip(),
        "parameters":    str(row.get("Parameters:", "")).strip(),
        "return_type":   str(row.get("Return Type:", "")).strip(),
        "score":         round(score, 3),
    }


def rag_query_for_edge(src: str, tgt: str,
                       embed_model, metadata: list, embeddings,
                       top_k: int = 3, threshold: float = 0.40) -> list:
    """
    Two-pass retrieval for a single chain edge (src -> tgt):

    Pass 1 — structural filter (exact return_type match):
        Look for rows in the CSV where:
          - return_type contains `tgt`  (the method actually returns what we need)
          - function_name starts with the source type prefix (called ON src)
        This is precise — no semantic drift.

    Pass 2 — semantic similarity fallback:
        If Pass 1 finds nothing, fall back to embedding similarity using a
        richer query that includes both type names + the function keyword.
    """
    # normalise: strip "odb." / "openroad." prefix for loose matching
    src_short = src.replace("odb.", "").replace("openroad.", "")
    tgt_short = tgt.replace("odb.", "").replace("openroad.", "")

    # ── Pass 1a: strict structural filter ────────────────────────────────────
    # return_type contains tgt AND function_name contains src
    strict = []
    for row in metadata:
        rt = str(row.get("Return Type:", "")).strip()
        fn = str(row.get("Function Name:", "")).strip()
        if (tgt_short in rt or tgt in rt) and \
           (src_short.lower() in fn.lower() or src.lower() in fn.lower()):
            strict.append(_row_to_hit(row, 1.0))
    if strict:
        return strict[:top_k]

    # ── Pass 1b: loose structural filter ─────────────────────────────────────
    # return_type contains tgt (ignore src constraint — handles shortcuts like
    # design.getBlock() that skip intermediate C++ objects not in the Python API)
    loose = []
    for row in metadata:
        rt = str(row.get("Return Type:", "")).strip()
        if tgt_short in rt or tgt in rt:
            loose.append(_row_to_hit(row, 0.9))
    if loose:
        return loose[:top_k]

    # ── Pass 2: semantic fallback ─────────────────────────────────────────────
    # Richer query: include both type names and the transition keyword
    query = f"{src} get {tgt} {src_short} {tgt_short} method"
    q_emb   = embed_model.encode(query, convert_to_tensor=True)
    scores  = cos_sim(q_emb, embeddings).cpu().numpy().flatten()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    results = []
    for i in top_idx:
        if scores[i] < threshold:
            break
        results.append(_row_to_hit(metadata[i], float(scores[i])))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Skeleton builder + GPT code completion
# ─────────────────────────────────────────────────────────────────────────────

# Maps OpenROAD type names to clean Python variable names
_VAR_NAMES = {
    "openroad.Design":  "design",   # pre-available — no acquisition line needed
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
}

def _var(type_name: str) -> str:
    return _VAR_NAMES.get(type_name, type_name.split(".")[-1].lower())


def build_skeleton(chain: list, edge_apis: list) -> str:
    """
    Build the schema-correct skeleton from the verified chain and RAG APIs.

    openroad.Design is always the first node. `design` is pre-available in the
    OpenROAD shell — no acquisition line is emitted for it.
    """
    lines = ["# --- Causal acquisition chain (schema-verified, do not modify) ---"]

    # design is pre-available — start traversal from it
    prev_var = "design"

    for (src, tgt), api in zip(zip(chain[:-1], chain[1:]), edge_apis):
        tgt_var = _var(tgt)

        if api is None:
            lines.append(f"{tgt_var} = ???  # TODO: find method on {src} that returns {tgt}")
            prev_var = tgt_var
            continue

        # Extract bare method name: "openroad.Design.getBlock(" → "getBlock"
        fn     = api["function_name"]
        method = fn.split("(")[0].split(".")[-1].strip()
        params = api["parameters"].strip()
        if params.lower() == "nan":
            params = ""

        # Turn CSV type-descriptor params (e.g. "str(name)") into a clear
        # placeholder so GPT substitutes the actual value from the task, rather
        # than inventing a variable called `name` and using `global`.
        # Parameters with no arguments (getBlock, getNets, …) are left as-is.
        if params:
            # Extract all bare identifiers from the param string as placeholder names
            # e.g. "str(name)" → "<name>", "int(x), str(y)" → "<x>, <y>"
            placeholders = re.findall(r'\b(?:str|int|float|bool)\((\w+)\)', params)
            if placeholders:
                params = ", ".join(f'"<{p}>"' for p in placeholders)
            # else leave params unchanged (e.g. already a literal like "0")

        call = f"{prev_var}.{method}({params})"
        lines.append(f"{tgt_var} = {call}  # {src} -> {tgt}")
        prev_var = tgt_var

    lines += [
        "",
        "# --- Task logic (implement below) ---",
        "# ...",
    ]
    return "\n".join(lines)


SKELETON_SYSTEM_PROMPT = """\
You are an expert OpenROAD Python API programmer.
Generate a Python script that runs inside the OpenROAD interactive Python shell.

Rules you must follow:
1. The variables `design` (openroad.Design) and `tech` (openroad.Tech) are already
   available in the shell — do NOT import or re-create them.
2. The acquisition skeleton below is VERIFIED and LOCKED — copy every line of it
   verbatim into your output, then add the task logic after it.
3. Do NOT re-acquire any object already defined in the skeleton.
4. ALL OpenROAD API methods are INSTANCE methods — call them on objects, never on the class.
5. Every Python function you define must be called at the end of the script.
6. Output ONLY the Python code. No explanation, no markdown fences.
"""


def complete_skeleton(task: str, skeleton: str, api_key: str, model: str) -> str:
    user_msg = (
        f"Task: {task}\n\n"
        f"Skeleton (copy every line, then add task logic after the last comment):\n"
        f"{skeleton}\n\n"
        f"Output the complete script:"
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SKELETON_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": 600,
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
            text = body["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = "\n".join(
                    line for line in text.splitlines()
                    if not line.strip().startswith("```")
                )
            return text.strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (2 ** attempt)
                print(f"    [rate limit] waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    [HTTP {e.code}] {e.reason}")
                return ""
        except Exception as e:
            print(f"    [error] {e}")
            return ""
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — OpenROAD execution
# ─────────────────────────────────────────────────────────────────────────────

def run_in_openroad(code: str, openroad_path: str, slave_fd: int,
                    output_queue: queue.Queue,
                    run_dir: str = "",
                    load_design_time: int = 5,
                    max_wait_time: int = 120,
                    command_flush_time: float = 0.1) -> tuple:
    """
    Start a fresh OpenROAD shell, execute code, return (stdout, has_traceback).
    Pass/fail = not has_traceback.

    run_dir: working directory to switch to before launching OpenROAD.
    The loadDesign utility uses relative paths like ../design/nangate45/ which
    only resolve correctly from inside src_1_reflector/ (or equivalent).
    """
    orig_dir = os.getcwd()
    openroad_path = os.path.abspath(openroad_path)   # resolve before any chdir
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
                    proc, cmd, output_queue, max_wait_time, command_flush_time
                )
                break
            except RuntimeError:
                print("    [OpenROAD crashed] restarting...", flush=True)
                proc = runOpenROADShell(openroad_path, load_design_time, slave_fd, "")
                time.sleep(load_design_time)
                clearQueue(output_queue)

        proc.terminate()
        proc.wait()
        return stdout, has_tb
    finally:
        os.chdir(orig_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    ext = os.path.splitext(args.dataset)[1].lower()
    df  = pd.read_excel(args.dataset) if ext in (".xlsx", ".xls") else pd.read_csv(args.dataset)

    if "prompt" not in df.columns:
        print(f"ERROR: no 'prompt' column. Available: {list(df.columns)}")
        sys.exit(1)

    prompts = df["prompt"].dropna().tolist()
    if args.numCases > 0:
        prompts = prompts[:args.numCases]

    print("\nLoading embedding model...", flush=True)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    _, metadata, embeddings = load_rag(args.ragApi, embed_model)

    # ── OpenROAD PTY setup (once, shared across all cases) ────────────────────
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
        print(f"[Init] OpenROAD PTY ready.", flush=True)
    else:
        print("[Init] --openroadPath not set — Stage 4 (execution) will be skipped.", flush=True)

    print(f"\nLoaded  : {len(prompts)} prompts from {args.dataset}")
    print(f"Model   : {args.model}")
    print("=" * 70)

    passed  = 0
    results = []

    for i, task in enumerate(prompts, 1):
        task = str(task).strip()
        print(f"\n[{i}/{len(prompts)}] {task}")
        print("─" * 60)

        # ── Stage 1: chain extraction ──────────────────────────────────────
        chain = extract_chain(task, args.openaiKey, args.model)
        if not chain:
            print("  [Stage 1] FAILED — no chain extracted")
            results.append({"prompt": task, "chain": "[FAILED]",
                             "skeleton": "", "generated_code": ""})
            continue
        print(f"  [Stage 1] {fmt_chain(chain)}")

        # ── Stage 2: per-edge RAG retrieval ───────────────────────────────
        edge_apis  = []
        api_summary = []
        for src, tgt in zip(chain[:-1], chain[1:]):
            hits  = rag_query_for_edge(src, tgt, embed_model, metadata, embeddings)

            print(f"\n  [Stage 2] [{src}] -> [{tgt}]")

            if hits:
                best = hits[0]
                print(f"           Hit  : {best['function_name']}  (score={best['score']})")
                print(f"                  {best['description']}")
                edge_apis.append(best)
                api_summary.append(f"{src}->{tgt}: {best['function_name']}")
            else:
                print(f"           Hit  : [none above threshold]")
                edge_apis.append(None)
                api_summary.append(f"{src}->{tgt}: NOT FOUND")

        # ── Stage 3: build skeleton + complete ────────────────────────────
        skeleton = build_skeleton(chain, edge_apis)

        print(f"\n  [Stage 3] Skeleton:")
        for line in skeleton.splitlines():
            print(f"    {line}")

        print(f"\n  [Stage 3] Generating code from skeleton...")
        code = complete_skeleton(task, skeleton, args.openaiKey, args.model)

        print(f"\n  [Stage 3] Generated code:")
        for line in code.splitlines():
            print(f"    {line}")

        # ── Stage 4: run in OpenROAD ──────────────────────────────────────
        ora_pass   = None
        ora_stdout = ""

        if args.openroadPath and code:
            print(f"\n  [Stage 4] Running in OpenROAD...", flush=True)
            try:
                ora_stdout, has_tb = run_in_openroad(
                    code, args.openroadPath, slave_fd, output_queue,
                    run_dir=args.runDir,
                    load_design_time=args.loadDesignTime,
                    max_wait_time=args.maxWaitTime,
                )
                ora_pass = not has_tb
            except Exception as exc:
                ora_stdout = f"[exec error] {exc}"
                ora_pass   = False

            status = "PASS" if ora_pass else "FAIL"
            print(f"  [Stage 4] {status}")
            if ora_stdout:
                # print first 10 lines of output
                for line in ora_stdout.splitlines()[:10]:
                    print(f"    {line}")
            if ora_pass:
                passed += 1

        results.append({
            "prompt":         task,
            "chain":          fmt_chain(chain),
            "node_apis":      " | ".join(api_summary),
            "skeleton":       skeleton,
            "generated_code": code,
            "openroad_result": ("PASS" if ora_pass else "FAIL") if ora_pass is not None else "SKIPPED",
            "openroad_output": ora_stdout[:1000],
        })

    print("\n" + "=" * 70)
    if args.openroadPath:
        print(f"OpenROAD: {passed}/{len(results)} passed")
    if args.output:
        pd.DataFrame(results).to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Saved → {args.output}")
    print(f"Done: {len(results)} prompts processed.")

    if stop_event:
        stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True, help=".xlsx or .csv with a 'prompt' column")
    parser.add_argument("--ragApi",     required=True, help="Path to RAGAPIs.csv")
    parser.add_argument("--openaiKey",  required=True, help="OpenAI API key")
    parser.add_argument("--model",          default="gpt-4.1-mini")
    parser.add_argument("--output",         default="",  help="Optional: save results to CSV")
    parser.add_argument("--numCases",       default=0,   type=int, help="Number of prompts to run (0 = all)")
    parser.add_argument("--openroadPath",   default="",  help="Path to openroad binary (enables Stage 4)")
    parser.add_argument("--runDir",         default="src_1_reflector", help="Working directory for OpenROAD (loadDesign uses relative paths)")
    parser.add_argument("--loadDesignTime", default=5,   type=int,   help="Seconds to wait for design load")
    parser.add_argument("--maxWaitTime",    default=120, type=int,   help="Max seconds to wait for OpenROAD output")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
