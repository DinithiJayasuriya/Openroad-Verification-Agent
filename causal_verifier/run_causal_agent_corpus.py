"""run_causal_agent_corpus.py — Causal agent runner for EDA-Corpus-v2/TestSet.xlsx.

Drop-in companion to run_causal_agent.py that speaks the same CLI interface
as the other EDA-Corpus scripts (--testSetPath, --RAGApiPath, --sheets, etc.)
instead of the generic --dataset / --ragApi flags.

Dataset layout expected (EDA-Corpus-v2/TestSet.xlsx):
  Sheet 'Prompt' : one prompt per row, no header (header=None)
  Sheet 'Code'   : one ground-truth script per row, same indexing
  (Sheet 'Sheet' is ignored)

Usage:
  conda activate prompt
  python causal_verifier/run_causal_agent_corpus.py \\
      --testSetPath    EDA-Corpus-v2/TestSet.xlsx \\
      --RAGApiPath     RAGData/RAGAPIs.csv \\
      --RAGCodePiecePath RAGData/RAGCodePiece.csv \\
      --openaiKey      sk-... \\
      --openroadPath   OpenROAD/build/src/openroad \\
      --runDir         src_1_reflector \\
      --numCases       20 \\
      --budget         6 \\
      --resultPath     result/causal_corpus_run.csv
"""

import argparse
import os
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd
from openpyxl import Workbook
from sentence_transformers import SentenceTransformer

# ── path setup (mirror run_causal_agent.py) ────────────────────────────────────
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAUSAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src_1_reflector"))
sys.path.insert(0, os.path.join(_ROOT, "src_1_agentic"))
sys.path.insert(0, _CAUSAL_DIR)

from util import readOpenROADOutput
from verifier import OpenROADStaticVerifier

from causal_state    import CausalAgentState
from causal_verifier import CausalVerifier
from controller      import CausalController
from dispatcher      import CausalDispatcher
from llm_verifier    import CausalLLMVerifier

# Re-use all bootstrap / run-loop functions from the main script
from run_causal_agent import (
    run_bootstrap,
    run_case,
    execute_in_openroad,
    _clean_output,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(path: str, num_cases: int = 0, prompt_file: str = ""):
    """Load prompts (and ground-truth code) from EDA-Corpus TestSet.xlsx.

    Returns (prompts, gt_codes) — both plain lists of strings.
    Blank / whitespace-only rows are skipped.

    If prompt_file is given, only prompts whose text matches a line in that
    file (stripped, exact substring match) are included.
    """
    prompt_df = pd.read_excel(path, "Prompt", header=None).rename(columns={0: "text"})
    code_df   = pd.read_excel(path, "Code",   header=None).rename(columns={0: "text"})

    filter_set = []
    if prompt_file and os.path.isfile(prompt_file):
        with open(prompt_file) as f:
            filter_set = [line.strip() for line in f if line.strip()]
        print(f"[Filter] Restricting to {len(filter_set)} prompts from {prompt_file}")

    prompts  = []
    gt_codes = []
    for p, c in zip(prompt_df["text"], code_df["text"]):
        p_str = str(p).strip()
        c_str = str(c).strip()
        if not p_str or p_str.lower() == "nan":
            continue
        if filter_set and not any(f in p_str or p_str in f for f in filter_set):
            continue
        prompts.append(p_str)
        gt_codes.append(c_str if c_str.lower() != "nan" else "")

    if num_cases > 0:
        prompts  = prompts[:num_cases]
        gt_codes = gt_codes[:num_cases]

    return prompts, gt_codes


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def RunCorpus(args):
    # ── load dataset ──────────────────────────────────────────────────────────
    prompts, gt_codes = load_corpus(args.testSetPath, args.numCases, args.promptFile)
    print(f"\nLoaded  : {len(prompts)} prompts from {args.testSetPath}")
    print(f"Model   : {args.model}")
    print("=" * 70, flush=True)

    # ── RAG setup ─────────────────────────────────────────────────────────────
    print("\nLoading embedding model...", flush=True)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    rag_df    = pd.read_csv(args.RAGApiPath)
    metadata  = []
    documents = []
    for _, row in rag_df.iterrows():
        desc = str(row.get("Description:", "")).strip()
        if not desc or desc.lower() == "nan":
            continue
        documents.append(f"OpenROAD Python API Description:{desc}")
        metadata.append(row.to_dict())
    print(f"  [RAG] Encoding {len(documents)} API entries...", flush=True)
    embeddings = embed_model.encode(documents, convert_to_tensor=True,
                                    show_progress_bar=False)

    # ── verifiers ─────────────────────────────────────────────────────────────
    static_verifier = OpenROADStaticVerifier(args.RAGApiPath)
    causal_ver      = CausalVerifier(metadata=metadata)
    llm_ver         = CausalLLMVerifier(api_key=args.openaiKey, model=args.model,
                                        fail_open=True)
    print("[Init] Static + causal + LLM semantic verifiers ready.", flush=True)

    # ── code examples (RAGCodePiece.csv) ──────────────────────────────────────
    code_pieces = []
    if args.RAGCodePiecePath and os.path.isfile(args.RAGCodePiecePath):
        cp_df = pd.read_csv(args.RAGCodePiecePath)
        for _, row in cp_df.iterrows():
            desc = str(row.get("Description:", "")).strip()
            code = str(row.get("Code Piece:", "")).strip()
            if desc and code and desc.lower() != "nan" and code.lower() != "nan":
                code_pieces.append({"description": desc, "code": code})
        print(f"[Init] Loaded {len(code_pieces)} code examples.", flush=True)

    # ── OpenROAD PTY ──────────────────────────────────────────────────────────
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
        print("[Init] --openroadPath not set — OpenROAD execution skipped.", flush=True)

    # ── case loop ─────────────────────────────────────────────────────────────
    passed  = 0
    results = []

    for i, (task, gt_code) in enumerate(zip(prompts, gt_codes), 1):
        task = task.strip()
        print(f"\n[{i}/{len(prompts)}] {task[:100]}", flush=True)

        state = CausalAgentState(task=task, max_budget=args.budget)

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
        lv = state.llm_result
        ctrl_history = " | ".join(
            f"{o.action}:{o.result[:50]}"
            for o in state.observations
            if not o.is_bootstrap
        )

        results.append({
            "prompt":           task,
            "ground_truth":     gt_code,
            "chain":            " -> ".join(state.chain),
            "node_apis":        state.api_summary,
            "generated_code":   state.committed_code,
            "causal_verdict":   (
                "PASS" if sv and sv.passed
                else f"FAIL(L{sv.layer_failed}): {'; '.join(sv.issues[:2])}"
                if sv else "N/A"
            ),
            "llm_verdict":      (
                f"PASS(conf={lv.confidence:.2f})" if lv and lv.passed
                else f"FAIL(conf={lv.confidence:.2f}): {'; '.join(lv.issues[:1])}"
                if lv else "not_run"
            ),
            "openroad_result":  (
                ("PASS" if ora_pass else "FAIL") if ora_pass is not None else "SKIPPED"
            ),
            "openroad_output":  _clean_output(ora_stdout)[:1000],
            "ctrl_actions":     ctrl_history,
            "steps_used":       state.step,
            "budget_used":      state.budget_used,
        })

    # ── save results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if args.openroadPath:
        print(f"OpenROAD: {passed}/{len(results)} passed")

    if args.resultPath:
        out = args.resultPath
        df  = pd.DataFrame(results)
        if out.endswith(".xlsx"):
            df.to_excel(out, index=False)
        else:
            df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved → {out}")

    print(f"Done: {len(results)} cases processed.")
    if stop_event:
        stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Causal agent runner for EDA-Corpus-v2/TestSet.xlsx"
    )
    parser.add_argument("--testSetPath",      required=True,
                        help="Path to EDA-Corpus-v2/TestSet.xlsx")
    parser.add_argument("--RAGApiPath",       default="RAGData/RAGAPIs.csv",
                        help="Path to RAGAPIs.csv")
    parser.add_argument("--RAGCodePiecePath", default="RAGData/RAGCodePiece.csv",
                        help="Path to RAGCodePiece.csv (optional)")
    parser.add_argument("--openaiKey",        required=True,
                        help="OpenAI API key")
    parser.add_argument("--model",            default="gpt-4.1-mini")
    parser.add_argument("--numCases",         default=0, type=int,
                        help="Number of cases to run (0 = all ~140)")
    parser.add_argument("--budget",           default=6, type=int,
                        help="Controller action budget per case")
    parser.add_argument("--openroadPath",     default="",
                        help="Path to openroad binary (skip execution if empty)")
    parser.add_argument("--runDir",           default="src_1_reflector",
                        help="Working dir for OpenROAD shell")
    parser.add_argument("--resultPath",       default="result/causal_corpus_run.csv",
                        help="Output file (.csv or .xlsx)")
    parser.add_argument("--promptFile",       default="",
                        help="Text file with one prompt per line — run only matching prompts")
    parser.add_argument("--loadDesignTime",   default=5,   type=int)
    parser.add_argument("--maxWaitTime",      default=120, type=int)
    args = parser.parse_args()
    RunCorpus(args)


if __name__ == "__main__":
    main()
