"""run_causal_7b_corpus.py — 7B local model as generator + causal verifier pipeline.

Combines:
  - Code generation: OpenROAD-Agent-7B-3rdStage (local, same as test_iterate_github.py)
  - Verification:    Causal pipeline from run_causal_agent_corpus.py
                     (chain extraction, node-specific RAG, static+causal+LLM verification)

Flow per task:
  1. Causal extract  — GPT-4.1-mini extracts causal chain
  2. Node gate       — validate types against structured RAG
  3. Causal RAG      — node-specific retrieval (edge APIs)
  4. 7B generates    — local model produces code using chain context + RAG
  5. Causal verify   — static + causal + LLM verification
  6. Retry loop      — if verification fails, feed feedback to 7B (up to budget)
  7. Execute         — run committed code in OpenROAD

Usage:
  /mnt/ssd1/dinithi/OpenROAD-Agent/venv/bin/python causal_verifier_4_2/run_causal_7b_corpus.py \\
      --modelName src/Saved_Model/OpenROAD-Agent-7B-3rdStage \\
      --testSetPath EDA-Corpus-v2/TestSet.xlsx \\
      --RAGApiPath RAGData/RAGAPIs.csv \\
      --RAGCodePiecePath RAGData/RAGCodePiece.csv \\
      --openaiKey sk-... \\
      --openroadPath OpenROAD/build/src/openroad \\
      --resultPath result/causal_7b_corpus.xlsx \\
      --numCases 20 --budget 6
"""

import argparse
import gc
import os
import queue
import re
import sys
import threading
import time
from typing import Optional

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import numpy as np
import pandas as pd
import torch
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAUSAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CAUSAL_DIR)
sys.path.insert(0, os.path.join(_ROOT, "src_1_agentic"))
sys.path.insert(0, os.path.join(_ROOT, "src_1_reflector"))
sys.path.insert(0, os.path.join(_ROOT, "src"))  # src first — use src/util.py's answerWithRAG (CPU-safe)

from util import (
    readOpenROADOutput, runOpenROADShell, sendCommandOpenROAD,
    processCodeString, clearQueue, generate, modelUtility,
    prepareDocuments, answerWithRAG,
)
from verifier import OpenROADStaticVerifier

from causal_state    import CausalAgentState, VerifierSnapshot
from causal_verifier import CausalVerifier
from llm_verifier    import CausalLLMVerifier

from run_causal_agent import (
    run_bootstrap,
    execute_in_openroad,
    _clean_output,
    bootstrap_causal_extract,
    bootstrap_causal_rag,
    bootstrap_static_verify,
    _build_generation_context,
    _rewrite_traversal_chain,
)

try:
    from run_causal_agent import bootstrap_node_gate
except ImportError:
    bootstrap_node_gate = None

try:
    from structured_rag_gate import StructuredRAGGate
except ImportError:
    StructuredRAGGate = None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader (same as run_causal_agent_corpus.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(path: str, num_cases: int = 0, prompt_file: str = ""):
    prompt_df = pd.read_excel(path, "Prompt", header=None).rename(columns={0: "text"})
    code_df   = pd.read_excel(path, "Code",   header=None).rename(columns={0: "text"})

    filter_set = []
    if prompt_file and os.path.isfile(prompt_file):
        with open(prompt_file) as f:
            filter_set = [line.strip() for line in f if line.strip()]
        print(f"[Filter] Restricting to {len(filter_set)} prompts from {prompt_file}")

    prompts, gt_codes = [], []
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
# 7B model loader (from test_iterate_github.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_7b_model(model_name: str):
    """Load the 7B fine-tuned model and tokenizer. Returns (model, tokenizer)."""
    if "llama" in model_name.lower() or "retrained" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            pad_token='<|end_of_text|>',
            eos_token='<|eot_id|>',
            cache_dir=None, truncation=True,
            padding_side="right", trust_remote_code=True,
            device_map="balanced_low_0",
        )
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|eot_id|>", "<|end_of_text|>"]})
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            pad_token='<|endoftext|>',
            eos_token='<|im_end|>',
            cache_dir=None, truncation=True,
            padding_side="right", trust_remote_code=True,
            device_map="balanced_low_0",
        )
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>", "<|im_start|>"]})

    if "retrained" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="balanced_low_0", torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, model_name, is_trainable=False, device_map="balanced_low_0")
    elif "7b" in model_name.lower() and "agent" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            device_map="balanced_low_0", torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, model_name, is_trainable=False, device_map="balanced_low_0")
    elif "32b" in model_name.lower() and "agent" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            device_map="balanced_low_0", torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, model_name, is_trainable=False, device_map="balanced_low_0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="balanced_low_0",
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 7B generation with chain context injected into RAG
# ─────────────────────────────────────────────────────────────────────────────

def generate_with_7b(task: str, model, tokenizer, util, state: CausalAgentState,
                     rag_context: str, max_new_tokens: int = 512,
                     wrong_code: str = "", error_msg: str = "") -> str:
    """Generate code using the 7B model.

    Builds the prompt with:
      - Chain context from causal extraction (injected into RAG context)
      - Original RAG context from answerWithRAG()
      - Verifier feedback (if retrying)
    """
    # Build chain context string from causal pipeline
    chain_ctx = ""
    if state.chain and state.edge_apis:
        chain_ctx = _build_generation_context(
            state.chain, state.edge_apis,
            paths=state.paths or None,
            all_edges=state.all_edges or None,
            action_node=state.action_node,
        )

    # Combine chain context with RAG context
    combined_context = ""
    if chain_ctx:
        combined_context += f"Causal Chain (follow this acquisition order):\n{chain_ctx}\n\n"
    if rag_context:
        combined_context += rag_context

    # Build prompt using the model's template
    if wrong_code and error_msg:
        # Retry: include wrong code and error
        if combined_context:
            prompt = util.ragWrongCodePromptTemplateWithContext.format(
                question=task,
                context=combined_context,
                system_prompt=util.systemPrompt,
                wrongCode=wrong_code,
                message=error_msg,
            )
        else:
            prompt = util.ragWrongCodePromptTemplateWithoutContext.format(
                question=task,
                system_prompt=util.systemPrompt,
                wrongCode=wrong_code,
                message=error_msg,
            )
    else:
        # First attempt
        if combined_context:
            prompt = util.ragPromptTemplateWithContext.format(
                question=task,
                context=combined_context,
                system_prompt=util.systemPrompt,
            )
        else:
            prompt = util.ragPromptTemplateWithoutContext.format(
                question=task,
                system_prompt=util.systemPrompt,
            )

    decoded = generate(
        model=model, tokenizer=tokenizer, prompt=prompt,
        pastKeyValues=DynamicCache(),
        temperature=0.3, topP=0.9, maxNewTokens=max_new_tokens,
    )

    # Extract code from model output
    code = decoded.split("```python")[-1].split("```")[0]
    return code.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Verifier snapshot helper
# ─────────────────────────────────────────────────────────────────────────────

def _vresult_to_snapshot(vr) -> VerifierSnapshot:
    return VerifierSnapshot(
        passed       = vr.passed,
        layer_failed = vr.layer_failed,
        issues       = vr.issues,
        feedback     = vr.feedback,
        confidence   = 1.0 if vr.passed else 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-case runner: causal pipeline + 7B generation + verify + retry
# ─────────────────────────────────────────────────────────────────────────────

def run_case_7b(task: str, state: CausalAgentState,
                model_7b, tokenizer, util,
                api_key: str, openai_model: str,
                embed_model, metadata: list, embeddings,
                rag_embed_model, rag_embeddings, rag_allSplits, rag_allDict,
                static_verifier, causal_verifier,
                llm_verifier=None,
                max_new_tokens: int = 512) -> CausalAgentState:
    """Run causal chain extraction + 7B generation + verification loop.

    Returns state with .tool_calls_count set (total LLM + verifier invocations).
    """
    tool_calls = 0  # track all LLM / verifier invocations

    print(f"\n{'─'*60}", flush=True)
    print(f"TASK: {task}", flush=True)
    print(f"BUDGET: {state.max_budget}", flush=True)

    # ── Step 1: Causal extract (GPT-4.1-mini) ────────────────────────────────
    bootstrap_causal_extract(state, api_key, openai_model)
    tool_calls += 1  # OpenAI: chain extraction
    if state.chain:
        state.chain = _rewrite_traversal_chain(state.task, state.chain)

    # ── Step 2: Causal RAG (node-specific retrieval) ──────────────────────────
    bootstrap_causal_rag(state, embed_model, metadata, embeddings,
                         api_key=api_key, model=openai_model)

    # ── Get original RAG context (same as test_iterate_github.py) ─────────────
    rag_context = answerWithRAG(
        task, rag_embeddings, rag_embed_model, rag_allSplits, rag_allDict,
    )

    # ── Step 3: 7B generates code ────────────────────────────────────────────
    code = generate_with_7b(
        task, model_7b, tokenizer, util, state,
        rag_context=rag_context, max_new_tokens=max_new_tokens,
    )
    tool_calls += 1  # 7B: generation
    state.current_code = code
    if code:
        state.code_history.append(code)
    state.add_bootstrap_obs("7b_generate", f"generated {len(code)} chars")
    print(f"  [7b_generate] {len(code)} chars", flush=True)

    # ── Step 4: Causal verify ────────────────────────────────────────────────
    bootstrap_static_verify(state, static_verifier, causal_verifier)
    tool_calls += 1  # static + causal verifier

    # If bootstrap passed — run LLM verifier, then commit
    if state.static_result and state.static_result.passed:
        if llm_verifier is not None and state.current_code:
            from dispatcher import _build_chain_context
            chain_ctx = _build_chain_context(state.chain, state.edge_apis)
            lv_snap   = llm_verifier.verify(task, state.current_code, chain_ctx)
            tool_calls += 1  # OpenAI: LLM verifier
            state.llm_result = lv_snap
            if lv_snap.passed:
                print(f"  [verify] PASS + LLM PASS (conf={lv_snap.confidence:.2f})", flush=True)
                state.committed_code = state.best_code or state.current_code
                state.committed = True
                state.tool_calls_count = tool_calls
                return state
            else:
                print(f"  [verify] static PASS but LLM FAIL (conf={lv_snap.confidence:.2f})", flush=True)
        else:
            print(f"  [verify] bootstrap PASS — committing", flush=True)
            state.committed_code = state.best_code or state.current_code
            state.committed = True
            state.tool_calls_count = tool_calls
            return state

    # ── Step 5: Retry loop — feed verifier feedback to 7B ────────────────────
    attempt = 1
    while not state.committed and state.budget_remaining > 0:
        attempt += 1

        # Build feedback from verifier
        feedback_parts = []
        if state.static_result and not state.static_result.passed:
            feedback_parts.append(f"Static verifier failed (L{state.static_result.layer_failed}): "
                                  f"{'; '.join(state.static_result.issues[:3])}")
            if state.static_result.feedback:
                feedback_parts.append(state.static_result.feedback)
        if state.llm_result and not state.llm_result.passed:
            feedback_parts.append(f"LLM verifier failed: {'; '.join(state.llm_result.issues[:2])}")
            if state.llm_result.feedback:
                feedback_parts.append(state.llm_result.feedback)

        error_msg = "\n".join(feedback_parts) if feedback_parts else "Code verification failed."

        print(f"\n  [retry {attempt}] budget_left={state.budget_remaining}", flush=True)
        print(f"    feedback: {error_msg[:150]}", flush=True)

        # Re-generate with 7B
        code = generate_with_7b(
            task, model_7b, tokenizer, util, state,
            rag_context=rag_context,
            max_new_tokens=max_new_tokens,
            wrong_code=state.current_code,
            error_msg=error_msg,
        )
        tool_calls += 1  # 7B: re-generation
        state.current_code = code
        if code:
            state.code_history.append(code)
        state.budget_used += 1

        # Re-verify
        bootstrap_static_verify(state, static_verifier, causal_verifier)
        tool_calls += 1  # static + causal re-verify

        if state.static_result and state.static_result.passed:
            if llm_verifier is not None and state.current_code:
                from dispatcher import _build_chain_context
                chain_ctx = _build_chain_context(state.chain, state.edge_apis)
                lv_snap   = llm_verifier.verify(task, state.current_code, chain_ctx)
                tool_calls += 1  # OpenAI: LLM re-verify
                state.llm_result = lv_snap
                if lv_snap.passed:
                    print(f"  [retry {attempt}] PASS + LLM PASS (conf={lv_snap.confidence:.2f})", flush=True)
                    state.committed_code = state.best_code or state.current_code
                    state.committed = True
                    state.tool_calls_count = tool_calls
                    return state
            else:
                print(f"  [retry {attempt}] PASS — committing", flush=True)
                state.committed_code = state.best_code or state.current_code
                state.committed = True
                state.tool_calls_count = tool_calls
                return state

    # Budget exhausted — commit best available
    if not state.committed:
        state.committed_code = state.best_code or state.current_code
        state.committed = True
        print(f"  [loop] budget exhausted — committing best", flush=True)

    state.tool_calls_count = tool_calls
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def RunCorpus(args):
    # ── load dataset ──────────────────────────────────────────────────────────
    prompts, gt_codes = load_corpus(args.testSetPath, args.numCases, args.promptFile)
    print(f"\nLoaded  : {len(prompts)} prompts from {args.testSetPath}")
    print(f"Generator: {args.modelName} (local 7B)")
    print(f"Verifier : causal pipeline (OpenAI model: {args.openaiModel})")
    print("=" * 70, flush=True)

    # ── Load 7B model ────────────────────────────────────────────────────────
    print("\nLoading 7B model...", flush=True)
    model_7b, tokenizer = load_7b_model(args.modelName)
    util = modelUtility(args.modelName)
    print(f"  [7B] Model loaded: {args.modelName}", flush=True)

    # ── Causal RAG setup (all-MiniLM-L6-v2 for chain extraction) ─────────────
    print("\nLoading embedding models...", flush=True)
    embed_model_causal = SentenceTransformer("all-MiniLM-L6-v2")

    rag_df   = pd.read_csv(args.RAGApiPath)
    metadata = []
    documents_causal = []
    for _, row in rag_df.iterrows():
        desc = str(row.get("Description:", "")).strip()
        if not desc or desc.lower() == "nan":
            continue
        documents_causal.append(f"OpenROAD Python API Description:{desc}")
        metadata.append(row.to_dict())
    print(f"  [Causal RAG] Encoding {len(documents_causal)} API entries...", flush=True)
    embeddings_causal = embed_model_causal.encode(
        documents_causal, convert_to_tensor=True, show_progress_bar=False,
    )

    # ── Original RAG setup (mxbai-embed-large-v1 for 7B generator) ───────────
    print("  [7B RAG] Loading mxbai-embed-large-v1...", flush=True)
    embed_model_7b = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    apiDf = pd.read_csv(args.RAGApiPath)
    apiDocuments, apiDocumentsDict = prepareDocuments(df=apiDf)
    templateDf = pd.read_csv(args.RAGCodePiecePath)
    templateDocuments, templateDocumentsDict = prepareDocuments(df=templateDf, api=False)
    allSplits = apiDocuments + templateDocuments
    allDict   = {**apiDocumentsDict, **templateDocumentsDict}
    rag_embeddings = embed_model_7b.encode(allSplits)
    print(f"  [7B RAG] {len(allSplits)} entries encoded.", flush=True)

    # ── Verifiers ─────────────────────────────────────────────────────────────
    static_verifier = OpenROADStaticVerifier(args.RAGApiPath)
    causal_ver      = CausalVerifier(metadata=metadata)
    llm_ver         = CausalLLMVerifier(api_key=args.openaiKey, model=args.openaiModel,
                                         fail_open=True)
    print("[Init] Static + causal + LLM semantic verifiers ready.", flush=True)

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

    # ── Case loop ─────────────────────────────────────────────────────────────
    passed  = 0
    results = []
    all_latencies = []

    for i, (task, gt_code) in enumerate(zip(prompts, gt_codes), 1):
        task = task.strip()
        print(f"\n[{i}/{len(prompts)}] {task[:100]}", flush=True)
        task_start = time.time()

        state = CausalAgentState(task=task, max_budget=args.budget)

        state = run_case_7b(
            task, state,
            model_7b=model_7b, tokenizer=tokenizer, util=util,
            api_key=args.openaiKey, openai_model=args.openaiModel,
            embed_model=embed_model_causal, metadata=metadata,
            embeddings=embeddings_causal,
            rag_embed_model=embed_model_7b,
            rag_embeddings=rag_embeddings, rag_allSplits=allSplits,
            rag_allDict=allDict,
            static_verifier=static_verifier, causal_verifier=causal_ver,
            llm_verifier=llm_ver,
            max_new_tokens=512,
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

        task_latency = time.time() - task_start
        all_latencies.append(task_latency)

        sv = state.static_result
        lv = state.llm_result

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
            "budget_used":      state.budget_used,
            "tool_calls":       getattr(state, 'tool_calls_count', 0),
            "latency_s":        round(task_latency, 2),
        })

        gc.collect()

    # ── Save results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if args.openroadPath:
        print(f"OpenROAD: {passed}/{len(results)} passed")
    if all_latencies:
        print(f"Avg latency: {sum(all_latencies)/len(all_latencies):.1f}s / case")

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
        description="7B generator + causal verifier for EDA-Corpus-v2/TestSet.xlsx"
    )
    parser.add_argument("--modelName",         required=True,
                        help="Path to 7B model (e.g. src/Saved_Model/OpenROAD-Agent-7B-3rdStage)")
    parser.add_argument("--testSetPath",       required=True,
                        help="Path to EDA-Corpus-v2/TestSet.xlsx")
    parser.add_argument("--RAGApiPath",        default="RAGData/RAGAPIs.csv")
    parser.add_argument("--RAGCodePiecePath",  default="RAGData/RAGCodePiece.csv")
    parser.add_argument("--openaiKey",         required=True,
                        help="OpenAI API key (for chain extraction + LLM verifier)")
    parser.add_argument("--openaiModel",       default="gpt-4.1-mini",
                        help="OpenAI model for chain extraction + verification")
    parser.add_argument("--numCases",          default=0, type=int,
                        help="Number of cases (0 = all)")
    parser.add_argument("--budget",            default=6, type=int,
                        help="Max retry budget per case")
    parser.add_argument("--openroadPath",      default="",
                        help="Path to openroad binary")
    parser.add_argument("--runDir",            default="src_1_reflector")
    parser.add_argument("--resultPath",        default="result/causal_7b_corpus.xlsx")
    parser.add_argument("--promptFile",        default="",
                        help="Text file to filter prompts")
    parser.add_argument("--loadDesignTime",    default=5, type=int)
    parser.add_argument("--maxWaitTime",       default=120, type=int)
    args = parser.parse_args()
    RunCorpus(args)


if __name__ == "__main__":
    main()
