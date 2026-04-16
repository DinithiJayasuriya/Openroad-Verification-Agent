"""
adversarial_gen.py — Creator-Solver Evolution Pipeline for Verifier Evaluation.

Pipeline (creator-solver style):
  1. Load 30 seed prompts — complex_prompt strings only:
       10 from EDA Corpus  (DB-v2.xlsx,    single-step baseline)
       10 from Dataset 2   (dataset2.xlsx,  3-step mutation tasks)
       10 from Dataset 5   (dataset5.xlsx,  4-step tasks, passing subset)
  2. Creator (LLM-A) generates K harder variants per seed (small per-seed batches,
     so output always fits within the 4096-token budget — no truncation).
  3. Solver (LLM-B) runs each variant through the full causal pipeline.
  4. Categorise each variant:
       too_easy   — passed AND budget_used ≤ 1 (no repair needed)
       sweet_spot — passed with repair OR failed at L2/L3/L5 with specific issues
       too_hard   — L1 format failure OR budget exhausted with no useful signal
  5. Sweet-spot tasks become seeds for the next evolution round.
  6. Stop when ≥ target_n (default 100) sweet-spot tasks are collected.

Output: one Excel file, two sheets:
  "all_tasks"  — every generated task with category + solver metadata
  "sweet_spot" — the curated target_n tasks for verifier evaluation

Usage:
    python causal_verifier/adversarial_gen.py \\
        --ragApi    RAGData/RAGAPIs.csv \\
        --edaCorpus EDA-Corpus-v2/DB-v2.xlsx \\
        --dataset2  causal_verifier/dataset2.xlsx \\
        --dataset5  causal_verifier/dataset5.xlsx \\
        --openaiKey sk-... \\
        --model     gpt-4.1-mini \\
        --variantsK 3 \\
        --targetN   100 \\
        --maxRounds 10 \\
        --output    causal_verifier/adversarial_evolved.xlsx
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAUSAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src_1_reflector"))
sys.path.insert(0, os.path.join(_ROOT, "src_1_agentic"))
sys.path.insert(0, _CAUSAL_DIR)

from verifier        import OpenROADStaticVerifier
from causal_state    import CausalAgentState
from causal_verifier import CausalVerifier
from llm_verifier    import CausalLLMVerifier
from run_causal_agent import run_case
from llm_adversary   import LLMAdversary


# ── Dataset 5 passing indices (from best run: gpt-4-1-mini causal_L4) ────────
# Indices 12 (INV prefix) and 18 (BUF prefix) failed — excluded.
_DS5_PASSING_INDICES = [0, 2, 3, 5, 6, 10, 13, 14, 16, 17]


# ─────────────────────────────────────────────────────────────────────────────
# Seed loading
# ─────────────────────────────────────────────────────────────────────────────

def load_seeds_from_checkpoint(checkpoint_path: str) -> List[str]:
    """Load seeds from the sweet_spot sheet of a previous run's output file.

    Used to resume: run 1 stops at 50 sweet-spots, run 2 calls this to use
    those 50 as seeds and continue towards 100.
    """
    df = pd.read_excel(checkpoint_path, sheet_name="sweet_spot")
    prompts = [str(r["complex_prompt"]).strip() for _, r in df.iterrows()
               if pd.notna(r.get("complex_prompt")) and
               str(r["complex_prompt"]).strip().lower() != "nan"]
    print(f"[seeds] Checkpoint: {len(prompts)} sweet-spot prompts loaded "
          f"from {checkpoint_path}", flush=True)
    return prompts


def load_seeds(eda_corpus_path: str, dataset2_path: str, dataset5_path: str,
               n_eda: int = 10, n_ds2: int = 10, n_ds5: int = 10,
               seed: int = 42) -> List[str]:
    """Return a list of complex_prompt strings (seeds).

    Sources:
      - EDA Corpus: baseline single-step prompts from prompt0 column.
      - Dataset 2:  3-step mutation tasks — complex_prompt only.
      - Dataset 5:  4-step tasks — complex_prompt only, from passing subset.

    Returns only the complex_prompt string; steps are intentionally discarded
    so all seeds share a uniform format regardless of their origin.
    """
    rng = random.Random(seed)
    seeds: List[str] = []

    # EDA Corpus — sample n_eda from prompt0 column
    eda_df = pd.read_excel(eda_corpus_path)
    eda_prompts = [str(p).strip() for p in eda_df["prompt0"].dropna()
                   if str(p).strip() and str(p).strip().lower() != "nan"]
    sampled_eda = rng.sample(eda_prompts, min(n_eda, len(eda_prompts)))
    seeds.extend(sampled_eda)
    print(f"[seeds] EDA Corpus: {len(sampled_eda)} prompts loaded", flush=True)

    # Dataset 2 — all complex_prompts (10 tasks)
    ds2_df = pd.read_excel(dataset2_path)
    ds2_prompts = [str(r["complex_prompt"]).strip() for _, r in ds2_df.iterrows()
                   if pd.notna(r.get("complex_prompt")) and
                   str(r["complex_prompt"]).strip().lower() != "nan"]
    sampled_ds2 = rng.sample(ds2_prompts, min(n_ds2, len(ds2_prompts)))
    seeds.extend(sampled_ds2)
    print(f"[seeds] Dataset 2:  {len(sampled_ds2)} prompts loaded", flush=True)

    # Dataset 5 — complex_prompts from passing indices only
    ds5_df = pd.read_excel(dataset5_path)
    ds5_passing = [str(ds5_df.iloc[i]["complex_prompt"]).strip()
                   for i in _DS5_PASSING_INDICES
                   if i < len(ds5_df)]
    sampled_ds5 = rng.sample(ds5_passing, min(n_ds5, len(ds5_passing)))
    seeds.extend(sampled_ds5)
    print(f"[seeds] Dataset 5:  {len(sampled_ds5)} prompts loaded", flush=True)

    print(f"[seeds] Total seeds: {len(seeds)}", flush=True)
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Solver — wraps the causal pipeline (unchanged from original design)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    task_dict:      Dict
    passed:         bool
    layer_failed:   int
    issues:         List[str] = field(default_factory=list)
    api_diffs:      List[Dict] = field(default_factory=list)
    budget_used:    int  = 0
    generated_code: str  = ""
    causal_verdict: str  = ""


class AdversarialSolver:
    """LLM-B: runs the full causal pipeline on a task dict."""

    def __init__(self, api_key: str, model: str, rag_api_path: str,
                 rag_code_piece_path: str = "", budget: int = 6):
        self.api_key = api_key
        self.model   = model
        self.budget  = budget

        print("  [solver] Loading embedding model...", flush=True)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        rag_df = pd.read_csv(rag_api_path)
        self.metadata  = []
        documents      = []
        for _, row in rag_df.iterrows():
            desc = str(row.get("Description:", "")).strip()
            if not desc or desc.lower() == "nan":
                desc = str(row.get("Function Name:", "")).strip()
            documents.append(f"OpenROAD Python API Description:{desc}")
            self.metadata.append(row.to_dict())
        self.embeddings = self.embed_model.encode(
            documents, convert_to_tensor=True, show_progress_bar=False
        )
        print(f"  [solver] RAG: {len(self.metadata)} entries encoded.", flush=True)

        self.static_ver = OpenROADStaticVerifier(rag_api_path)
        self.causal_ver = CausalVerifier(metadata=self.metadata)
        self.llm_ver    = CausalLLMVerifier(api_key=api_key, model=model, fail_open=True)

        self.code_pieces = []
        if rag_code_piece_path and os.path.isfile(rag_code_piece_path):
            cp_df = pd.read_csv(rag_code_piece_path)
            for _, row in cp_df.iterrows():
                d = str(row.get("Description:", "")).strip()
                c = str(row.get("Code Piece:", "")).strip()
                if d and c and d.lower() != "nan":
                    self.code_pieces.append({"description": d, "code": c})

    def solve(self, task_dict: Dict) -> SolverResult:
        task  = _task_to_prompt(task_dict)
        state = CausalAgentState(task=task, max_budget=self.budget)
        state = run_case(
            task, state,
            api_key=self.api_key, model=self.model,
            embed_model=self.embed_model,
            metadata=self.metadata,
            embeddings=self.embeddings,
            static_verifier=self.static_ver,
            causal_verifier=self.causal_ver,
            code_pieces=self.code_pieces,
            llm_verifier=self.llm_ver,
        )

        sv = state.static_result
        lv = state.llm_result

        if sv is None:
            return SolverResult(task_dict=task_dict, passed=False, layer_failed=0,
                                issues=["no verifier result"], budget_used=state.budget_used)

        if not sv.passed:
            passed, layer_failed, issues = False, sv.layer_failed, sv.issues
            api_diffs_src = sv.api_diffs or []
        elif lv is not None and not lv.passed:
            passed, layer_failed, issues = False, 5, lv.issues
            api_diffs_src = []
        else:
            passed, layer_failed, issues = True, 0, []
            api_diffs_src = []

        api_diffs = []
        for d in api_diffs_src:
            try:
                api_diffs.append({
                    "src":          getattr(d, "src_type",     ""),
                    "tgt":          getattr(d, "tgt_type",     ""),
                    "code_methods": getattr(d, "code_methods", ""),
                    "rag_method":   getattr(d, "rag_method",   ""),
                })
            except Exception:
                pass

        verdict = ("PASS" if passed
                   else f"FAIL(L{layer_failed}): {'; '.join(issues[:2])}")

        return SolverResult(
            task_dict     = task_dict,
            passed        = passed,
            layer_failed  = layer_failed,
            issues        = issues,
            api_diffs     = api_diffs,
            budget_used   = state.budget_used,
            generated_code= state.committed_code or state.current_code,
            causal_verdict= verdict,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Categoriser
# ─────────────────────────────────────────────────────────────────────────────

# Layers that carry meaningful verifier signal (API / causal / semantic)
_INFORMATIVE_LAYERS = {2, 3, 5}


def categorize(result: SolverResult, max_budget: int) -> str:
    """Return one of: 'too_easy' | 'sweet_spot' | 'too_hard'.

    too_easy:
      Passed AND budget_used ≤ 1 — no repair needed; doesn't exercise the
      verifier in any meaningful way.

    too_hard:
      - L1 failure (format / AST parse error) — task is structurally malformed.
      - No verifier result at all.
      - Budget exhausted AND the failure layer carries no useful signal
        (i.e. not L2/L3/L5).

    sweet_spot:
      Everything else:
      - Passed but required at least 2 budget steps (repair happened).
      - Failed at an informative layer (L2/L3/L5) with specific issues.
    """
    if not result.passed and result.layer_failed == 0:
        # No verifier result — pipeline error, not useful
        return "too_hard"

    if result.layer_failed == 1:
        # Structural / format failure — task is malformed
        return "too_hard"

    if result.passed:
        if result.budget_used <= 1:
            return "too_easy"
        return "sweet_spot"     # passed after repair

    # Not passed
    if result.budget_used >= max_budget and result.layer_failed not in _INFORMATIVE_LAYERS:
        return "too_hard"       # exhausted budget, no useful signal

    if result.layer_failed in _INFORMATIVE_LAYERS:
        return "sweet_spot"     # specific, localised failure — informative

    return "too_hard"


# ─────────────────────────────────────────────────────────────────────────────
# Evolution loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evolution_loop(
    adversary:    LLMAdversary,
    solver:       AdversarialSolver,
    seeds:        List[str],
    variants_k:   int = 3,
    target_n:     int = 100,
    max_rounds:   int = 10,
    target_steps: int = 4,
) -> List[Dict]:
    """Run the creator-solver evolution loop.

    Round 0: generate variants_k variants from each seed → solve → categorise.
    Round 1+: sweet-spot tasks from the previous round become the new seeds.
    Stops when ≥ target_n sweet-spot tasks are collected or max_rounds reached.

    Returns a list of all row dicts (all_tasks sheet).
    """
    all_rows:      List[Dict] = []
    sweet_spots:   List[Dict] = []    # accumulates across all rounds
    current_seeds: List[str]  = list(seeds)

    for rnd in range(max_rounds):
        if not current_seeds:
            print(f"[evo] Round {rnd}: no seeds — stopping.", flush=True)
            break
        if len(sweet_spots) >= target_n:
            print(f"[evo] Reached {len(sweet_spots)} sweet-spot tasks — done.", flush=True)
            break

        print(f"\n{'='*60}", flush=True)
        print(f"[evo] Round {rnd} — {len(current_seeds)} seeds × {variants_k} variants "
              f"= up to {len(current_seeds) * variants_k} candidates", flush=True)
        print(f"[evo] Sweet-spot so far: {len(sweet_spots)}/{target_n}", flush=True)

        new_sweet_seeds: List[str] = []

        for s_idx, seed_prompt in enumerate(current_seeds, 1):
            print(f"\n  [evo] Seed {s_idx}/{len(current_seeds)}: "
                  f"{seed_prompt[:70]}", flush=True)

            # Creator: generate variants_k harder variants
            variants = adversary.generate_variants(
                seed_prompt, n_variants=variants_k, target_steps=target_steps
            )
            print(f"  [creator] Generated {len(variants)} variants", flush=True)

            for v_idx, task_dict in enumerate(variants, 1):
                prompt_preview = task_dict.get("complex_prompt", "")[:70]
                print(f"\n    [solver] Variant {v_idx}/{len(variants)}: "
                      f"{prompt_preview}", flush=True)

                result   = solver.solve(task_dict)
                category = categorize(result, solver.budget)

                status = "PASS" if result.passed else f"FAIL(L{result.layer_failed})"
                print(f"    [solver] → {status} | budget={result.budget_used} "
                      f"| category={category}", flush=True)

                row = {
                    **task_dict,
                    "seed_prompt":    seed_prompt,
                    "round":          rnd,
                    "category":       category,
                    "solver_passed":  result.passed,
                    "layer_failed":   result.layer_failed,
                    "issues":         "; ".join(result.issues[:3]),
                    "causal_verdict": result.causal_verdict,
                    "budget_used":    result.budget_used,
                    "generated_code": result.generated_code,
                    "api_diffs":      json.dumps(result.api_diffs),
                }
                all_rows.append(row)

                if category == "sweet_spot":
                    sweet_spots.append(row)
                    new_sweet_seeds.append(task_dict.get("complex_prompt", ""))
                    if len(sweet_spots) >= target_n:
                        break  # inner variant loop

            if len(sweet_spots) >= target_n:
                break  # seed loop

        n_easy  = sum(1 for r in all_rows if r["round"] == rnd and r["category"] == "too_easy")
        n_sweet = sum(1 for r in all_rows if r["round"] == rnd and r["category"] == "sweet_spot")
        n_hard  = sum(1 for r in all_rows if r["round"] == rnd and r["category"] == "too_hard")
        print(f"\n[evo] Round {rnd} summary: "
              f"too_easy={n_easy}, sweet_spot={n_sweet}, too_hard={n_hard} | "
              f"total sweet_spots={len(sweet_spots)}", flush=True)

        # Sweet-spot tasks become next-round seeds
        current_seeds = new_sweet_seeds

    return all_rows, sweet_spots


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _task_to_prompt(task_dict: Dict) -> str:
    """Flatten step_1..step_N into a numbered prompt string for the solver."""
    steps = [task_dict.get(f"step_{i}", "") for i in range(1, 6)
             if task_dict.get(f"step_{i}", "")]
    if not steps:
        return task_dict.get("complex_prompt", "")
    return " ".join(f"{i+1}. {s}" for i, s in enumerate(steps))


def _save_dataset(all_rows: List[Dict], sweet_spots: List[Dict],
                  output_path: str) -> None:
    task_cols = ["complex_prompt", "step_1", "step_2", "step_3", "step_4"]
    meta_cols = ["seed_prompt", "round", "category", "solver_passed",
                 "layer_failed", "issues", "causal_verdict",
                 "budget_used", "api_diffs", "generated_code"]

    def _ordered_df(rows):
        df = pd.DataFrame(rows)
        ordered = [c for c in task_cols if c in df.columns] + \
                  [c for c in meta_cols  if c in df.columns]
        return df[ordered]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _ordered_df(all_rows).to_excel(writer, sheet_name="all_tasks",  index=False)
        _ordered_df(sweet_spots).to_excel(writer, sheet_name="sweet_spot", index=False)

    print(f"\n[save] {len(all_rows)} total rows → {output_path} "
          f"(sheets: all_tasks, sweet_spot)", flush=True)
    print(f"[save] Sweet-spot tasks saved: {len(sweet_spots)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Creator-Solver evolution pipeline for verifier evaluation dataset."
    )
    parser.add_argument("--ragApi",      required=True,
                        help="Path to RAGAPIs.csv")
    parser.add_argument("--edaCorpus",   default="EDA-Corpus-v2/DB-v2.xlsx",
                        help="Path to EDA Corpus DB-v2.xlsx")
    parser.add_argument("--dataset2",    default="causal_verifier/dataset2.xlsx",
                        help="Path to dataset2.xlsx (3-step mutation tasks)")
    parser.add_argument("--dataset5",    default="causal_verifier/dataset5.xlsx",
                        help="Path to dataset5.xlsx (4-step tasks, passing subset used)")
    parser.add_argument("--ragCodePiece", default="",
                        help="Optional path to RAGCodePiece.csv")
    parser.add_argument("--openaiKey",   required=True,
                        help="OpenAI API key")
    parser.add_argument("--model",       default="gpt-4.1-mini",
                        help="LLM model for both creator and solver")
    parser.add_argument("--variantsK",   default=3, type=int,
                        help="Harder variants generated per seed per round (default 3)")
    parser.add_argument("--targetN",     default=100, type=int,
                        help="Target number of sweet-spot tasks to collect (default 100)")
    parser.add_argument("--maxRounds",   default=10, type=int,
                        help="Maximum evolution rounds (default 10)")
    parser.add_argument("--targetSteps", default=4, type=int,
                        help="Number of steps in generated tasks (default 4)")
    parser.add_argument("--budget",      default=6, type=int,
                        help="Solver action budget per task (default 6)")
    parser.add_argument("--nEda",        default=10, type=int,
                        help="Seed prompts from EDA Corpus (default 10)")
    parser.add_argument("--nDs2",        default=10, type=int,
                        help="Seed prompts from Dataset 2 (default 10)")
    parser.add_argument("--nDs5",        default=10, type=int,
                        help="Seed prompts from Dataset 5 passing subset (default 10)")
    parser.add_argument("--seedRng",     default=42, type=int,
                        help="Random seed for seed sampling (default 42)")
    parser.add_argument("--resumeFrom",  default="",
                        help="Path to a previous run's output Excel file. "
                             "Seeds are loaded from its sweet_spot sheet instead "
                             "of the original seed files. Use this to continue "
                             "from a 50-task checkpoint towards 100.")
    parser.add_argument("--output",      default="causal_verifier/adversarial_evolved.xlsx",
                        help="Output Excel path")
    args = parser.parse_args()

    # ── Load seeds ────────────────────────────────────────────────────────────
    print("[init] Loading seeds...", flush=True)
    if args.resumeFrom:
        seeds = load_seeds_from_checkpoint(args.resumeFrom)
    else:
        seeds = load_seeds(
            eda_corpus_path = args.edaCorpus,
            dataset2_path   = args.dataset2,
            dataset5_path   = args.dataset5,
            n_eda           = args.nEda,
            n_ds2           = args.nDs2,
            n_ds5           = args.nDs5,
            seed            = args.seedRng,
        )

    # ── Init creator ──────────────────────────────────────────────────────────
    print("\n[init] Initialising creator (LLM-A)...", flush=True)
    adversary = LLMAdversary(
        api_key      = args.openaiKey,
        model        = args.model,
        rag_api_path = args.ragApi,
    )

    # ── Init solver ───────────────────────────────────────────────────────────
    print("\n[init] Initialising solver (LLM-B)...", flush=True)
    solver = AdversarialSolver(
        api_key             = args.openaiKey,
        model               = args.model,
        rag_api_path        = args.ragApi,
        rag_code_piece_path = args.ragCodePiece,
        budget              = args.budget,
    )

    # ── Run evolution ─────────────────────────────────────────────────────────
    all_rows, sweet_spots = run_evolution_loop(
        adversary    = adversary,
        solver       = solver,
        seeds        = seeds,
        variants_k   = args.variantsK,
        target_n     = args.targetN,
        max_rounds   = args.maxRounds,
        target_steps = args.targetSteps,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    _save_dataset(all_rows, sweet_spots, args.output)

    print(f"\n[done] Final: {len(sweet_spots)}/{args.targetN} sweet-spot tasks collected "
          f"across {len(all_rows)} total candidates.")


if __name__ == "__main__":
    main()
