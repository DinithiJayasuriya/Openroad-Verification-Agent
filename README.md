# OpenROAD Verification Agent

A causal-chain–grounded verification framework for LLM-generated
[OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) Python scripts.
The agent plans the object-acquisition chain a task *must* follow, retrieves
the concrete OpenROAD APIs for each edge of that chain, constrains code
generation with those APIs, and validates the generated code against the
chain before execution.

Companion paper: `Paper_2026_ICCAD_openroad_agent.pdf`.

## Why

Off-the-shelf LLMs frequently fail OpenROAD scripting tasks because the
OpenROAD Python API requires a strict object-acquisition order
(`Tech → Design → Block → Instances → …`). Retrieval-only RAG misses the
glue calls — `design.getBlock()` is semantically orthogonal to the user
task and is almost never retrieved. This agent makes that chain explicit,
borrows verified sub-chains across tasks, and blocks bad code
*before* it touches the OpenROAD shell.

## Pipeline

A four-step pipeline (`causal_verifier/pipeline.py`):

1. **Chain extraction** (`chain_extractor.py`) — task text →
   mandatory OpenROAD type sequence.
2. **Node-specific retrieval** (`node_retriever.py`) — each edge of
   the chain → concrete API signature from `RAGAPIs.csv`, cache-aware.
3. **Constraint-injected generation** — the chain and required APIs
   are injected into the code-generation prompt.
4. **Causal validation** (`chain_validator.py`) — AST-level check
   that the generated code actually walks the chain.

Successful chains are memoized in `chain_cache.py` and borrowed by
future tasks (so `Design → Block` is resolved once, then reused).

## Layout

```
causal_verifier/
  pipeline.py              # CausalPipeline entry point
  chain_extractor.py       # Step 1
  node_retriever.py        # Step 2
  chain_validator.py       # Step 4
  chain_cache.py           # cross-task sub-chain cache
  causal_verifier.py       # static multi-layer verifier
  llm_verifier.py          # LLM-as-judge verifier
  controller.py, dispatcher.py
  run_causal_agent*.py     # evaluation drivers
  adversarial_*.xlsx       # adversarial task sets
  NeurIPS/                 # flow-level pipeline variants
Paper_2026_ICCAD_openroad_agent.pdf
```

## Quick start

```python
from causal_verifier.pipeline import CausalPipeline

pipeline = CausalPipeline(
    rag_api_path="RAGData/RAGAPIs.csv",
    cache_path="cache/chain_cache.json",
    openai_key="sk-...",
)

# Plan: extract chain + retrieve APIs + build constraint prompt
result = pipeline.plan(task)
constraint_prompt = result.constraint_prompt

# Generate code with your LLM, injecting `constraint_prompt`
#   ...

# Validate before execution
val = pipeline.validate(task, generated_code, result)
if val.passed:
    pipeline.record_success(case_id, task, result)
```

## Requirements

- Python 3.10+
- OpenROAD Python bindings (for end-to-end execution)
- `openai` (optional, for LLM chain extraction / verification)
- `sentence-transformers` (for semantic retrieval signals)
- `openpyxl` (for `.xlsx` datasets)

## Evaluation

The `run_causal_agent*.py` drivers run the pipeline against the
EDA-Corpus v2 test set and the adversarial datasets shipped in
`causal_verifier/`. The `NeurIPS/` subdirectory contains flow-level
variants (multi-stage physical-design flows).

## Citation

If you use this work, please cite the accompanying ICCAD 2026 paper
(`Paper_2026_ICCAD_openroad_agent.pdf`).
