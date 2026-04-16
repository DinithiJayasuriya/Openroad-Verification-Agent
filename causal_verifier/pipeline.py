"""pipeline.py — CausalPipeline: full 4-step causal chain orchestrator.

Steps
-----
1. CausalChainExtractor   — task text → mandatory type-sequence
2. NodeRetriever          — chain edges → per-edge API signatures (cache-aware)
3. Code generation prompt — inject causal constraints into the generation prompt
4. CausalChainValidator   — verify generated code obeys the chain
5. ChainCache.record      — cache successful chains for future borrowing

The pipeline is stateless between calls except for the ChainCache.

Usage (standalone)
------------------
    pipeline = CausalPipeline(
        rag_api_path="RAGData/RAGAPIs.csv",
        cache_path="cache/chain_cache.json",  # optional
        openai_key="sk-...",                  # optional, for LLM chain extraction
    )

    # Before code generation: get the constraint prompt to inject
    result = pipeline.plan(task)
    constraint_prompt = result.constraint_prompt

    # After code generation: validate
    val = pipeline.validate(task, generated_code, result)
    if val.passed:
        pipeline.record_success(case_id, task, result)

    # Access the full pipeline result
    print(result.chain)       # CausalChain
    print(result.node_apis)   # List[NodeAPIEntry]
    print(result.borrowed_edges)  # edges borrowed from cache
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .chain_extractor  import CausalChainExtractor, CausalChain
from .node_retriever   import NodeRetriever, NodeAPIEntry, _normalize, _short
from .chain_validator  import CausalChainValidator, CausalValidationResult
from .chain_cache      import ChainCache


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class CausalPipelineResult:
    """Returned by CausalPipeline.plan().  Passed to validate() and record_success()."""
    chain:            CausalChain
    node_apis:        List[NodeAPIEntry]
    constraint_prompt: str          # ready-to-inject prompt section
    borrowed_edges:   List[Tuple[str, str]] = field(default_factory=list)
    validation:       Optional[CausalValidationResult] = None

    @property
    def has_cached_edges(self) -> bool:
        return len(self.borrowed_edges) > 0

    def summary(self) -> str:
        borrowed = len(self.borrowed_edges)
        total    = len(self.node_apis)
        lines = [
            f"Chain : {self.chain}",
            f"Edges : {total} ({borrowed} borrowed from cache)",
        ]
        for api in self.node_apis:
            tag = " [CACHED]" if api.source == "cache" else ""
            lines.append(
                f"  {_short(api.source_type)} → {_short(api.target_type)}"
                f"  via {api.method_name}(){tag}"
            )
        if self.validation:
            lines.append(str(self.validation))
        return "\n".join(lines)


# ── Pipeline ─────────────────────────────────────────────────────────────────

class CausalPipeline:
    """
    Orchestrates the four-step causal pipeline.

    Parameters
    ----------
    rag_api_path : str
        Path to RAGAPIs.csv.
    cache_path : str, optional
        Path to the JSON chain-cache file.  Created on first successful run.
    openai_key : str, optional
        OpenAI key for LLM-based chain extraction fallback.
    openai_model : str
        Model for LLM extraction fallback (default gpt-4.1-mini).
    """

    def __init__(
        self,
        rag_api_path:  str,
        cache_path:    Optional[str] = None,
        openai_key:    Optional[str] = None,
        openai_model:  str = "gpt-4.1-mini",
    ):
        self.extractor  = CausalChainExtractor(openai_key, openai_model)
        self.retriever  = NodeRetriever(rag_api_path)
        self.validator  = CausalChainValidator()
        self.cache      = ChainCache(cache_path)

    # ── Step 1+2: Plan ────────────────────────────────────────────────────────

    def plan(self, task: str) -> CausalPipelineResult:
        """
        Extract the causal chain for *task* and retrieve node-specific APIs.

        Returns a CausalPipelineResult with:
          - chain               : extracted causal chain
          - node_apis           : one NodeAPIEntry per edge (cache-aware)
          - constraint_prompt   : section to inject into the generation prompt
          - borrowed_edges      : edges filled from cache (not re-retrieved)
        """
        chain = self.extractor.extract(task)

        # Gather cached edges before retrieval
        cached_map: Dict[Tuple[str, str], NodeAPIEntry] = (
            self.cache.find_matching_edges(chain)
        )
        borrowed_edges = list(cached_map.keys())

        node_apis = self.retriever.get_chain_apis(chain, self.cache)

        constraint_prompt = self._build_constraint_prompt(chain, node_apis)

        return CausalPipelineResult(
            chain=chain,
            node_apis=node_apis,
            constraint_prompt=constraint_prompt,
            borrowed_edges=borrowed_edges,
        )

    # ── Step 3: Validate ─────────────────────────────────────────────────────

    def validate(
        self,
        task:   str,
        code:   str,
        result: CausalPipelineResult,
    ) -> CausalValidationResult:
        """Validate *code* against the causal chain from a previous plan() call."""
        val = self.validator.validate(code, result.chain, result.node_apis)
        result.validation = val
        return val

    # ── Step 4: Record success ────────────────────────────────────────────────

    def record_success(
        self,
        case_id: str,
        task:    str,
        result:  CausalPipelineResult,
    ) -> None:
        """Cache all edges from a successfully executed run."""
        self.cache.record_success(case_id, task, result.chain, result.node_apis)

    # ── Convenience: plan + validate in one call ──────────────────────────────

    def run(
        self,
        task:    str,
        code:    str,
        case_id: Optional[str] = None,
    ) -> CausalPipelineResult:
        """
        Full pipeline: plan → validate → (optionally) cache.

        If *code* is empty, returns the plan result with constraint_prompt only.
        If *code* is provided, validation is run and attached to result.validation.
        If validation passes and *case_id* is given, the chain is cached.
        """
        result = self.plan(task)
        if not code.strip():
            return result

        val = self.validate(task, code, result)
        if val.passed and case_id is not None:
            self.record_success(case_id, task, result)

        return result

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_constraint_prompt(
        self,
        chain: CausalChain,
        node_apis: List[NodeAPIEntry],
    ) -> str:
        """
        Build the constraint section that is injected into the code-generation
        prompt.  Structured so the LLM MUST follow the chain.
        """
        lines = [
            "=== CAUSAL ACQUISITION CHAIN (MANDATORY) ===",
            f"Task requires traversing: {' → '.join(_short(n.type_name) for n in chain.nodes)}",
            "",
            "You MUST acquire objects in this EXACT order:",
        ]

        for i, (api, node) in enumerate(zip(node_apis, chain.nodes[1:])):
            src_short = _short(api.source_type)
            tgt_short = _short(api.target_type)
            cached_tag = " [CACHED: verified in a previous successful case]" if api.source == "cache" else ""

            if api.method_name.startswith("<UNKNOWN"):
                # Unknown edge — give a hint
                lines.append(
                    f"  Step {i+1}: Get {tgt_short} from {src_short}"
                    f"  ← [WARNING: API unknown, search RAG for: "
                    f"'{src_short} get {tgt_short}']"
                )
            else:
                if api.params.strip():
                    call = f"{src_short.lower()}.{api.method_name}({api.params})"
                else:
                    call = f"{src_short.lower()}.{api.method_name}()"
                lines.append(
                    f"  Step {i+1}: {tgt_short} = {call}"
                    f"{cached_tag}"
                )
                if api.description:
                    lines.append(f"           # {api.description}")

        lines += [
            "",
            "RULES:",
            "  • Do NOT skip any step in the sequence above.",
            "  • Do NOT call a method on the wrong object type.",
            "  • Steps marked [CACHED] are verified correct — copy them exactly.",
            "  • After acquiring all objects, implement the task logic.",
            "==============================================",
        ]
        return "\n".join(lines)

    # ── Cache status ──────────────────────────────────────────────────────────

    def cache_summary(self) -> str:
        return self.cache.summary()
