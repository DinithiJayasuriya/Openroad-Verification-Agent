"""causal_verifier — Causal Chain Extraction, Retrieval, Validation, and Caching.

Pipeline stages
---------------
1. CausalChainExtractor  (chain_extractor.py)
   Task text → mandatory object-acquisition chain
   e.g. "Find instance 486" → [openroad.Design → odb.dbBlock → odb.dbInst]

2. NodeRetriever         (node_retriever.py)
   Chain → per-edge API signatures from RAGAPIs.csv
   e.g. (odb.dbBlock → odb.dbInst) → "block.findInst(name)"

3. CausalChainValidator  (chain_validator.py)
   code + chain → verifies variable-binding sequence matches the chain

4. ChainCache            (chain_cache.py)
   Stores successful (chain, edge→API) pairs; lends sub-chains to new tasks

5. CausalPipeline        (pipeline.py)
   Orchestrates 1-4; returns constraint prompt + validation result
"""
from .chain_extractor import CausalChainExtractor, CausalChain, ChainNode
from .node_retriever import NodeRetriever, NodeAPIEntry
from .chain_validator import CausalChainValidator, CausalValidationResult
from .chain_cache import ChainCache, CachedChainEntry
from .pipeline import CausalPipeline, CausalPipelineResult

__all__ = [
    "CausalChainExtractor", "CausalChain", "ChainNode",
    "NodeRetriever", "NodeAPIEntry",
    "CausalChainValidator", "CausalValidationResult",
    "ChainCache", "CachedChainEntry",
    "CausalPipeline", "CausalPipelineResult",
]
