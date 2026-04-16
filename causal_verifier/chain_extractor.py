"""chain_extractor.py — Compatibility stub for the causal_verifier package.

The original chain_extractor was only available as a .pyc compiled for a
different Python version.  This stub provides the three exports that the
package __init__.py and node_retriever.py require:

    CausalChain          — ordered type-acquisition path + edges property
    ChainNode            — single node in the chain (type name + description)
    CausalChainExtractor — LLM-based extractor (minimal stub for import compat)

The flow pipeline only uses NodeRetriever._lookup() directly, so the full
extractor logic is not needed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ChainNode:
    """One node in a CausalChain (type name + optional description)."""
    type_name:   str
    description: str = ""

    def __repr__(self) -> str:
        return self.type_name.rsplit(".", 1)[-1]


@dataclass
class CausalChain:
    """Ordered type-acquisition path.

    Attributes
    ----------
    types : Ordered list of fully-qualified type-name strings.
    nodes : Optional ChainNode list (parallel to types).
    task  : Original task description.

    Properties
    ----------
    edges : List of consecutive (src, tgt) type-name pairs.
    """
    types: List[str]        = field(default_factory=list)
    nodes: List[ChainNode]  = field(default_factory=list)
    task:  str              = ""

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return [(self.types[i], self.types[i + 1])
                for i in range(len(self.types) - 1)]

    def __repr__(self) -> str:
        return " → ".join(t.rsplit(".", 1)[-1] for t in self.types)


class CausalChainExtractor:
    """Minimal stub — the flow pipeline does not use LLM chain extraction."""

    def __init__(self, *args, **kwargs):
        pass

    def extract(self, task: str) -> Optional[CausalChain]:
        """Return None; flow pipeline uses FlowMultiChainExtractor instead."""
        return None
