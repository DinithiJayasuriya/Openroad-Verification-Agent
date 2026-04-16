"""chain_cache.py — Persistent cache of successful causal chains.

When a task is solved correctly, we record which edge → API mapping was used.
Future tasks whose causal chain shares sub-edges can borrow those verified
API bindings instead of re-running RAG retrieval for those edges.

Example
-------
Case 3 solved with chain:  openroad.Design → odb.dbBlock → odb.dbInst
  edge 0:  openroad.Design → odb.dbBlock  via design.getBlock()
  edge 1:  odb.dbBlock → odb.dbInst       via block.findInst(name)

Case 7 needs chain: openroad.Design → odb.dbBlock → odb.dbNet
  edge 0:  openroad.Design → odb.dbBlock  is cached from case 3 ← BORROWED
  edge 1:  odb.dbBlock → odb.dbNet        retrieved fresh from RAG

On-disk format: a JSON file mapping edge-key strings to NodeAPIEntry dicts.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .chain_extractor import CausalChain
from .node_retriever   import NodeAPIEntry, _normalize


# ── Cached entry ──────────────────────────────────────────────────────────────

@dataclass
class CachedChainEntry:
    """Record of one successfully used chain."""
    case_id:     str                      # task identifier / case number
    task:        str                      # original task text
    chain_types: List[str]               # type_names in order
    edge_apis:   Dict[str, dict]          # json-key "src|tgt" → NodeAPIEntry dict
    success:     bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CachedChainEntry":
        return CachedChainEntry(**d)


# ── ChainCache ────────────────────────────────────────────────────────────────

class ChainCache:
    """
    In-memory (optionally file-backed) cache of verified causal chain edges.

    Usage
    -----
    cache = ChainCache(cache_path="cache/chain_cache.json")

    # Record after a successful run
    cache.record_success("case_3", task, chain, node_apis)

    # Retrieve cached edges for a new chain
    borrowed = cache.find_matching_edges(new_chain)
    # borrowed: Dict[(norm_src, norm_tgt), NodeAPIEntry]
    """

    def __init__(self, cache_path: Optional[str] = None):
        self._path: Optional[str] = cache_path
        # edge_store: (norm_src, norm_tgt) → List[NodeAPIEntry]
        # multiple entries allowed (from different successful cases)
        self._edge_store: Dict[Tuple[str, str], List[NodeAPIEntry]] = {}
        self._entries: List[CachedChainEntry] = []

        if cache_path and os.path.exists(cache_path):
            self._load(cache_path)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_success(
        self,
        case_id: str,
        task: str,
        chain: CausalChain,
        node_apis: List[NodeAPIEntry],
    ) -> None:
        """Store all edges of a successfully executed chain."""
        edge_apis: Dict[str, dict] = {}
        for entry in node_apis:
            key_str  = f"{entry.source_type}|{entry.target_type}"
            edge_apis[key_str] = asdict(entry)
            norm_key = (_normalize(entry.source_type), _normalize(entry.target_type))
            # Prepend: most recent successful case takes priority
            existing = self._edge_store.get(norm_key, [])
            # Avoid exact-duplicate entries
            if not any(e.full_signature == entry.full_signature for e in existing):
                # Mark this entry as coming from cache for future users
                cached_entry = NodeAPIEntry(
                    source_type=entry.source_type,
                    target_type=entry.target_type,
                    method_name=entry.method_name,
                    full_signature=entry.full_signature,
                    params=entry.params,
                    description=f"[CACHED from {case_id}] {entry.description}",
                    is_list=entry.is_list,
                    source="cache",
                )
                self._edge_store[norm_key] = [cached_entry] + existing

        record = CachedChainEntry(
            case_id=case_id, task=task,
            chain_types=chain.type_names,
            edge_apis=edge_apis,
        )
        self._entries.append(record)

        if self._path:
            self._save()

    # ── Borrowing ─────────────────────────────────────────────────────────────

    def find_matching_edges(
        self, chain: CausalChain
    ) -> Dict[Tuple[str, str], NodeAPIEntry]:
        """
        Return a dict of cached NodeAPIEntry objects for edges present in *chain*.

        Only edges that appear in the cache are returned; missing edges must be
        retrieved fresh from RAG.  The returned dict uses normalised type-name
        tuples as keys so NodeRetriever.get_chain_apis() can merge easily.
        """
        result: Dict[Tuple[str, str], NodeAPIEntry] = {}
        for src, tgt in chain.edges:
            ns, nt = _normalize(src), _normalize(tgt)
            entries = self._edge_store.get((ns, nt), [])
            if entries:
                result[(ns, nt)] = entries[0]   # most recent first
        return result

    def has_edge(self, src: str, tgt: str) -> bool:
        return (_normalize(src), _normalize(tgt)) in self._edge_store

    def edge_count(self) -> int:
        return len(self._edge_store)

    def summary(self) -> str:
        lines = [f"ChainCache: {len(self._entries)} recorded cases, "
                 f"{self.edge_count()} unique edges cached"]
        for (s, t), entries in sorted(self._edge_store.items()):
            lines.append(f"  {s} → {t}  [{entries[0].method_name}]"
                         f"  (from {entries[0].description[:40]})")
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        assert self._path
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        data = [e.to_dict() for e in self._entries]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                entry = CachedChainEntry.from_dict(d)
                self._entries.append(entry)
                for key_str, api_dict in entry.edge_apis.items():
                    src_str, tgt_str = key_str.split("|", 1)
                    api = NodeAPIEntry(**api_dict)
                    api.source = "cache"
                    norm_key = (_normalize(src_str), _normalize(tgt_str))
                    self._edge_store.setdefault(norm_key, []).append(api)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass   # corrupt cache — start fresh
