"""node_retriever.py — Node-specific API retrieval.

For each edge (source_type → target_type) in a CausalChain, this module finds
the exact API method (from RAGAPIs.csv) that navigates from source to target.

Instead of one big RAG similarity pull that can hallucinate irrelevant results,
we do a structural lookup: "what method on type A returns type B?"

This prevents the classic EDA-agent bug:
  code calls a dbDatabase method on a dbBlock variable, because a single RAG
  retrieval returned both types' APIs mixed together.
"""

import csv
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .chain_extractor import CausalChain


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NodeAPIEntry:
    """The API that bridges one edge of the causal chain."""
    source_type:   str          # e.g. "openroad.Design"
    target_type:   str          # e.g. "odb.dbBlock"
    method_name:   str          # e.g. "getBlock"
    full_signature: str         # e.g. "openroad.Design.getBlock()"
    params:        str          # raw parameter string from CSV
    description:   str          # human-readable description from CSV
    is_list:       bool = False  # True when return type is list(X)
    source:        str = "rag"   # "rag" | "cache" | "supplement"

    @property
    def call_template(self) -> str:
        """Return a usage hint string like  var.getBlock()  or  var.findInst(name)."""
        if self.params.strip():
            return f"var.{self.method_name}({self.params.strip()})"
        return f"var.{self.method_name}()"

    def __repr__(self) -> str:
        return (f"NodeAPIEntry({self.source_type} → {self.target_type} "
                f"via {self.method_name})")


# ── Supplement: edges not in RAGAPIs.csv ─────────────────────────────────────
# These fill gaps where the CSV is silent.  Verified manually from OpenROAD source.

_SUPPLEMENT: List[Tuple[str, str, str, str, str, str]] = [
    # (source_type, target_type, method_name, full_sig, params, description)
    ("odb.dbBlock",   "odb.dbBTerm",        "findBTerm",
     "odb.dbBlock.findBTerm(", "str(name)", "Find a block terminal (port) by name"),
    ("odb.dbBlock",   "odb.dbNet",          "findNet",
     "odb.dbBlock.findNet(",   "str(name)", "Find a net by name"),
    ("odb.dbBlock",   "odb.dbTech",         "getTech",
     "odb.dbBlock.getTech(",   "",          "Get the odb.dbTech from the block"),
    ("openroad.Tech", "odb.dbTech",         "getDB",
     "openroad.Tech.getDB(",   "",
     "Get the odb.dbDatabase; call .getTech() on it for odb.dbTech"),
    ("odb.dbDatabase","odb.dbTech",         "getTech",
     "odb.dbDatabase.getTech(","",          "Get odb.dbTech from the database"),
    ("odb.dbTech",    "odb.dbTechLayer",    "findLayer",
     "odb.dbTech.findLayer(",  "str(name)", "Find a technology layer by name"),
    ("openroad.Design","odb.dbDatabase",    "getDb",
     "openroad.Design.getDb(", "",          "Get the odb.dbDatabase from the design"),
    ("openroad.Design","openroad.Tech",     "getTech",
     "openroad.Design.getTech(","",         "Get the openroad.Tech from the design"),
    ("openroad.Tech", "odb.dbDatabase",     "getDB",
     "openroad.Tech.getDB(",   "",          "Get the odb.dbDatabase from Tech"),
    ("odb.dbInst",    "odb.dbMaster",       "getMaster",
     "odb.dbInst.getMaster(",  "",          "Get the library cell (master) of an instance"),
    ("odb.dbInst",    "odb.dbITerm",        "getITerms",
     "odb.dbInst.getITerms(",  "",          "Get all instance terminals (ITerm list)"),
    ("odb.dbNet",     "odb.dbITerm",        "getITerms",
     "odb.dbNet.getITerms(",   "",          "Get all instance terminals on this net"),
    ("odb.dbITerm",   "odb.dbInst",         "getInst",
     "odb.dbITerm.getInst(",   "",          "Get the instance that owns this ITerm"),
    ("odb.dbITerm",   "odb.dbNet",          "getNet",
     "odb.dbITerm.getNet(",    "",          "Get the net connected to this ITerm"),
    ("odb.dbITerm",   "odb.dbMTerm",        "getMTerm",
     "odb.dbITerm.getMTerm(",  "",          "Get the master terminal of this ITerm"),
    ("odb.dbRow",     "odb.dbSite",         "getSite",
     "odb.dbRow.getSite(",     "",          "Get the site type of a row"),
    ("odb.dbInst",    "odb.Rect",           "getBBox",
     "odb.dbInst.getBBox(",    "",          "Get the bounding box of an instance"),
    ("odb.dbBlock",   "odb.Rect",           "getCoreArea",
     "odb.dbBlock.getCoreArea(","",         "Get the core area rectangle"),
]


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _normalize(type_name: str) -> str:
    """Canonical form: strip whitespace, normalise openroad.odb.X → odb.X."""
    t = type_name.strip()
    t = re.sub(r"^openroad\.odb\.", "odb.", t)
    # strip list(...) wrapper for matching purposes
    t = re.sub(r"^list\((.+)\)$", r"\1", t)
    return t


def _short(type_name: str) -> str:
    """Last component: odb.dbBlock → dbBlock."""
    return _normalize(type_name).rsplit(".", 1)[-1]


# ── NodeRetriever ─────────────────────────────────────────────────────────────

class NodeRetriever:
    """
    Builds an edge-lookup table from RAGAPIs.csv at construction time.
    Then answers: "which method goes from type A to type B?"

    Usage
    -----
    retriever = NodeRetriever("RAGData/RAGAPIs.csv")
    entries   = retriever.get_chain_apis(chain)
    # entries is a list of NodeAPIEntry in chain-edge order
    """

    def __init__(self, rag_api_path: str):
        # edge_map: (norm_source, norm_target) → List[NodeAPIEntry]
        self._edge_map: Dict[Tuple[str, str], List[NodeAPIEntry]] = {}
        self._load_csv(rag_api_path)
        self._load_supplement()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_csv(self, path: str) -> None:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn  = row.get("Function Name:", "").strip()
                    rt  = row.get("Return Type:",   "").strip()
                    par = row.get("Parameters:",    "").strip()
                    dsc = row.get("Description:",   "").strip()
                    if not fn or not rt:
                        continue
                    entry = self._parse_row(fn, rt, par, dsc, source="rag")
                    if entry:
                        self._register(entry)
        except FileNotFoundError:
            pass   # will still work via supplement

    def _parse_row(self, fn: str, rt: str, par: str, dsc: str,
                   source: str) -> Optional[NodeAPIEntry]:
        """Parse 'openroad.Design.getBlock(' into a NodeAPIEntry."""
        # Strip everything from the first '(' onwards, then strip whitespace
        fn_clean = fn.split("(")[0].strip()
        if "." not in fn_clean:
            return None
        parts = fn_clean.rsplit(".", 1)
        receiver = parts[0]
        method   = parts[1].strip()
        is_list  = rt.startswith("list(")
        target   = _normalize(rt)
        if not target:
            return None
        return NodeAPIEntry(
            source_type=_normalize(receiver),
            target_type=target,
            method_name=method,
            full_signature=fn,
            params=par,
            description=dsc,
            is_list=is_list,
            source=source,
        )

    def _load_supplement(self) -> None:
        for src, tgt, meth, sig, par, dsc in _SUPPLEMENT:
            entry = NodeAPIEntry(
                source_type=_normalize(src),
                target_type=_normalize(tgt),
                method_name=meth,
                full_signature=sig,
                params=par,
                description=dsc,
                is_list=False,
                source="supplement",
            )
            self._register(entry)

    def _register(self, entry: NodeAPIEntry) -> None:
        key = (_normalize(entry.source_type), _normalize(entry.target_type))
        self._edge_map.setdefault(key, []).append(entry)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_chain_apis(
        self,
        chain: CausalChain,
        cache: Optional["ChainCache"] = None,  # type: ignore[name-defined]
    ) -> List[NodeAPIEntry]:
        """
        Return one NodeAPIEntry per edge in *chain*.

        If *cache* is provided, borrowed entries (from previous successful
        chains) replace RAG entries for matching edges.
        """
        results: List[NodeAPIEntry] = []
        # Gather cache hints for this chain (if available)
        cached: Dict[Tuple[str, str], NodeAPIEntry] = {}
        if cache is not None:
            cached = cache.find_matching_edges(chain)

        for src, tgt in chain.edges:
            ns, nt = _normalize(src), _normalize(tgt)
            # Cache takes priority (previously verified)
            if (ns, nt) in cached:
                entry = cached[(ns, nt)]
                results.append(entry)
                continue
            entry = self._lookup(ns, nt)
            results.append(entry)

        return results

    def _lookup(self, norm_src: str, norm_tgt: str) -> NodeAPIEntry:
        """Find the best entry for (src → tgt); return a placeholder if missing.

        Priority:
          1. RAG CSV entries (authoritative; prefer non-list first, then list)
          2. Supplement entries (non-list then list)
        """
        candidates = self._edge_map.get((norm_src, norm_tgt), [])

        # RAG CSV entries are authoritative; use their insertion (CSV row) order.
        # Supplement entries only fill gaps (no RAG entry exists).
        rag  = [c for c in candidates if c.source == "rag"]
        supp = [c for c in candidates if c.source != "rag"]
        chosen = rag[0] if rag else (supp[0] if supp else None)

        if chosen:
            return chosen
        # Unknown edge — create a placeholder so the pipeline can still proceed
        short_src = _short(norm_src)
        short_tgt = _short(norm_tgt)
        return NodeAPIEntry(
            source_type=norm_src,
            target_type=norm_tgt,
            method_name=f"<UNKNOWN: {short_src}→{short_tgt}>",
            full_signature=f"# TODO: find method on {norm_src} that returns {norm_tgt}",
            params="",
            description=f"[NOT FOUND] No API found for {norm_src} → {norm_tgt}",
            source="rag",
        )

    # ── Inspection helpers ────────────────────────────────────────────────────

    def known_edges(self) -> Set[Tuple[str, str]]:
        return set(self._edge_map.keys())

    def apis_for_type(self, type_name: str) -> List[NodeAPIEntry]:
        """All APIs where source_type == type_name (useful for prompt building)."""
        norm = _normalize(type_name)
        out = []
        for (src, _), entries in self._edge_map.items():
            if src == norm:
                out.extend(entries)
        return out
