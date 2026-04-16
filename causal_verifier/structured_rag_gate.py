"""structured_rag_gate.py — Hard Gate: validates extracted causal chain types
against RAGAPIs_structured.csv and OpenROAD source.

Logic for each type node T in a causal chain:
  1. T in RAGAPIs_structured.csv (Receiver Type or Return Type col) → VALID
  2. T not in RAG, but IS in OpenROAD source (db.h classes + module types)
       → RAG_MISS: real type, just missing from RAG.
         Appends a placeholder row to the structured CSV so future lookups
         can find it (and a human can fill in real methods).
  3. T not in RAG AND not in source → HALLUCINATION.
       Provides corrective feedback so the chain extractor can rewrite.

Leaf-action nodes (bracket notation e.g. [isOutputSignal]) are NOT types —
they are skipped here and validated later in the RAG retrieval stage.
"""

from __future__ import annotations

import csv
import difflib
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Known types from OpenROAD C++ source (db.h class declarations + module hdrs).
# These are the types that are valid in the OpenROAD Python bindings.
# ─────────────────────────────────────────────────────────────────────────────

# odb types — all class db* from OpenROAD/src/odb/include/odb/db.h
_ODB_SHORT_NAMES: Set[str] = {
    "dbBlock", "dbBPin", "dbBTerm", "dbNet", "dbInst", "dbITerm",
    "dbMaster", "dbMTerm", "dbDatabase", "dbTech", "dbTechLayer", "dbSite",
    "dbRow", "dbChip", "dbLib", "dbVia", "dbViaParams", "dbWire", "dbBox",
    "dbTrackGrid", "dbGCellGrid", "dbTransform", "dbObstruction", "dbFill",
    "dbBlockage", "dbMPin", "dbTechVia", "dbTechNonDefaultRule",
    "dbTechLayerRule", "dbGroup", "dbModule", "dbModInst", "dbModNet",
    "dbGuide", "dbRegion", "dbBusPort", "dbGlobalConnect", "dbNetTrack",
    "dbPolygon", "dbAccessPoint",
    # Geometry helpers (exposed as odb.Rect / odb.Point)
    "Rect", "Point",
}

# Fully-qualified odb forms (odb.dbNet etc.)
_ODB_QUALIFIED: Set[str] = {f"odb.{t}" for t in _ODB_SHORT_NAMES}

# Module-level types (flow tools + top-level objects)
_MODULE_TYPES: Set[str] = {
    "openroad.Design", "openroad.Tech", "openroad.Timing",
    "gpl.Replace",
    "grt.GlobalRouter",
    "ppl.IOPlacer",  "ppl.Parameters",
    "cts.TritonCTS", "cts.CtsOptions", "cts.TechChar",
    "drt.TritonRoute", "drt.ParamStruct",
    "mpl.MacroPlacer",
    "dpl.Opendp",
    "pdn.PdnGen",    "pdn.VoltageDomain",
    "ifp.InitFloorplan",
    "psm.PDNSim",
    # Short / unqualified forms the LLM sometimes uses
    "Replace", "GlobalRouter", "IOPlacer", "TritonCTS", "TritonRoute",
    "MacroPlacer", "Opendp", "PdnGen", "InitFloorplan",
    # Top-level module namespaces that appear as standalone identifiers
    "openroad", "odb",
}

_ALL_SOURCE_TYPES: Set[str] = _ODB_SHORT_NAMES | _ODB_QUALIFIED | _MODULE_TYPES

_LEAF_RE = re.compile(r'^\[.+\]$')   # e.g. [isOutputSignal]

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class TypeStatus(Enum):
    VALID         = "VALID"          # present in structured RAG
    RAG_MISS      = "RAG_MISS"       # real OpenROAD type, but absent from RAG
    HALLUCINATION = "HALLUCINATION"  # not in RAG AND not in OpenROAD source
    LEAF_ACTION   = "LEAF_ACTION"    # bracket notation — not a type, skipped


@dataclass
class TypeReport:
    type_name:       str
    normalized:      str            # short/clean form used for lookup
    status:          TypeStatus
    closest_real:    Optional[str] = None   # nearest real type (hallucinations)
    rag_entry_added: bool          = False  # True when we appended to the CSV


@dataclass
class GateReport:
    """Full gate report for one causal extraction result."""
    path_reports:       List[List[TypeReport]]   # [path_idx][node_idx]
    had_hallucinations: bool = False
    had_rag_misses:     bool = False
    rewrite_feedback:   str  = ""   # non-empty → pass back to chain extractor

    def summary(self) -> str:
        """One-line human-readable summary."""
        counts: Dict[str, int] = {s.value: 0 for s in TypeStatus}
        for path in self.path_reports:
            for tr in path:
                counts[tr.status.value] += 1
        parts = [f"{v}×{k}" for k, v in counts.items() if v > 0]
        return " | ".join(parts) if parts else "no types"

    def hallucinated_types(self) -> List[Tuple[str, Optional[str]]]:
        """List of (type_name, closest_real) for every hallucinated node."""
        out = []
        seen = set()
        for path in self.path_reports:
            for tr in path:
                if tr.status == TypeStatus.HALLUCINATION and tr.type_name not in seen:
                    out.append((tr.type_name, tr.closest_real))
                    seen.add(tr.type_name)
        return out

    def rag_miss_types(self) -> List[str]:
        """List of unique type names flagged as RAG_MISS."""
        out = []
        seen = set()
        for path in self.path_reports:
            for tr in path:
                if tr.status == TypeStatus.RAG_MISS and tr.type_name not in seen:
                    out.append(tr.type_name)
                    seen.add(tr.type_name)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main gate class
# ─────────────────────────────────────────────────────────────────────────────

class StructuredRAGGate:
    """Hard Gate: validates extracted type nodes against structured RAG + source."""

    def __init__(self, structured_csv_path: str):
        self.csv_path         = structured_csv_path
        self._all_rag_types:  Set[str] = set()
        self._load_csv()

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _load_csv(self) -> None:
        """Build a flat set of all known type tokens from the structured CSV."""
        if not os.path.isfile(self.csv_path):
            print(f"  [gate] WARNING: structured CSV not found: {self.csv_path}",
                  flush=True)
            return
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
        except Exception as exc:
            print(f"  [gate] WARNING: could not read CSV: {exc}", flush=True)
            return

        for col in ("Receiver Type", "Return Type"):
            if col not in df.columns:
                continue
            for raw_val in df[col].dropna():
                self._register_type(str(raw_val).strip())

        print(
            f"  [gate] Loaded {len(self._all_rag_types)} unique type tokens "
            f"from {os.path.basename(self.csv_path)}",
            flush=True,
        )

    def _register_type(self, raw: str) -> None:
        """Add raw, its list-unwrapped form, and its short name to the known set."""
        if not raw or raw.lower() == "nan":
            return
        self._all_rag_types.add(raw)
        # unwrap list(T) → T
        inner = re.sub(r'^list\((.+)\)$', r'\1', raw).strip()
        if inner != raw:
            self._all_rag_types.add(inner)
        # short name (last component after last dot)
        short = raw.rsplit(".", 1)[-1]
        self._all_rag_types.add(short)
        if inner != raw:
            self._all_rag_types.add(inner.rsplit(".", 1)[-1])

    # ── Lookup helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(type_name: str) -> str:
        """Return a canonical form: strip list(...) wrapper, keep qualification."""
        t = type_name.strip()
        t = re.sub(r'^list\((.+)\)$', r'\1', t).strip()
        return t

    def _in_rag(self, type_name: str) -> bool:
        norm  = self._normalize(type_name)
        short = norm.rsplit(".", 1)[-1]
        return (type_name in self._all_rag_types or
                norm      in self._all_rag_types or
                short     in self._all_rag_types)

    @staticmethod
    def _in_source(type_name: str) -> bool:
        norm  = StructuredRAGGate._normalize(type_name)
        short = norm.rsplit(".", 1)[-1]
        return (type_name in _ALL_SOURCE_TYPES or
                norm      in _ALL_SOURCE_TYPES or
                short     in _ALL_SOURCE_TYPES)

    @staticmethod
    def _closest_real_type(type_name: str) -> Optional[str]:
        """Find nearest known type by edit distance / prefix heuristic.

        Searches only short names (no dots) to avoid spurious qualified-form
        hits (e.g. 'odb.Point' scoring high because 'd','b','P','i','n' appear
        in order). Returns the best short-name match, qualified with 'odb.'
        prefix when appropriate.
        """
        norm  = StructuredRAGGate._normalize(type_name)
        short = norm.rsplit(".", 1)[-1]

        # Search only over short names to keep edit-distance meaningful
        short_candidates = sorted(
            {c for c in _ALL_SOURCE_TYPES if "." not in c}
        )
        matches = difflib.get_close_matches(short, short_candidates, n=3, cutoff=0.55)
        if matches:
            best = matches[0]   # highest-scoring short name
            # Qualify odb db* types; leave module types unqualified
            if best.startswith("db"):
                return f"odb.{best}"
            return best

        # Prefix heuristic — useful for typos like dbPin → dbBPin/dbMPin
        prefix = short[:max(4, len(short) - 2)].lower()
        prefix_matches = [c for c in short_candidates
                          if c.lower().startswith(prefix)]
        if prefix_matches:
            best = prefix_matches[0]
            return f"odb.{best}" if best.startswith("db") else best
        return None

    # ── CSV write-back (RAG miss) ─────────────────────────────────────────────

    def _add_rag_miss_entry(self, type_name: str) -> None:
        """Append a placeholder row for a RAG-miss type to the structured CSV."""
        norm  = self._normalize(type_name)
        short = norm.rsplit(".", 1)[-1]
        new_row = {
            "Description": (
                f"[RAG_MISS AUTO-ADDED] '{norm}' is a real OpenROAD type found "
                f"in the C++ source but was absent from RAGAPIs_structured.csv. "
                f"Please fill in the correct method signatures."
            ),
            "Receiver Type": norm,
            "Method Name":   "UNKNOWN",
            "Parameters":    "",
            "Return Type":   "",
        }
        try:
            file_exists = os.path.isfile(self.csv_path)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Description", "Receiver Type",
                                "Method Name", "Parameters", "Return Type"],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(new_row)
            # Update in-memory set so we don't flag it again
            self._register_type(norm)
            print(
                f"  [gate] RAG_MISS: appended placeholder for '{norm}' "
                f"to {os.path.basename(self.csv_path)}",
                flush=True,
            )
        except Exception as exc:
            print(f"  [gate] WARNING: could not write to CSV: {exc}", flush=True)

    # ── Core validation ───────────────────────────────────────────────────────

    def validate(self, paths: List[List[str]], task: str = "") -> GateReport:
        """Validate all type nodes across extracted paths.

        Parameters
        ----------
        paths : list of paths, each path = list of type-name strings.
        task  : original task text (used only for logging).

        Returns
        -------
        GateReport with per-path TypeReports and an overall rewrite_feedback
        string (non-empty when hallucinations were found).
        """
        report         = GateReport(path_reports=[])
        hallucinations: List[Tuple[str, Optional[str]]] = []
        seen_rag_miss:  Set[str] = set()

        for path in paths:
            path_report: List[TypeReport] = []

            for type_name in path:
                # ── Leaf-action node: skip type validation ─────────────────
                if _LEAF_RE.match(type_name):
                    path_report.append(TypeReport(
                        type_name=type_name,
                        normalized=type_name,
                        status=TypeStatus.LEAF_ACTION,
                    ))
                    continue

                norm = self._normalize(type_name)

                # ── Check 1: in structured RAG ─────────────────────────────
                if self._in_rag(type_name):
                    path_report.append(TypeReport(
                        type_name=type_name,
                        normalized=norm,
                        status=TypeStatus.VALID,
                    ))
                    continue

                # ── Check 2: in OpenROAD source but not in RAG ─────────────
                if self._in_source(type_name):
                    tr = TypeReport(
                        type_name=type_name,
                        normalized=norm,
                        status=TypeStatus.RAG_MISS,
                    )
                    if norm not in seen_rag_miss:
                        self._add_rag_miss_entry(type_name)
                        tr.rag_entry_added = True
                        seen_rag_miss.add(norm)
                    report.had_rag_misses = True
                    path_report.append(tr)
                    continue

                # ── Check 3: not in source → hallucination ─────────────────
                closest = self._closest_real_type(type_name)
                tr = TypeReport(
                    type_name=type_name,
                    normalized=norm,
                    status=TypeStatus.HALLUCINATION,
                    closest_real=closest,
                )
                report.had_hallucinations = True
                # Only add (type, closest) once per unique type_name
                if not any(h[0] == type_name for h in hallucinations):
                    hallucinations.append((type_name, closest))
                path_report.append(tr)

            report.path_reports.append(path_report)

        # ── Build corrective feedback for hallucinated types ──────────────
        if hallucinations:
            lines = [
                "The following types in the causal chain do NOT exist in "
                "OpenROAD Python API:"
            ]
            for ht, cr in hallucinations:
                if cr:
                    lines.append(
                        f"  - '{ht}' is INVALID. "
                        f"Closest real type: '{cr}'. "
                        f"Rewrite the path to use '{cr}' instead."
                    )
                else:
                    lines.append(
                        f"  - '{ht}' is INVALID and has no close match. "
                        f"Use only known OpenROAD types such as: "
                        f"odb.dbBlock, odb.dbNet, odb.dbInst, odb.dbBTerm, "
                        f"odb.dbITerm, odb.dbMaster, odb.dbTechLayer, "
                        f"gpl.Replace, grt.GlobalRouter, ppl.IOPlacer, "
                        f"cts.TritonCTS, drt.TritonRoute, mpl.MacroPlacer."
                    )
            report.rewrite_feedback = "\n".join(lines)

        return report

    # ── Convenience: type-restricted RAG lookup hint ──────────────────────────

    def get_valid_methods_for_type(
        self, receiver_type: str, top_k: int = 5
    ) -> List[Dict]:
        """Return up to top_k rows from the structured CSV whose Receiver Type
        matches receiver_type (exact or short-name match).

        Used downstream for type-restricted RAG retrieval.
        """
        if not os.path.isfile(self.csv_path):
            return []
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
        except Exception:
            return []

        norm  = self._normalize(receiver_type)
        short = norm.rsplit(".", 1)[-1]

        mask = df["Receiver Type"].astype(str).apply(
            lambda v: (
                v.strip() == norm or
                v.strip() == receiver_type or
                v.strip().rsplit(".", 1)[-1] == short
            )
        )
        rows = df[mask].head(top_k)
        return rows.to_dict(orient="records")
