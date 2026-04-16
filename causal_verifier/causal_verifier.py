"""causal_verifier.py — Chain-aware code verifier (redesigned).

Three layers:

  L1 — Syntax
       ast.parse() + non-empty check.
       Delegates to static verifier for this; kept here for completeness.

  L2 — Causal flow
       For each edge (src → tgt) in the chain, check that the acquisition
       method appears in the code and is called before the target variable
       is used. No type inference — just method-name presence and ordering.

  L3 — API diff analysis  (the key layer)
       For each chain edge:
         code_map[edge] = method the LLM actually used in the generated code
         rag_map[edge]  = method RAG retrieved (from edge_apis)
       Any mismatch = hallucinated API.
       Produces per-edge diff entries used by the controller to decide
       re_retrieve_edge for the specific mismatched edge(s).

       Re-retrieval hint: includes the hallucinated method name + lower
       similarity threshold + code examples from RAGCodePiece.csv.

Result carries:
  passed        bool
  layer_failed  int  (0=pass, 1=syntax, 2=flow, 3=api_diff)
  issues        List[str]   — human-readable per-issue strings
  feedback      str         — full repair context for the controller
  api_diffs     List[APIEdgeDiff]  — structured diff per mismatched edge
"""

import ast
import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# odb module-level attribute validation
# ─────────────────────────────────────────────────────────────────────────────

_ODB_MODULE_ACCESS = re.compile(r'\bodb\.([A-Za-z][A-Za-z0-9_]*)')

_VALID_ODB_CLASSES: Set[str] = {
    # Constructable / directly usable types (uppercase-first)
    "Rect", "Point",
    # Constructable / directly usable types (db-prefixed, lowercase-first)
    "dbDatabase", "dbBlock", "dbChip", "dbInst", "dbNet", "dbBTerm",
    "dbITerm", "dbMaster", "dbMTerm", "dbTech", "dbTechLayer",
    "dbRow", "dbSite", "dbWire", "dbSWire", "dbBox", "dbRegion",
    "dbModule", "dbModInst", "dbGroup", "dbAccessPoint",
    "dbGDSLib", "dbVia", "dbTechVia", "dbTechNonDefaultRule",
}

# Maps hallucinated odb.* enum names → what to do instead.
# Do NOT suggest another odb.* name — none of them are accessible as module constants.
_ODB_ENUM_HINTS: Dict[str, str] = {
    "SignalType":   "Do not use odb.SignalType. Call net.getSigType() — it returns a string like 'SIGNAL', 'POWER', 'GROUND'. Compare with: net.getSigType() == 'SIGNAL'",
    "dbSigType":    "Do not use odb.dbSigType. Call net.getSigType() — it returns a string like 'SIGNAL', 'POWER', 'GROUND'. Compare with: net.getSigType() == 'SIGNAL'",
    "Orient":       "Do not use odb.Orient. Call inst.getOrient() — it returns a string like 'R0', 'R90', 'MY', 'MX'. Pass the string directly to inst.setOrient('MY').",
    "Orientation":  "Do not use odb.Orientation. Call inst.getOrient() — it returns a string like 'R0', 'R90', 'MY', 'MX'. Pass the string directly to inst.setOrient('MY').",
    "dbOrientType": "Do not use odb.dbOrientType. Call inst.getOrient() — it returns a string like 'R0', 'R90', 'MY', 'MX'. Pass the string directly to inst.setOrient('MY').",
}

from causal_state import VerifierSnapshot


# ─────────────────────────────────────────────────────────────────────────────
# Variable name table  (src type → expected variable name in generated code)
# ─────────────────────────────────────────────────────────────────────────────

_VAR_NAMES: Dict[str, str] = {
    "openroad.Design":   "design",
    "odb.dbDatabase":    "db",
    "odb.dbBlock":       "block",
    "odb.dbInst":        "inst",
    "odb.dbNet":         "net",
    "odb.dbBTerm":       "bterm",
    "odb.dbITerm":       "iterm",
    "odb.dbMaster":      "master",
    "odb.dbMTerm":       "mterm",
    "odb.dbTechLayer":   "layer",
    "odb.dbTech":        "tech",
    "odb.dbRow":         "row",
    "odb.dbSite":        "site",
    "odb.Rect":          "bbox",
    "odb.Point":         "pt",
    "openroad.Tech":     "tech",
    "openroad.Timing":   "timing",
    "gpl.Replace":       "placer",
    "ifp.InitFloorplan": "floorplan",
    "cts.TritonCTS":     "cts",
    "grt.GlobalRouter":  "router",
    "ppl.IOPlacer":      "io_placer",
    "drt.TritonRoute":   "detailed_router",
}

def _var(type_name: str) -> str:
    return _VAR_NAMES.get(type_name, type_name.split(".")[-1].lower())


# ─────────────────────────────────────────────────────────────────────────────
# Structured diff result per edge
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APIEdgeDiff:
    """Mismatch between what the LLM used and what RAG retrieved for one edge."""
    src_type:       str
    tgt_type:       str
    rag_method:     str          # what RAG said to use
    code_methods:   List[str]    # what LLM actually called on src_var
    is_missing:     bool         # True if rag_method absent entirely from code
    is_hallucinated: bool        # True if code used method(s) not matching rag


# ─────────────────────────────────────────────────────────────────────────────
# Code scanner — extract method calls per variable name
# ─────────────────────────────────────────────────────────────────────────────

def _extract_method_name(api: Optional[dict]) -> Optional[str]:
    if api is None:
        return None
    fn = api.get("function_name", "")
    if fn.endswith("("):
        # Chained pattern e.g. "A.getTech().getDB().findLayer(" —
        # strip the trailing bare ( then take the last dotted segment.
        return fn[:-1].split(".")[-1].strip() or None
    return fn.split("(")[0].split(".")[-1].strip() or None




def _scan_methods_on_var(code: str, var_name: str) -> List[str]:
    """
    Return all method names called on `var_name` anywhere in the code.

    Handles:
      - Direct calls:        var.method(...)
      - In list comprehensions: [x for x in var.method()]
      - In conditions:       if var.method():
      - In for loops:        for x in var.method():
      - Chained:             var.method1().method2()  → [method1]

    Uses a simple regex scan — avoids all AST type-inference complexity.
    """
    # Match: var_name.methodName( — capture methodName
    pattern = re.compile(
        r'\b' + re.escape(var_name) + r'\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\('
    )
    return list(dict.fromkeys(m.group(1) for m in pattern.finditer(code)))


def _scan_all_method_calls(code: str) -> Set[str]:
    """Return every method name used anywhere in the code."""
    pattern = re.compile(r'\.([A-Za-z_][A-Za-z0-9_]*)\s*\(')
    return {m.group(1) for m in pattern.finditer(code)}


# ─────────────────────────────────────────────────────────────────────────────
# CausalVerifier
# ─────────────────────────────────────────────────────────────────────────────

class CausalVerifier:
    """
    Chain-aware verifier based on API diff analysis.

    Parameters
    ----------
    metadata : list of RAG row dicts (optional).
        When provided, builds a known-methods table per type so that
        the terminal-node check (L3b) can catch hallucinated methods
        called on acquired objects (e.g. net.isSelected vs net.isSelect).
    """

    def __init__(self, metadata: Optional[list] = None):
        # Build type → set-of-known-methods from RAG CSV rows.
        # Function names are like "odb.dbNet.isSelect()" or "openroad.Design.getBlock()"
        # so the class name is always the second-to-last dotted segment.
        self._known: Dict[str, Set[str]] = {}
        if metadata:
            for row in metadata:
                fn    = str(row.get("Function Name:", "")).strip()
                parts = fn.split("(")[0].split(".")
                if len(parts) >= 2:
                    cls    = parts[-2].strip()   # e.g. "dbNet"
                    method = parts[-1].strip()   # e.g. "isSelect"
                    if cls and method:
                        self._known.setdefault(cls, set()).add(method)

        # Ground-truth supplements for methods that exist in the OpenROAD API
        # but are missing or misattributed in the RAG CSV.
        # These allow uses_known=True so L3 doesn't flag them as hallucinations
        # when RAG retrieved a module-level shortcut instead of the instance method.
        _KNOWN_SUPPLEMENTS: Dict[str, Set[str]] = {
            "dbDatabase": {"getTech", "getChip", "getLibs", "findMaster"},
            "dbTech":     {"findLayer", "getLayers", "getVias"},
            "dbBlock":    {"findNet", "findInst", "findBTerm", "getInsts",
                           "getNets", "getBTerms", "getITerms"},
            # dbNet: RAG CSV is missing getBTerms and getWire which are valid
            # OpenROAD methods. Without these, a correct net.getBTerms() call
            # would be falsely flagged by L3b when dbNet appears as a Path 2
            # terminal in multi-path chain verification.
            "dbNet":      {"getBTerms", "getWire"},
        }
        for cls, methods in _KNOWN_SUPPLEMENTS.items():
            self._known.setdefault(cls, set()).update(methods)

    def verify(self, code: str, chain: list,
               edge_apis: list,
               all_edges: list = None,
               paths: list = None) -> VerifierSnapshot:
        """
        Run L1 → L2 → L3 and return a VerifierSnapshot.

        The snapshot carries `.api_diffs` (set as an extra attribute) so the
        controller/dispatcher can act on specific mismatched edges directly.

        paths: all extracted causal paths (List[List[str]]). When provided,
               L3b checks the terminal node of EVERY path, not just chain[-1]
               (chain = paths[0]). This catches hallucinated methods on Path 2+
               terminals (e.g. net.getPins() when dbNet is Path 2's terminal).
        """
        # ── L1: syntax ────────────────────────────────────────────────────────
        if not code.strip():
            return self._fail(1, ["Code is empty."], "Code is empty.", [])
        try:
            ast.parse(code)
        except SyntaxError as e:
            return self._fail(1,
                [f"SyntaxError at line {e.lineno}: {e.msg}"],
                f"Syntax error at line {e.lineno}: {e.msg}",
                [],
            )

        if len(chain) < 2:
            # Nothing to verify chain-wise — pass through
            snap = self._pass([], [])
            return snap

        # Build type environment once; shared by L2 and L3.
        type_env = self._build_type_env(code, chain, edge_apis)

        # ── L2: causal flow ───────────────────────────────────────────────────
        l2_issues = self._check_flow(code, chain, edge_apis, type_env)
        if l2_issues:
            return self._fail(2, l2_issues,
                self._flow_feedback(l2_issues, chain, edge_apis), [])

        # ── L2b: odb module attribute check ───────────────────────────────────
        odb_issues = self._check_odb_module_accesses(code)
        if odb_issues:
            feedback = (
                "odb module attribute check FAILED — hallucinated module-level names:\n"
                + "\n".join(f"  {i}" for i in odb_issues)
            )
            return self._fail(2, odb_issues, feedback, [])

        # ── L2c: null-safety check ────────────────────────────────────────────
        # Any variable assigned from a find*() call that returns a single
        # odb.db* object must be None-checked before its attributes are accessed.
        # Skipping this causes Signal 11 / tool crashes when the object is absent.
        null_issues = self._check_null_safety(code, type_env)
        if null_issues:
            feedback = (
                "Null-safety check FAILED — find*() result used without None guard:\n"
                + "\n".join(f"  {i}" for i in null_issues)
            )
            return self._fail(2, null_issues, feedback, [])

        # ── L3: API diff ──────────────────────────────────────────────────────
        diffs = self._build_api_diffs(code, chain, edge_apis, type_env)
        hard_diffs = [d for d in diffs if d.is_hallucinated or d.is_missing]

        if hard_diffs:
            issues   = [self._diff_issue(d) for d in hard_diffs]
            feedback = self._diff_feedback(hard_diffs, chain, edge_apis)
            snap = self._fail(3, issues, feedback, diffs)
            return snap

        # ── L3b: terminal-node method check ───────────────────────────────────
        # Check methods called on the terminal variable of EVERY extracted path
        # against the known-methods table built from RAG metadata.
        # Previously only chain[-1] (paths[0] terminal) was checked, so a
        # hallucinated method on a Path 2+ terminal (e.g. net.getPins() when
        # dbNet is Path 2's terminal) would silently pass.
        if self._known and chain:
            all_paths = paths if paths else [chain]
            seen_terminals: Set[str] = set()
            tnode_issues: List[str] = []
            for path in all_paths:
                if not path:
                    continue
                terminal = path[-1]
                if terminal in seen_terminals:
                    continue   # same terminal already checked via another path
                seen_terminals.add(terminal)
                tnode_issues.extend(self._check_terminal_methods(code, path))
            if tnode_issues:
                feedback = (
                    "Terminal-node method check FAILED — hallucinated method(s) "
                    "called on acquired object:\n" +
                    "\n".join(f"  {iss}" for iss in tnode_issues)
                )
                return self._fail(3, tnode_issues, feedback, diffs)

        # ── PASS ──────────────────────────────────────────────────────────────
        # Soft diffs (rag had no hit — can't compare) are stored as warnings
        warnings = [self._diff_issue(d) for d in diffs if not d.is_hallucinated]
        return self._pass(warnings, diffs)

    # ── L2: flow check ────────────────────────────────────────────────────────

    def _build_type_env(self, code: str, chain: list,
                        edge_apis: list) -> Dict[str, str]:
        """
        Variable Origin tracking: forward-walk the AST and build var → OpenROAD-type.

        Seeded with design = openroad.Design (always pre-available in the shell).
        For every assignment  var = recv.method()  where recv's type is known and
        (recv_type, method) is a recognised acquisition, records var → tgt_type.
        """
        # (src_type, method_name) → return_type  — built from chain + well-known methods
        lookup: Dict[Tuple[str, str], str] = {
            ("openroad.Design", "getBlock"): "odb.dbBlock",
            ("openroad.Design", "getDb"):    "odb.dbDatabase",
            ("openroad.Design", "getTech"):  "openroad.Tech",
            ("odb.dbDatabase",  "getTech"):  "odb.dbTech",
        }
        for i, (src, tgt) in enumerate(zip(chain[:-1], chain[1:])):
            # Leaf-action edges ([methodName] bracket notation) don't produce a
            # typed variable — skip them in the lookup so we don't pollute return types.
            if tgt.startswith("[") and tgt.endswith("]"):
                continue
            api    = edge_apis[i] if i < len(edge_apis) else None
            method = _extract_method_name(api)
            if method:
                lookup[(src, method)] = tgt

        env: Dict[str, str] = {"design": "openroad.Design"}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return env
        self._type_walk_body(tree.body, env, lookup)
        return env

    def _type_walk_body(self, stmts: list, env: Dict[str, str],
                        lookup: Dict[Tuple[str, str], str]) -> None:
        """Sequential statement walk — respects execution order."""
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        t = self._infer_call_type(stmt.value, env, lookup)
                        if t:
                            env[tgt.id] = t
                        # List/set/generator comprehension:
                        # e.g. nand_insts = [inst for inst in block.getInsts() if ...]
                        # Type the comprehension loop variable so downstream
                        # method checks on it don't false-fire.
                        elif isinstance(stmt.value,
                                        (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                            self._walk_comprehension(stmt.value, env, lookup)
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and stmt.value:
                    t = self._infer_call_type(stmt.value, env, lookup)
                    if t:
                        env[stmt.target.id] = t
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    # Case 1: for inst in block.getInsts() — call return type
                    iter_type = self._infer_call_type(stmt.iter, env, lookup)
                    # Case 2: for net in nets — same type as the collection variable
                    if iter_type is None and isinstance(stmt.iter, ast.Name):
                        iter_type = env.get(stmt.iter.id)
                    if iter_type:
                        env[stmt.target.id] = iter_type
                self._type_walk_body(stmt.body, env, lookup)
            elif isinstance(stmt, ast.If):
                self._type_walk_body(stmt.body, env, lookup)
                self._type_walk_body(stmt.orelse, env, lookup)
            elif isinstance(stmt, ast.Try):
                self._type_walk_body(stmt.body, env, lookup)
                for h in stmt.handlers:
                    self._type_walk_body(h.body, env, lookup)

    def _walk_comprehension(self, comp, env: Dict[str, str],
                            lookup: Dict[Tuple[str, str], str]) -> None:
        """Type the loop variable(s) inside a list/set/generator comprehension.

        e.g. [inst for inst in block.getInsts() if inst.getMaster()...]
        → env['inst'] = 'odb.dbInst'

        Mirrors the for-loop logic in _type_walk_body so the type environment
        is populated and downstream _check_flow checks don't false-fire.
        """
        for generator in comp.generators:
            if isinstance(generator.target, ast.Name):
                # Case 1: iter is a call  e.g. block.getInsts()
                iter_type = self._infer_call_type(generator.iter, env, lookup)
                # Case 2: iter is a variable  e.g. [x for x in insts]
                if iter_type is None and isinstance(generator.iter, ast.Name):
                    iter_type = env.get(generator.iter.id)
                if iter_type:
                    env[generator.target.id] = iter_type

    def _infer_call_type(self, expr, env: Dict[str, str],
                         lookup: Dict[Tuple[str, str], str]) -> Optional[str]:
        """Return the OpenROAD type if expr is a method call on a known-typed variable."""
        if not isinstance(expr, ast.Call):
            return None
        func = expr.func
        if not isinstance(func, ast.Attribute):
            return None
        recv = func.value
        if not isinstance(recv, ast.Name):
            return None
        recv_type = env.get(recv.id)
        if recv_type:
            return lookup.get((recv_type, func.attr))
        return None

    def _check_flow(self, code: str, chain: list,
                    edge_apis: list,
                    type_env: Optional[Dict[str, str]] = None) -> List[str]:
        """
        Structural causal flow check using the type environment.

        For each edge (src → tgt):
          1. Source variable of type `src` must exist (Chain Continuity).
          2. The acquisition method must be called on a `src`-typed variable
             (Parent Validation).
          3. If the method appears on a variable of the WRONG type →
             HIERARCHY VIOLATION (B3 Hierarchical Causal Misalignment).
          4. If the method is absent entirely → MISSING STEP.
        """
        if type_env is None:
            type_env = self._build_type_env(code, chain, edge_apis)

        # Invert: type → [variable names]
        type_vars: Dict[str, List[str]] = {}
        for var, typ in type_env.items():
            type_vars.setdefault(typ, []).append(var)

        issues = []
        for i, (src, tgt) in enumerate(zip(chain[:-1], chain[1:])):
            # openroad.Design is always pre-available — never missing
            if src == "openroad.Design":
                continue

            api             = edge_apis[i] if i < len(edge_apis) else None
            expected_method = _extract_method_name(api)
            src_short       = src.split(".")[-1]
            # Leaf-action tgt is [methodName] — strip brackets for display
            tgt_short       = tgt[1:-1] if (tgt.startswith("[") and tgt.endswith("]")) else tgt.split(".")[-1]
            src_vars        = type_vars.get(src, [])

            # ── Check 1: source type must be present in env (Chain Continuity) ─
            if not src_vars:
                issues.append(
                    f"Step {i+1} MISSING SOURCE: no variable of type '{src_short}' "
                    f"found — cannot acquire {tgt_short}. "
                    f"Did you forget the preceding acquisition step?"
                )
                continue

            if not expected_method:
                continue

            # ── Check 2: method called on a correctly-typed receiver ───────────
            # Accept the RAG-suggested method OR any known valid alternative for
            # this edge — either is sufficient to satisfy the acquisition step.
            _EDGE_ALTERNATIVES: Dict[Tuple[str, str], List[str]] = {
                ("odb.dbBlock", "odb.dbInst"):  ["getInsts", "findInst"],
                ("odb.dbBlock", "odb.dbNet"):   ["getNets",  "findNet"],
                ("odb.dbBlock", "odb.dbBTerm"): ["getBTerms","findBTerm"],
                ("odb.dbBlock", "odb.dbITerm"): ["getITerms","findITerm"],
                ("odb.dbInst",  "odb.dbITerm"): ["getITerms","findITerm"],
                ("odb.dbNet",   "odb.dbITerm"): ["getITerms"],
                ("odb.dbNet",   "odb.dbBTerm"): ["getBTerms"],
            }
            accepted = _EDGE_ALTERNATIVES.get((src, tgt), [expected_method])
            if expected_method not in accepted:
                accepted = [expected_method] + accepted

            correct_call = any(
                any(m in _scan_methods_on_var(code, v) for m in accepted)
                for v in src_vars
            )
            if correct_call:
                continue  # edge satisfied

            # ── Check 3: method exists but on the wrong receiver (B3) ──────────
            pattern = re.compile(
                r'\b(\w+)\s*\.\s*' + re.escape(expected_method) + r'\s*\('
            )
            wrong_callers = [
                (m.group(1), type_env.get(m.group(1), "unknown"))
                for m in pattern.finditer(code)
                if type_env.get(m.group(1), "unknown") not in ("unknown", src)
            ]

            if wrong_callers:
                caller_var, caller_type = wrong_callers[0]
                issues.append(
                    f"Step {i+1} HIERARCHY VIOLATION: '{expected_method}()' is called "
                    f"on '{caller_var}' (type {caller_type.split('.')[-1]}) but "
                    f"this method belongs to {src_short}, not "
                    f"{caller_type.split('.')[-1]}. "
                    f"Fix: call it on a {src_short} variable → "
                    f"{src_vars[0]}.{expected_method}(...)"
                )
            else:
                # Method completely absent from code
                issues.append(
                    f"Step {i+1} MISSING: '{src_vars[0]}' ({src_short}) "
                    f"never calls '{expected_method}()' to acquire {tgt_short}. "
                    f"Expected: {src_vars[0]}.{expected_method}(...)"
                )

        return issues

    def _check_odb_module_accesses(self, code: str) -> List[str]:
        """
        Scan for odb.UpperCaseName accesses (no parentheses) and flag any
        that are not in the known valid odb module exports.

        Catches hallucinated enum/constant names like odb.SignalType, odb.Orient,
        odb.Orientation. For known patterns, provides actionable fix instructions
        (method calls to use instead) rather than suggesting another odb.* name.
        """
        issues = []
        seen: Set[str] = set()
        for m in _ODB_MODULE_ACCESS.finditer(code):
            name = m.group(1)
            if name in seen:
                continue
            seen.add(name)
            if name not in _VALID_ODB_CLASSES:
                hint = _ODB_ENUM_HINTS.get(name, f"'odb.{name}' does not exist as a module-level constant.")
                issues.append(hint)
        return issues

    # ── L2c: null-safety ─────────────────────────────────────────────────────

    # Methods that return a single odb.db* object (not a list) and CAN return
    # None — these require a guard before attribute access.
    _NULLABLE_PREFIXES = ("find",)   # find* always nullable
    # get* methods that return a collection never return None; skip them.
    _COLLECTION_RETURN_RE = re.compile(r'^list\(', re.IGNORECASE)

    def _check_null_safety(self, code: str,
                           type_env: Dict[str, str]) -> List[str]:
        """L2c: find*() results that return a single odb.db* object must be
        None-checked before any attribute access on the assigned variable.

        Cases 2, 7, 17 crashed OpenROAD (Signal 11) because findInst/getNet
        returned None and the code immediately called .method() on it.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        issues: List[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            var_name = target.id

            # RHS must be a method call  var = obj.method(...)
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not isinstance(func, ast.Attribute):
                continue
            method_name = func.attr

            # Only check methods whose names start with a nullable prefix
            if not any(method_name.startswith(p) for p in self._NULLABLE_PREFIXES):
                continue

            # The assigned variable must be of a single odb.db* type
            var_type = type_env.get(var_name, "")
            if not (var_type.startswith("odb.db") or var_type.startswith("db")):
                continue
            # Skip list returns (collections are never None in OpenROAD)
            if self._COLLECTION_RETURN_RE.match(var_type):
                continue

            # Check: is there any attribute access on var_name in the code?
            attr_re = re.compile(
                r'\b' + re.escape(var_name) + r'\s*\.\s*[A-Za-z_]'
            )
            if not attr_re.search(code):
                continue   # variable never dereferenced — safe

            # Check: is there a None guard anywhere in the code for var_name?
            none_guard_re = re.compile(
                r'\bif\s+' + re.escape(var_name) + r'\s+is\s+not\s+None\b'
                r'|\bif\s+' + re.escape(var_name) + r'\s+is\s+None\b'
                r'|\bif\s+not\s+' + re.escape(var_name) + r'\s*[:\n]'
                r'|\bif\s+' + re.escape(var_name) + r'\s*:'
                r'|\bif\s+' + re.escape(var_name) + r'\s*(?:and|or)\b',
                re.IGNORECASE,
            )
            if none_guard_re.search(code):
                continue   # guard exists — safe

            issues.append(
                f"Safety Violation: '{var_name}' is assigned from '{method_name}()' "
                f"which can return None, but is accessed without a None check. "
                f"Add: 'if {var_name} is not None:' before accessing its attributes "
                f"to prevent a tool crash."
            )

        return issues

    def _flow_feedback(self, issues: List[str], chain: list,
                       edge_apis: list) -> str:
        lines = ["Causal flow check FAILED — acquisition step(s) entirely absent:"]
        for issue in issues:
            lines.append(f"  {issue}")
        lines.append("\nFull expected acquisition sequence:")
        for i, (src, tgt) in enumerate(zip(chain[:-1], chain[1:]), 1):
            api    = edge_apis[i-1] if i-1 < len(edge_apis) else None
            method = _extract_method_name(api) or "???"
            src_var = _var(src)
            lines.append(
                f"  Step {i}: {src_var}.{method}() → {tgt.split('.')[-1]}"
            )
        return "\n".join(lines)

    # ── L3: API diff ─────────────────────────────────────────────────────────

    def _build_api_diffs(self, code: str, chain: list,
                         edge_apis: list,
                         type_env: Optional[Dict[str, str]] = None) -> List[APIEdgeDiff]:
        """
        For every edge:
          1. Find what method(s) the LLM called on variables of the source type
             (uses type_env, not just the canonical variable name).
          2. Compare with what RAG retrieved.
          3. Build an APIEdgeDiff for any mismatch.

        The `bool(code_methods)` gate that previously suppressed misses when the
        correct receiver had zero calls (because the method was on a wrong-typed
        receiver) is removed: a missing RAG method is flagged regardless.
        """
        if type_env is None:
            type_env = self._build_type_env(code, chain, edge_apis)

        # Invert: type → [variable names known to hold that type]
        type_vars: Dict[str, List[str]] = {}
        for var, typ in type_env.items():
            type_vars.setdefault(typ, []).append(var)

        diffs = []
        for i, (src, tgt) in enumerate(zip(chain[:-1], chain[1:])):
            api        = edge_apis[i] if i < len(edge_apis) else None
            rag_method = _extract_method_name(api)

            # Collect methods called on ALL known variables of the source type.
            # Falls back to canonical name + plural if type_env has no entry.
            src_vars     = type_vars.get(src, [_var(src), _var(src) + "s"])
            code_methods: List[str] = []
            for sv in src_vars:
                for m in _scan_methods_on_var(code, sv):
                    if m not in code_methods:
                        code_methods.append(m)

            if not rag_method:
                if code_methods:
                    diffs.append(APIEdgeDiff(
                        src_type=src, tgt_type=tgt,
                        rag_method="<no RAG hit>",
                        code_methods=code_methods,
                        is_missing=False,
                        is_hallucinated=False,
                    ))
                continue

            src_short_key = src.split(".")[-1]
            src_known     = self._known.get(src_short_key, set())
            uses_known    = any(m in src_known for m in code_methods)

            # Flag a mismatch when the RAG method is absent AND the LLM did not
            # use any other known-valid method for this source type.
            # NOTE: `bool(code_methods)` gate removed — if the correct receiver
            # has no calls at all (method was called on the wrong receiver), that
            # is still a real mismatch that L2 already reported as a hierarchy
            # violation; L3 records it for the controller's re-retrieval logic.
            # MANDATORY methods (confidence >= 0.85) are always enforced.
            # SUGGESTED methods (0.70–0.85) are soft: any known-valid alternative is fine.
            rag_status   = api.get("status", "MANDATORY") if api else "MANDATORY"
            is_mandatory = rag_status == "MANDATORY"

            if is_mandatory:
                # Hard enforcement: code must use exactly the RAG method (or a
                # known valid alternative for this edge).
                is_mismatch = rag_method not in code_methods
            else:
                # Soft: mismatch only if code used nothing known for this type.
                is_mismatch = rag_method not in code_methods and not uses_known

            if is_mismatch:
                diffs.append(APIEdgeDiff(
                    src_type=src, tgt_type=tgt,
                    rag_method=rag_method,
                    code_methods=code_methods,
                    is_missing=True,
                    is_hallucinated=True,
                ))

        return diffs

    def _diff_issue(self, d: APIEdgeDiff) -> str:
        src_s = d.src_type.split(".")[-1]
        tgt_s = d.tgt_type.split(".")[-1]
        if d.rag_method == "<no RAG hit>":
            return (f"No RAG coverage for {src_s}→{tgt_s}: "
                    f"code used {d.code_methods} — unverified.")
        return (
            f"API mismatch for {src_s}→{tgt_s}: "
            f"code used {d.code_methods}, RAG says '{d.rag_method}()'. "
            f"Likely hallucination."
        )

    def _diff_feedback(self, diffs: List[APIEdgeDiff],
                       chain: list, edge_apis: list) -> str:
        lines = ["API diff analysis — hallucinated method(s) detected:"]
        for d in diffs:
            src_s = d.src_type.split(".")[-1]
            tgt_s = d.tgt_type.split(".")[-1]
            lines.append(f"\n  Edge: {src_s} → {tgt_s}")
            lines.append(f"    Code used   : {d.code_methods}")
            lines.append(f"    RAG says    : {d.rag_method}()")
            lines.append(f"    Action      : re-retrieve this edge with lower threshold")
            lines.append(f"                  + code examples from RAGCodePiece.csv")
        lines.append("\nFull expected chain:")
        for i, (src, tgt) in enumerate(zip(chain[:-1], chain[1:]), 1):
            api    = edge_apis[i-1] if i-1 < len(edge_apis) else None
            method = _extract_method_name(api) or "???"
            lines.append(
                f"  Step {i}: {src.split('.')[-1]}.{method}() → {tgt.split('.')[-1]}"
            )
        return "\n".join(lines)

    # ── L3b: terminal-node method check ──────────────────────────────────────

    def _check_terminal_methods(self, code: str, chain: list) -> List[str]:
        """
        For the terminal node type in the chain, scan every variable name that
        could hold that type and check that the methods called on it are known.

        Only fires when the known-methods table has ≥3 entries for that type
        (avoids false positives for sparse types).
        """
        tgt      = chain[-1]
        tgt_short = tgt.split(".")[-1]   # e.g. "dbNet"
        known    = self._known.get(tgt_short, set())
        if len(known) < 3:
            return []   # too sparse to be reliable

        tgt_var  = _var(tgt)
        issues   = []

        for var_name in (tgt_var, tgt_var + "s"):
            methods = _scan_methods_on_var(code, var_name)
            for m in methods:
                if m not in known:
                    close = difflib.get_close_matches(m, known, n=1, cutoff=0.6)
                    suggestion = f" Did you mean '{close[0]}()'?" if close else ""
                    issues.append(
                        f"'{var_name}.{m}()' — '{m}' is not a known method of "
                        f"{tgt_short}.{suggestion}"
                    )
        return issues

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fail(self, layer: int, issues: List[str],
              feedback: str, diffs: list) -> VerifierSnapshot:
        return VerifierSnapshot(
            passed=False, layer_failed=layer,
            issues=issues, feedback=feedback,
            api_diffs=diffs,
        )

    def _pass(self, warnings: List[str], diffs: list) -> VerifierSnapshot:
        return VerifierSnapshot(
            passed=True, layer_failed=0,
            issues=warnings,
            feedback="PASS" + (
                f"\nWarnings (no RAG coverage):\n" +
                "\n".join(f"  - {w}" for w in warnings)
                if warnings else ""
            ),
            api_diffs=diffs,
        )
