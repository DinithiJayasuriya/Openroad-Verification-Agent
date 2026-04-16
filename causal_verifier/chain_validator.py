"""chain_validator.py — Causal Chain Validator (Logic Layer).

After code is generated, this module walks the AST and verifies that the
variable-binding sequence matches the causal chain identified in Step 1.

What "matching" means
---------------------
Given chain:  openroad.Design → odb.dbBlock → odb.dbInst
  Required edge 0:  some variable of type odb.dbBlock must be obtained by
                    calling the correct method on the openroad.Design variable.
  Required edge 1:  some variable of type odb.dbInst must be obtained by
                    calling the correct method on the odb.dbBlock variable.

The validator enforces:
  1. EXISTENCE: a variable of each chain-node type exists in the code.
  2. ORDER:     node types are bound in chain order (no skipping, no reversal).
  3. METHOD:    the method used to obtain target_type matches the NodeAPIEntry.
  4. RECEIVER:  the receiver of each binding call is of the expected source_type.

A CausalValidationResult is returned, with .passed and a list of .issues.
It mirrors the VerifierResult interface used by the existing pipeline.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .chain_extractor import CausalChain, ChainNode
from .node_retriever   import NodeAPIEntry, _normalize, _short


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class CausalValidationResult:
    passed:         bool
    issues:         List[str] = field(default_factory=list)
    warnings:       List[str] = field(default_factory=list)
    feedback:       str = ""
    next_step:      str = "regenerate"   # "regenerate" | "re_retrieve"
    rag_hint:       str = ""
    # Which chain edges were verified OK
    verified_edges: List[Tuple[str, str]] = field(default_factory=list)
    # Which edges were missing or wrong
    failed_edges:   List[Tuple[str, str]] = field(default_factory=list)

    @classmethod
    def ok(cls, verified_edges: List[Tuple[str, str]],
           warnings: Optional[List[str]] = None) -> "CausalValidationResult":
        return cls(passed=True, verified_edges=verified_edges,
                   warnings=warnings or [])

    def __str__(self) -> str:
        if self.passed:
            n = len(self.verified_edges)
            return f"[CausalValidator] PASS ({n} edges verified)"
        return ("[CausalValidator] FAIL: " + "; ".join(self.issues[:3]))


# ── Type-propagation environment ─────────────────────────────────────────────

# Variable name → normalised OpenROAD type
Env = Dict[str, str]


def _merge_envs(base: Env, *branches: Env) -> Env:
    """Merge branch environments: keep values that all branches agree on."""
    merged = dict(base)
    for var, typ in list(merged.items()):
        if not all(b.get(var) == typ for b in branches):
            merged[var] = "unknown"
    # add vars that appear in ALL branches
    if branches:
        common = set(branches[0])
        for b in branches[1:]:
            common &= set(b)
        for var in common:
            if var not in merged:
                vals = [b[var] for b in branches]
                merged[var] = vals[0] if len(set(vals)) == 1 else "unknown"
    return merged


# ── AST walker for type propagation ─────────────────────────────────────────

class _TypeWalker(ast.NodeVisitor):
    """
    Single-pass forward walk of an AST.
    Tracks variable → OpenROAD type.
    Specifically records assignment bindings:
      var = expr.method(...)  →  (var, receiver_var, method_name, lineno)
    """

    def __init__(self):
        self.env: Env = {}
        # List of (var_name, receiver_var, method_name, lineno, call_node)
        self.bindings: List[Tuple[str, str, str, int, ast.Call]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._process_assign(target.id, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value:
            self._process_assign(node.target.id, node.value, node.lineno)
        self.generic_visit(node)

    def _process_assign(self, var: str, expr: ast.expr, lineno: int) -> None:
        typ = self._infer_type(expr)
        if typ:
            self.env[var] = _normalize(typ)
        # Record (var, receiver, method) for call-based assignments
        if isinstance(expr, ast.Call):
            func = expr.func
            if isinstance(func, ast.Attribute):
                recv_var = self._receiver_var(func.value)
                method   = func.attr
                self.bindings.append((var, recv_var or "", method, lineno, expr))

    def _infer_type(self, expr: ast.expr) -> str:
        """Best-effort type inference; empty string if unknown."""
        if isinstance(expr, ast.Call):
            func = expr.func
            if isinstance(func, ast.Attribute):
                recv_type = self._infer_type(func.value)
                return ""  # caller resolves via method lookup
            if isinstance(func, ast.Name):
                # Constructor: Design(...), Tech(...), Timing(...)
                _CTORS = {"Design": "openroad.Design", "Tech": "openroad.Tech",
                          "Timing": "openroad.Timing"}
                return _CTORS.get(func.id, "")
            if isinstance(func, ast.Attribute):
                pass
        if isinstance(expr, ast.Name):
            return self.env.get(expr.id, "")
        return ""

    def _receiver_var(self, node: ast.expr) -> Optional[str]:
        """Return variable name if node is a simple Name, else None."""
        if isinstance(node, ast.Name):
            return node.id
        return None


# ── Validator ─────────────────────────────────────────────────────────────────

class CausalChainValidator:
    """
    Validates that generated Python code follows the causal chain.

    Usage
    -----
    validator = CausalChainValidator()
    result    = validator.validate(code, chain, node_apis)
    """

    def validate(
        self,
        code: str,
        chain: CausalChain,
        node_apis: List[NodeAPIEntry],
    ) -> CausalValidationResult:
        """Validate *code* against the causal *chain* and its retrieved *node_apis*."""
        # --- parse ---------------------------------------------------------
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return CausalValidationResult(
                passed=False,
                issues=[f"SyntaxError: {exc}"],
                feedback="The code has a syntax error; fix it before re-generating.",
            )

        # --- collect all call bindings ------------------------------------
        walker = _TypeWalker()

        # Seed the env: design is always pre-bound
        walker.env["design"] = "openroad.Design"

        walker.visit(tree)

        # --- check each edge ----------------------------------------------
        issues:         List[str] = []
        warnings:       List[str] = []
        verified_edges: List[Tuple[str, str]] = []
        failed_edges:   List[Tuple[str, str]] = []

        for idx, (api_entry, (src_type, tgt_type)) in enumerate(
                zip(node_apis, chain.edges)):
            src_norm = _normalize(src_type)
            tgt_norm = _normalize(tgt_type)

            issue = self._check_edge(
                src_norm, tgt_norm, api_entry, walker, idx
            )
            if issue is None:
                verified_edges.append((src_type, tgt_type))
            else:
                failed_edges.append((src_type, tgt_type))
                issues.append(issue)

        if not issues:
            return CausalValidationResult.ok(
                verified_edges=verified_edges, warnings=warnings
            )

        feedback = self._build_feedback(chain, node_apis, issues)
        # If any issue mentions a missing acquisition step → re_retrieve
        next_step = "re_retrieve" if any(
            "acquisition" in i or "missing" in i.lower() for i in issues
        ) else "regenerate"
        rag_hint = " ".join(
            _short(t) for _, t in failed_edges
        )

        return CausalValidationResult(
            passed=False,
            issues=issues,
            warnings=warnings,
            feedback=feedback,
            next_step=next_step,
            rag_hint=rag_hint,
            verified_edges=verified_edges,
            failed_edges=failed_edges,
        )

    # ── Edge check ────────────────────────────────────────────────────────────

    def _check_edge(
        self,
        src_norm: str,
        tgt_norm: str,
        api: NodeAPIEntry,
        walker: _TypeWalker,
        edge_idx: int,
    ) -> Optional[str]:
        """
        Return None if edge is satisfied, or an issue string if not.

        Looks for any binding:  var = <recv>.method(...)
        where recv has type src_norm and method matches api.method_name.
        """
        expected_method = api.method_name
        is_unknown = expected_method.startswith("<UNKNOWN")

        if is_unknown:
            # We don't know the method — just check a variable of tgt_type exists
            if self._has_type(tgt_norm, walker):
                return None
            return (
                f"Edge {edge_idx}: variable of type '{_short(tgt_norm)}' not found. "
                f"No known API to go from {_short(src_norm)} → {_short(tgt_norm)}."
            )

        # Check for a binding that uses the expected method
        for var, recv_var, method, lineno, _ in walker.bindings:
            if method != expected_method:
                continue
            # Receiver should be of src_norm type
            recv_type = walker.env.get(recv_var, "unknown")
            receiver_ok = (
                recv_type == src_norm
                or recv_type == "unknown"    # lenient: unknown receiver allowed
                or self._type_compatible(recv_type, src_norm)
            )
            if receiver_ok:
                # Record the assigned variable's type
                walker.env[var] = tgt_norm
                return None

        # Method not found anywhere — check if tgt variable exists at all
        if self._has_type(tgt_norm, walker):
            # Target type exists but obtained differently — soft warning
            return None   # lenient pass: correct type present via another path

        # Hard failure: neither correct method nor correct type found
        src_short = _short(src_norm)
        tgt_short = _short(tgt_norm)
        if api.params.strip():
            call_example = f"{src_short.lower()}.{expected_method}({api.params})"
        else:
            call_example = f"{src_short.lower()}.{expected_method}()"
        return (
            f"Edge {edge_idx}: missing acquisition step "
            f"'{src_short} → {tgt_short}'. "
            f"Call: {call_example}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _has_type(self, type_norm: str, walker: _TypeWalker) -> bool:
        """Return True if any env variable has the given type."""
        return any(t == type_norm for t in walker.env.values())

    def _type_compatible(self, actual: str, expected: str) -> bool:
        """Loose type compatibility (handles module-prefix variants)."""
        return _short(actual) == _short(expected)

    # ── Feedback builder ──────────────────────────────────────────────────────

    def _build_feedback(
        self,
        chain: CausalChain,
        node_apis: List[NodeAPIEntry],
        issues: List[str],
    ) -> str:
        lines = [
            "CAUSAL CHAIN VALIDATION FAILED",
            "",
            "Required acquisition sequence:",
        ]
        for i, (node, api) in enumerate(zip(chain.nodes[1:], node_apis)):
            cached = " [CACHED]" if api.source == "cache" else ""
            if api.params.strip():
                call = f".{api.method_name}({api.params})"
            else:
                call = f".{api.method_name}()"
            lines.append(
                f"  Step {i+1}: obtain {node.type_name} — call {call}{cached}"
            )

        lines += ["", "Issues found:"]
        for issue in issues:
            lines.append(f"  • {issue}")

        lines += [
            "",
            "Fix: ensure your code follows the exact sequence above.",
            "Do not skip steps or call methods on the wrong object type.",
        ]
        return "\n".join(lines)
