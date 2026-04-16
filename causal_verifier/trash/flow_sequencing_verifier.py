"""flow_sequencing_verifier.py — L4: Flow Tool Sequencing Check and Repair.

Verifies that generated code respects the scientifically correct method
call order for OpenROAD flow tools, as defined in the tool library.

Four checks (all AST-based, no LLM needed):

  C1 — Getter present    : design.<getter>() must appear in the code.
  C2 — Method ordering   : for each (a, b) in ordering_constraints(),
                           the first call to a must appear on a lower line
                           number than the first call to b.
  C3 — Mode integrity    : if the tool has mutually exclusive method groups,
                           only one group's methods may appear (no mixing).
  C4 — Sandwich          : (only when graph.has_sandwich is True)
                           pre_var must be assigned BEFORE the getter line;
                           post_var must be assigned AFTER the last method line.

When checks fail, the Repairer applies surgical fixes directly to the code:
  - Wrong order       : swap the two method-call lines
  - Missing method    : insert the missing call at the correct position
  - Missing sandwich  : inject a measurement placeholder at the right position

Result carries both the issue list (for the controller) and the repaired
code (ready to re-verify or commit).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    from .flow_causal_graph import MultiPathCausalGraph
    from .flow_tool_library import MutuallyExclusiveGroup
except ImportError:
    from flow_causal_graph import MultiPathCausalGraph          # type: ignore
    from flow_tool_library import MutuallyExclusiveGroup        # type: ignore


# ── Issue dataclass ───────────────────────────────────────────────────────────

@dataclass
class SequencingIssue:
    """A single L4 sequencing violation.

    Attributes
    ----------
    check       : Which check failed: "C1" | "C2" | "C3" | "C4_pre" | "C4_post"
    description : Human-readable description of the violation.
    method_a    : Primary method / variable involved (earlier in the sequence).
    method_b    : Secondary method / variable involved (later in the sequence).
    line_a      : Line number of method_a in the code (-1 if absent).
    line_b      : Line number of method_b in the code (-1 if absent).
    repair_hint : Specific instruction for the generator or repair agent.
    """
    check:        str
    description:  str
    method_a:     str = ""
    method_b:     str = ""
    line_a:       int = -1
    line_b:       int = -1
    repair_hint:  str = ""


# ── L4 result ─────────────────────────────────────────────────────────────────

@dataclass
class L4SequencingResult:
    """Result of the L4 sequencing check.

    Attributes
    ----------
    passed        : True only when all four checks pass.
    issues        : List of SequencingIssue (one per violation).
    feedback      : Multi-line string summarising all violations for the
                    controller / repair agent.
    repaired_code : Code after surgical repair; empty string if passed or
                    if repair was not possible.
    """
    passed:        bool
    issues:        List[SequencingIssue] = field(default_factory=list)
    feedback:      str = ""
    repaired_code: str = ""

    @property
    def layer_failed(self) -> int:
        return 0 if self.passed else 4

    @classmethod
    def ok(cls) -> "L4SequencingResult":
        return cls(passed=True)


# ── AST helpers ───────────────────────────────────────────────────────────────

class _MethodCallCollector(ast.NodeVisitor):
    """Collect line numbers of every method call whose name is in target_set."""

    def __init__(self, target_set: Set[str]):
        self._targets = target_set
        # method_name → sorted list of line numbers
        self.calls: Dict[str, List[int]] = {}

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name in self._targets:
                self.calls.setdefault(name, []).append(node.lineno)
        self.generic_visit(node)


class _AssignmentCollector(ast.NodeVisitor):
    """Collect line numbers of assignments to target variable names."""

    def __init__(self, target_vars: Set[str]):
        self._targets = target_vars
        self.assignments: Dict[str, List[int]] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in self._targets:
                self.assignments.setdefault(tgt.id, []).append(node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and node.target.id in self._targets:
            self.assignments.setdefault(node.target.id, []).append(node.lineno)
        self.generic_visit(node)


def _first_line(calls: Dict[str, List[int]], method: str) -> int:
    """Return the first line number for *method*, or -1 if absent."""
    lines = calls.get(method, [])
    return min(lines) if lines else -1


# ── Verifier ──────────────────────────────────────────────────────────────────

class FlowSequencingVerifier:
    """L4 verifier: checks method ordering in generated code against the graph.

    Usage
    -----
    verifier = FlowSequencingVerifier()
    result   = verifier.verify(code, graph)
    if not result.passed:
        print(result.feedback)
        fixed = result.repaired_code   # apply surgical repair
    """

    def verify(self, code: str, graph: MultiPathCausalGraph) -> L4SequencingResult:
        """Run all four checks and attempt repair on failure."""

        # Parse
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            issue = SequencingIssue(
                check="C0",
                description=f"SyntaxError — cannot run L4: {exc}",
                repair_hint="Fix the syntax error before sequencing checks.",
            )
            return L4SequencingResult(
                passed=False, issues=[issue],
                feedback=f"L4 skipped: SyntaxError — {exc}",
            )

        td  = graph.action_path.tool_def
        ap  = graph.action_path
        op  = graph.object_path

        # Collect all tool-related method call sites
        all_tool_methods = td.all_known_methods() | {td.getter}
        call_collector = _MethodCallCollector(all_tool_methods)
        call_collector.visit(tree)
        calls = call_collector.calls

        # Collect assignment sites for sandwich variables
        assign_collector = _AssignmentCollector({op.pre_var, op.post_var})
        assign_collector.visit(tree)
        assignments = assign_collector.assignments

        issues: List[SequencingIssue] = []

        # ── C1: getter present ────────────────────────────────────────────────
        getter_line = _first_line(calls, td.getter)
        if getter_line == -1:
            issues.append(SequencingIssue(
                check="C1",
                description=(
                    f"Tool acquisition missing: '{td.getter}()' not found. "
                    f"The code must call design.{td.getter}() to obtain a "
                    f"{td.tool_type} object before calling any tool methods."
                ),
                method_a=td.getter,
                line_a=-1,
                repair_hint=(
                    f"Add: {ap.tool_var} = design.{td.getter}()  "
                    f"before the first tool method call."
                ),
            ))

        # ── C2: method ordering ───────────────────────────────────────────────
        for (method_a, method_b) in ap.ordering_constraints():
            line_a = _first_line(calls, method_a)
            line_b = _first_line(calls, method_b)

            if line_a == -1 and line_b == -1:
                issues.append(SequencingIssue(
                    check="C2",
                    description=(
                        f"Both {method_a}() and {method_b}() are missing. "
                        f"Both must be called on {ap.tool_var} in this order."
                    ),
                    method_a=method_a, method_b=method_b,
                    line_a=-1, line_b=-1,
                    repair_hint=(
                        f"Add: {ap.tool_var}.{method_a}() then "
                        f"{ap.tool_var}.{method_b}() after acquiring the tool."
                    ),
                ))
            elif line_a == -1:
                issues.append(SequencingIssue(
                    check="C2",
                    description=(
                        f"{method_a}() is missing. "
                        f"It MUST be called before {method_b}() "
                        f"(found on line {line_b})."
                    ),
                    method_a=method_a, method_b=method_b,
                    line_a=-1, line_b=line_b,
                    repair_hint=(
                        f"Insert {ap.tool_var}.{method_a}() "
                        f"immediately before line {line_b} ({method_b})."
                    ),
                ))
            elif line_b == -1:
                issues.append(SequencingIssue(
                    check="C2",
                    description=(
                        f"{method_b}() is missing. "
                        f"It MUST be called after {method_a}() "
                        f"(found on line {line_a})."
                    ),
                    method_a=method_a, method_b=method_b,
                    line_a=line_a, line_b=-1,
                    repair_hint=(
                        f"Insert {ap.tool_var}.{method_b}() "
                        f"immediately after line {line_a} ({method_a})."
                    ),
                ))
            elif line_a > line_b:
                issues.append(SequencingIssue(
                    check="C2",
                    description=(
                        f"Wrong order: {method_b}() (line {line_b}) appears "
                        f"BEFORE {method_a}() (line {line_a}). "
                        f"{method_a} MUST precede {method_b}."
                    ),
                    method_a=method_a, method_b=method_b,
                    line_a=line_a, line_b=line_b,
                    repair_hint=(
                        f"Swap lines {line_a} and {line_b}: "
                        f"move {method_a}() before {method_b}()."
                    ),
                ))

        # ── C3: mode integrity (exclusive groups) ─────────────────────────────
        for group in td.exclusive_groups:
            # Determine which options' methods appear in the code
            active_options = [
                i for i, option in enumerate(group.options)
                if any(_first_line(calls, m) != -1 for m in option)
            ]
            if len(active_options) > 1:
                opt_strs = [
                    "+".join(group.options[i]) for i in active_options
                ]
                issues.append(SequencingIssue(
                    check="C3",
                    description=(
                        f"Mode mixing in '{group.group_id}': "
                        f"methods from {opt_strs} all appear. "
                        f"{group.description}"
                    ),
                    repair_hint=(
                        f"Choose exactly ONE mode for {group.group_id} and "
                        f"remove all methods belonging to the other mode(s)."
                    ),
                ))

        # ── C4: sandwich structure ─────────────────────────────────────────────
        if graph.has_sandwich and not issues:
            # Tool block span: getter line → last method line
            all_method_lines = [
                ln
                for m in ap.method_sequence
                for ln in calls.get(m, [])
            ]
            last_method_line = max(all_method_lines) if all_method_lines else getter_line

            # C4-pre: pre_var assigned before getter
            pre_lines = assignments.get(op.pre_var, [])
            if not pre_lines:
                issues.append(SequencingIssue(
                    check="C4_pre",
                    description=(
                        f"Sandwich missing: '{op.pre_var}' is never assigned. "
                        f"A pre-measurement must be captured BEFORE the tool block "
                        f"(before line {getter_line})."
                    ),
                    method_a=op.pre_var,
                    line_a=-1,
                    line_b=getter_line,
                    repair_hint=(
                        f"Add: {op.pre_var} = <measure {op.measurement.label}>  "
                        f"before line {getter_line} ({td.getter}()).\n"
                        f"Hint: {op.measurement.filter_hint}"
                    ),
                ))
            elif min(pre_lines) > getter_line != -1:
                issues.append(SequencingIssue(
                    check="C4_pre",
                    description=(
                        f"Sandwich order: '{op.pre_var}' assigned on line "
                        f"{min(pre_lines)}, AFTER the tool getter on line "
                        f"{getter_line}. Pre-measurement must come first."
                    ),
                    method_a=op.pre_var,
                    line_a=min(pre_lines),
                    line_b=getter_line,
                    repair_hint=(
                        f"Move the '{op.pre_var} = ...' assignment "
                        f"to before line {getter_line}."
                    ),
                ))

            # C4-post: post_var assigned after last method
            post_lines = assignments.get(op.post_var, [])
            if not post_lines:
                issues.append(SequencingIssue(
                    check="C4_post",
                    description=(
                        f"Sandwich missing: '{op.post_var}' is never assigned. "
                        f"A post-measurement must be captured AFTER the tool block "
                        f"(after line {last_method_line})."
                    ),
                    method_b=op.post_var,
                    line_a=last_method_line,
                    line_b=-1,
                    repair_hint=(
                        f"Add: {op.post_var} = <measure {op.measurement.label}>  "
                        f"after line {last_method_line} "
                        f"({ap.method_sequence[-1] if ap.method_sequence else td.getter}()).\n"
                        f"Hint: {op.measurement.filter_hint}"
                    ),
                ))
            elif max(post_lines) < last_method_line:
                issues.append(SequencingIssue(
                    check="C4_post",
                    description=(
                        f"Sandwich order: '{op.post_var}' last assigned on line "
                        f"{max(post_lines)}, BEFORE the last tool method on line "
                        f"{last_method_line}. Post-measurement must come after."
                    ),
                    method_b=op.post_var,
                    line_a=last_method_line,
                    line_b=max(post_lines),
                    repair_hint=(
                        f"Move the '{op.post_var} = ...' assignment "
                        f"to after line {last_method_line}."
                    ),
                ))

        if not issues:
            return L4SequencingResult.ok()

        feedback  = self._build_feedback(issues, graph)
        repaired  = _Repairer.repair(code, issues, graph)
        return L4SequencingResult(
            passed=False, issues=issues,
            feedback=feedback, repaired_code=repaired,
        )

    # ── Feedback builder ──────────────────────────────────────────────────────

    def _build_feedback(
        self, issues: List[SequencingIssue], graph: MultiPathCausalGraph
    ) -> str:
        lines = [
            f"L4 SEQUENCING FAIL — {len(issues)} issue(s) "
            f"for tool '{graph.action_path.tool_def.tool_id}':",
            "",
        ]
        for i, iss in enumerate(issues, 1):
            lines.append(f"  [{iss.check}] {iss.description}")
            if iss.repair_hint:
                lines.append(f"       Fix: {iss.repair_hint}")
        lines += [
            "",
            "Required ordering:",
            f"  {graph.action_path.action_summary}",
        ]
        if graph.has_sandwich:
            op = graph.object_path
            lines += [
                "",
                "Required sandwich:",
                f"  {op.pre_var} = <measure>   ← BEFORE tool block",
                f"  <tool actions>",
                f"  {op.post_var} = <measure>  ← AFTER tool block",
            ]
        return "\n".join(lines)


# ── Surgical repairer ─────────────────────────────────────────────────────────

class _Repairer:
    """Applies minimal line-level fixes to correct sequencing violations.

    Strategy: collect all line-level edits, sort bottom-up (highest line
    first) to avoid index shifting, then apply them.
    """

    @staticmethod
    def repair(
        code: str,
        issues: List[SequencingIssue],
        graph: MultiPathCausalGraph,
    ) -> str:
        """Return repaired code, or empty string if repair is not possible."""
        lines = code.splitlines()
        ap    = graph.action_path
        td    = ap.tool_def

        # Collect edits as (line_index_0based, action, content)
        # action: "swap_with" | "insert_before" | "insert_after"
        edits = []

        for iss in issues:
            idx_a = iss.line_a - 1   # 0-based; -1 means absent
            idx_b = iss.line_b - 1

            if iss.check == "C2":
                if iss.line_a != -1 and iss.line_b != -1:
                    # Wrong order: swap the two lines
                    edits.append(("swap", idx_a, idx_b))

                elif iss.line_a == -1 and iss.line_b != -1:
                    # method_a missing — insert before method_b line
                    indent = _leading_spaces(lines[idx_b])
                    new_line = f"{indent}{ap.tool_var}.{iss.method_a}()"
                    edits.append(("insert_before", idx_b, new_line))

                elif iss.line_a != -1 and iss.line_b == -1:
                    # method_b missing — insert after method_a line
                    indent = _leading_spaces(lines[idx_a])
                    new_line = f"{indent}{ap.tool_var}.{iss.method_b}()"
                    edits.append(("insert_after", idx_a, new_line))

            elif iss.check == "C1" and iss.line_a == -1:
                # Getter missing — insert at line 0 (top of script logic)
                # Find first non-comment, non-blank line as insertion point
                insert_at = _first_code_line(lines)
                new_line  = f"{ap.tool_var} = design.{td.getter}()"
                edits.append(("insert_before", insert_at, new_line))

            elif iss.check == "C4_pre" and iss.line_a == -1:
                # pre_var missing — insert placeholder before getter
                getter_idx = iss.line_b - 1
                if 0 <= getter_idx < len(lines):
                    indent   = _leading_spaces(lines[getter_idx])
                    ms       = graph.object_path.measurement
                    new_line = (
                        f"{indent}{graph.object_path.pre_var} = "
                        f"# TODO: measure {ms.label} — {ms.filter_hint}"
                    )
                    edits.append(("insert_before", getter_idx, new_line))

            elif iss.check == "C4_post" and iss.line_b == -1:
                # post_var missing — insert placeholder after last method
                last_idx = iss.line_a - 1
                if 0 <= last_idx < len(lines):
                    indent   = _leading_spaces(lines[last_idx])
                    ms       = graph.object_path.measurement
                    new_line = (
                        f"{indent}{graph.object_path.post_var} = "
                        f"# TODO: measure {ms.label} — {ms.filter_hint}"
                    )
                    edits.append(("insert_after", last_idx, new_line))

        if not edits:
            return ""   # no repair possible

        return _apply_edits(lines, edits)


# ── Edit application ──────────────────────────────────────────────────────────

def _apply_edits(lines: List[str], edits: list) -> str:
    """Apply collected edits bottom-up to avoid index shifting."""
    # Separate swaps (do first, no index change) from insertions (bottom-up)
    swaps      = [(a, b) for (op, a, b) in edits if op == "swap"]
    insertions = [(op, idx, content) for (op, idx, content) in edits
                  if op in ("insert_before", "insert_after")]

    result = list(lines)

    # Apply swaps
    for (i, j) in swaps:
        if 0 <= i < len(result) and 0 <= j < len(result):
            result[i], result[j] = result[j], result[i]

    # Apply insertions bottom-up (sort by index descending)
    for (op, idx, content) in sorted(insertions, key=lambda x: -x[1]):
        if op == "insert_before" and 0 <= idx <= len(result):
            result.insert(idx, content)
        elif op == "insert_after" and 0 <= idx < len(result):
            result.insert(idx + 1, content)

    return "\n".join(result)


def _leading_spaces(line: str) -> str:
    """Return the leading whitespace of a line (for indentation matching)."""
    return line[: len(line) - len(line.lstrip())]


def _first_code_line(lines: List[str]) -> int:
    """Return 0-based index of the first non-blank, non-comment line."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return i
    return 0
