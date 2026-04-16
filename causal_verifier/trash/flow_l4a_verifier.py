"""flow_l4a_verifier.py — L4a: High-level action ordering verifier.

Checks that generated code respects the ACTION GRAPH ordering defined by
MultiActionChains (the blue boxes) — not just a single tool's method order.

Four checks (all AST-based, no LLM):

  A1 — Getter present     : design.<getter>() must appear for EVERY tool
                            action in the graph.
  A2 — Cross-tool ordering: the last method of tool_i must appear BEFORE
                            the first getter of tool_{i+1}.
  A3 — Multi-tool sandwich: (only when MultiActionChains.sandwich is True)
                            pre_<label> must be assigned BEFORE the first
                            tool's getter; post_<label> must be assigned
                            AFTER the last tool's last method.
  A4 — Mode integrity     : per-tool mutually exclusive group check
                            (same logic as C3 in FlowSequencingVerifier).

Reuses SequencingIssue and L4SequencingResult from flow_sequencing_verifier.
Does NOT modify flow_sequencing_verifier.py.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

try:
    from .flow_sequencing_verifier import (
        SequencingIssue, L4SequencingResult,
        _MethodCallCollector, _AssignmentCollector, _first_line,
        _apply_edits, _leading_spaces, _first_code_line,
    )
    from .flow_multi_chain_extractor import MultiActionChains, ActionChain
    from .flow_tool_library import FlowToolDef
except ImportError:
    from flow_sequencing_verifier import (                          # type: ignore
        SequencingIssue, L4SequencingResult,
        _MethodCallCollector, _AssignmentCollector, _first_line,
        _apply_edits, _leading_spaces, _first_code_line,
    )
    from flow_multi_chain_extractor import MultiActionChains, ActionChain  # type: ignore
    from flow_tool_library import FlowToolDef                      # type: ignore


# ── FlowL4aVerifier ───────────────────────────────────────────────────────────

class FlowL4aVerifier:
    """L4a verifier: checks action-graph-level ordering in generated code.

    Usage
    -----
    verifier = FlowL4aVerifier()
    result   = verifier.verify(code, multi_chains)
    if not result.passed:
        print(result.feedback)
    """

    def verify(
        self, code: str, multi_chains: MultiActionChains
    ) -> L4SequencingResult:
        """Run A1–A4 checks against *multi_chains* and return a result."""

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return L4SequencingResult(
                passed=False,
                issues=[SequencingIssue(
                    check="A0",
                    description=f"SyntaxError — cannot run L4a: {exc}",
                    repair_hint="Fix the syntax error before sequencing checks.",
                )],
                feedback=f"L4a skipped: SyntaxError — {exc}",
            )

        tool_chains = multi_chains.tool_chains

        # ── Collect call sites for all tool methods + getters ─────────────────
        # One calls dict per tool chain so we can reason per-tool.
        all_known: Set[str] = set()
        for tc in tool_chains:
            td = tc.tool_def
            if td:
                all_known |= td.all_known_methods() | {td.getter}

        collector = _MethodCallCollector(all_known)
        collector.visit(tree)
        calls = collector.calls   # method_name → sorted list of line numbers

        # ── Collect sandwich variable assignments ─────────────────────────────
        sandwich_vars: Set[str] = set()
        if multi_chains.sandwich and multi_chains.metric_chains:
            label = multi_chains.metric_chains[0].action_node.metric_label or ""
            pre_var  = f"pre_{label}"
            post_var = f"post_{label}"
            sandwich_vars = {pre_var, post_var}
        else:
            pre_var = post_var = ""

        assign_collector = _AssignmentCollector(sandwich_vars)
        assign_collector.visit(tree)
        assignments = assign_collector.assignments

        issues: List[SequencingIssue] = []

        # ── A1: getter present for every tool ─────────────────────────────────
        for tc in tool_chains:
            td = tc.tool_def
            if not td:
                continue
            getter_line = _first_line(calls, td.getter)
            if getter_line == -1:
                issues.append(SequencingIssue(
                    check="A1",
                    description=(
                        f"Tool acquisition missing for '{td.tool_id}': "
                        f"design.{td.getter}() not found. "
                        f"Must acquire {td.tool_type} before calling its methods."
                    ),
                    method_a=td.getter,
                    line_a=-1,
                    repair_hint=(
                        f"Add: {td.var_name} = design.{td.getter}()  "
                        f"before the first '{td.tool_id}' method call."
                    ),
                ))

        # ── A2: cross-tool ordering ────────────────────────────────────────────
        # For each consecutive pair (tool_i, tool_{i+1}):
        # last method line of tool_i < first getter line of tool_{i+1}
        for i in range(len(tool_chains) - 1):
            tc_i  = tool_chains[i]
            tc_j  = tool_chains[i + 1]
            td_i  = tc_i.tool_def
            td_j  = tc_j.tool_def
            if not td_i or not td_j:
                continue

            # Last line of tool_i: last method call OR getter if no methods
            method_lines_i = [
                ln
                for m in tc_i.method_sequence
                for ln in calls.get(m, [])
            ]
            getter_line_i  = _first_line(calls, td_i.getter)
            last_line_i    = max(method_lines_i) if method_lines_i else getter_line_i

            # First line of tool_j: its getter
            getter_line_j  = _first_line(calls, td_j.getter)

            if last_line_i == -1 or getter_line_j == -1:
                # A1 already captures missing getters; skip cross-check
                continue

            if last_line_i >= getter_line_j:
                issues.append(SequencingIssue(
                    check="A2",
                    description=(
                        f"Wrong action order: '{td_j.tool_id}' starts "
                        f"(line {getter_line_j}) before '{td_i.tool_id}' "
                        f"finishes (line {last_line_i}). "
                        f"All '{td_i.tool_id}' calls must precede "
                        f"'{td_j.tool_id}'."
                    ),
                    method_a=td_i.tool_id,
                    method_b=td_j.tool_id,
                    line_a=last_line_i,
                    line_b=getter_line_j,
                    repair_hint=(
                        f"Move all '{td_j.tool_id}' code (from line "
                        f"{getter_line_j}) to after line {last_line_i}."
                    ),
                ))

        # ── A3: multi-tool sandwich ────────────────────────────────────────────
        if multi_chains.sandwich and pre_var and tool_chains:
            first_td = tool_chains[0].tool_def
            last_tc  = tool_chains[-1]
            last_td  = last_tc.tool_def

            # Anchor lines
            first_getter_line = _first_line(calls, first_td.getter) if first_td else -1
            last_method_lines = [
                ln
                for m in last_tc.method_sequence
                for ln in calls.get(m, [])
            ]
            last_getter_line  = _first_line(calls, last_td.getter) if last_td else -1
            last_tool_line    = (
                max(last_method_lines) if last_method_lines else last_getter_line
            )

            # A3-pre: pre_var assigned before first tool's getter
            pre_lines = assignments.get(pre_var, [])
            if not pre_lines:
                issues.append(SequencingIssue(
                    check="A3_pre",
                    description=(
                        f"Sandwich missing: '{pre_var}' is never assigned. "
                        f"A pre-measurement must be captured BEFORE the first "
                        f"tool block (before line {first_getter_line})."
                    ),
                    method_a=pre_var,
                    line_a=-1,
                    line_b=first_getter_line,
                    repair_hint=(
                        f"Add: {pre_var} = <measure>  "
                        f"before line {first_getter_line} "
                        f"(design.{first_td.getter if first_td else '?'}())."
                    ),
                ))
            elif first_getter_line != -1 and min(pre_lines) > first_getter_line:
                issues.append(SequencingIssue(
                    check="A3_pre",
                    description=(
                        f"Sandwich order: '{pre_var}' assigned on line "
                        f"{min(pre_lines)}, AFTER the first tool getter on "
                        f"line {first_getter_line}. Pre-measurement must come first."
                    ),
                    method_a=pre_var,
                    line_a=min(pre_lines),
                    line_b=first_getter_line,
                    repair_hint=(
                        f"Move '{pre_var} = ...' to before line {first_getter_line}."
                    ),
                ))

            # A3-post: post_var assigned after last tool's last method
            post_lines = assignments.get(post_var, [])
            if not post_lines:
                issues.append(SequencingIssue(
                    check="A3_post",
                    description=(
                        f"Sandwich missing: '{post_var}' is never assigned. "
                        f"A post-measurement must be captured AFTER the last "
                        f"tool block (after line {last_tool_line})."
                    ),
                    method_b=post_var,
                    line_a=last_tool_line,
                    line_b=-1,
                    repair_hint=(
                        f"Add: {post_var} = <measure>  "
                        f"after line {last_tool_line}."
                    ),
                ))
            elif last_tool_line != -1 and max(post_lines) < last_tool_line:
                issues.append(SequencingIssue(
                    check="A3_post",
                    description=(
                        f"Sandwich order: '{post_var}' last assigned on line "
                        f"{max(post_lines)}, BEFORE the last tool method on "
                        f"line {last_tool_line}. Post-measurement must come after."
                    ),
                    method_b=post_var,
                    line_a=last_tool_line,
                    line_b=max(post_lines),
                    repair_hint=(
                        f"Move '{post_var} = ...' to after line {last_tool_line}."
                    ),
                ))

        # ── A4: mode integrity (per tool, same as C3) ─────────────────────────
        for tc in tool_chains:
            td = tc.tool_def
            if not td:
                continue
            for group in td.exclusive_groups:
                active = [
                    i for i, option in enumerate(group.options)
                    if any(_first_line(calls, m) != -1 for m in option)
                ]
                if len(active) > 1:
                    opt_strs = ["+".join(group.options[i]) for i in active]
                    issues.append(SequencingIssue(
                        check="A4",
                        description=(
                            f"Mode mixing in '{td.tool_id}' group "
                            f"'{group.group_id}': methods from {opt_strs} "
                            f"all appear. {group.description}"
                        ),
                        repair_hint=(
                            f"Choose exactly ONE mode for '{group.group_id}' "
                            f"and remove all methods from the other mode(s)."
                        ),
                    ))

        if not issues:
            return L4SequencingResult.ok()

        feedback = self._build_feedback(issues, multi_chains)
        return L4SequencingResult(
            passed=False,
            issues=issues,
            feedback=feedback,
            repaired_code="",   # repair for multi-tool is left to re-generation
        )

    # ── Feedback builder ──────────────────────────────────────────────────────

    def _build_feedback(
        self, issues: List[SequencingIssue], multi_chains: MultiActionChains
    ) -> str:
        lines = [
            f"L4a ACTION ORDERING FAIL — {len(issues)} issue(s):",
            f"Expected order: {multi_chains.action_graph.ordering_summary()}",
            "",
        ]
        for iss in issues:
            lines.append(f"  [{iss.check}] {iss.description}")
            if iss.repair_hint:
                lines.append(f"         Fix: {iss.repair_hint}")
            lines.append("")

        if multi_chains.sandwich and multi_chains.metric_chains:
            label = multi_chains.metric_chains[0].action_node.metric_label or "metric"
            first_tool = multi_chains.tool_chains[0].tool_def
            last_tool  = multi_chains.tool_chains[-1].tool_def
            lines += [
                "Required sandwich:",
                f"  pre_{label} = <measure>         ← BEFORE design.{first_tool.getter if first_tool else '?'}()",
                f"  <all tool actions in order>",
                f"  post_{label} = <measure>        ← AFTER last method of {last_tool.tool_id if last_tool else '?'}",
            ]

        return "\n".join(lines)
