"""flow_causal_graph.py — Data structures for the Multi-Path Causal Graph.

For a flow task (e.g. "Run Global Placement and report unplaced instance count")
the causal extractor produces TWO parallel paths instead of one linear chain:

  ObjectPath   — DB hierarchy path to acquire the measurement objects
                 e.g.  Design → Block → Insts
                 Used for: pre-measurement and post-measurement

  ActionPath   — Tool acquisition + ordered method call sequence
                 e.g.  design.getReplace() → doInitialPlace() → doNesterovPlace()

Together they define the Sandwich Structure:

    pre_count  = <ObjectPath measurement>   ← BEFORE tool
    <ActionPath: acquire tool + call methods in sequence>
    post_count = <ObjectPath measurement>   ← AFTER tool
    print(delta)

The MultiPathCausalGraph combines both paths and exposes:
  - ordering_constraints() → List[(method_a, method_b)] for L4 to enforce
  - to_constraint_prompt()  → ready-to-inject string for the code generator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from .flow_tool_library import FlowToolDef, MeasurementSpec
except ImportError:
    from flow_tool_library import FlowToolDef, MeasurementSpec  # type: ignore


# ── Object Path ───────────────────────────────────────────────────────────────

@dataclass
class ObjectPath:
    """Causal acquisition chain used for pre/post measurement.

    Mirrors the existing CausalChain concept but scoped specifically to
    the objects needed for measurement (not for running the tool action).

    Attributes
    ----------
    nodes       : OpenROAD type names in acquisition order.
                  e.g. ["openroad.Design", "odb.dbBlock", "odb.dbInst"]
    measurement : Which MeasurementSpec this path supports.
    pre_var     : Python variable name to hold the pre-action measurement.
                  e.g. "pre_unplaced"
    post_var    : Python variable name to hold the post-action measurement.
                  e.g. "post_unplaced"
    """
    nodes:       List[str]
    measurement: MeasurementSpec
    pre_var:     str = "pre_count"
    post_var:    str = "post_count"

    @property
    def terminal_type(self) -> str:
        """The leaf type being measured (last node)."""
        return self.nodes[-1] if self.nodes else ""

    @property
    def acquisition_summary(self) -> str:
        """Human-readable chain e.g. 'Design → Block → Inst'."""
        return " → ".join(n.split(".")[-1] for n in self.nodes)


# ── Action Path ───────────────────────────────────────────────────────────────

@dataclass
class ActionPath:
    """Tool acquisition and ordered method execution sequence.

    Attributes
    ----------
    tool_def        : The FlowToolDef from the library (ground truth for
                      valid methods, getter, type name).
    tool_var        : Python variable name for the acquired tool object.
                      e.g. "placer" for gpl.Replace
    method_sequence : Ordered list of methods to call on tool_var.
                      Determined by the extractor from the task — may be
                      the full tool default or a task-appropriate subset.
                      L4 enforces that sequence[i] precedes sequence[i+1].
    """
    tool_def:        FlowToolDef
    tool_var:        str
    method_sequence: List[str] = field(default_factory=list)

    @property
    def acquisition_chain(self) -> List[str]:
        """Type path to acquire the tool: [openroad.Design, tool_type]."""
        return ["openroad.Design", self.tool_def.tool_type]

    @property
    def getter_call(self) -> str:
        """The call to acquire the tool e.g. 'design.getReplace()'."""
        return f"design.{self.tool_def.getter}()"

    def ordering_constraints(self) -> List[Tuple[str, str]]:
        """Pairs (a, b) where a MUST appear before b in the generated code.

        Derived from method_sequence — only consecutive pairs are constrained.
        e.g. ["doInitialPlace", "doNesterovPlace"] → [("doInitialPlace", "doNesterovPlace")]
        """
        seq = self.method_sequence
        return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

    @property
    def action_summary(self) -> str:
        """Human-readable action chain e.g. 'getReplace → doInitialPlace → doNesterovPlace'."""
        parts = [self.getter_call] + [f"{m}()" for m in self.method_sequence]
        return " → ".join(parts)


# ── Multi-Path Causal Graph ───────────────────────────────────────────────────

@dataclass
class MultiPathCausalGraph:
    """Full causal graph for a single-tool flow task.

    Produced by FlowChainExtractor.extract().
    Consumed by:
      - Code generator  (via to_constraint_prompt())
      - L4 verifier     (via ordering_constraints())
      - Repair agent    (via failed_constraints after L4 check)

    Attributes
    ----------
    task          : Original task string.
    object_path   : Measurement acquisition path (DB hierarchy).
    action_path   : Tool acquisition + method sequence.
    has_sandwich  : True when the task asks to report a before/after change.
                    When True, pre_var must be assigned before the tool block
                    and post_var must be assigned after — L4 enforces this.
    """
    task:         str
    object_path:  ObjectPath
    action_path:  ActionPath
    has_sandwich: bool = True

    def ordering_constraints(self) -> List[Tuple[str, str]]:
        """Delegate to ActionPath — these are what L4 enforces."""
        return self.action_path.ordering_constraints()

    def to_constraint_prompt(self) -> str:
        """Build the prompt section injected into the code generator.

        Tells the LLM exactly:
          1. The object acquisition path (for measurement)
          2. The tool getter + method call order
          3. The sandwich structure (if applicable)
          4. The ordering constraint that L4 will enforce
        """
        ap  = self.action_path
        op  = self.object_path
        td  = ap.tool_def
        ms  = op.measurement

        lines = [
            f"=== FLOW TASK: {td.tool_id.upper().replace('_', ' ')} ===",
            "",
        ]

        # ── Object path (measurement) ──────────────────────────────────────
        lines += [
            "OBJECT PATH (acquire for measurement — use BEFORE and AFTER tool):",
            f"  {op.acquisition_summary}",
            f"  Measurement : {ms.description}",
            f"  Filter hint : {ms.filter_hint}",
            f"  Pre-variable : {op.pre_var}  ← assign BEFORE tool actions",
            f"  Post-variable: {op.post_var}  ← assign AFTER tool actions",
            "",
        ]

        # ── Action path ────────────────────────────────────────────────────
        lines += [
            "ACTION PATH (tool acquisition and execution — IN THIS ORDER):",
            f"  Step 1: {ap.tool_var} = {ap.getter_call}   # acquire {td.tool_type}",
        ]
        for i, method in enumerate(ap.method_sequence, start=2):
            # Annotate ordering constraints inline
            constraints = ap.ordering_constraints()
            notes = []
            for (a, b) in constraints:
                if method == b:
                    notes.append(f"MUST come after {a}()")
                if method == a:
                    notes.append(f"MUST come before {b}()")
            note_str = f"  # {', '.join(notes)}" if notes else ""
            lines.append(f"  Step {i}: {ap.tool_var}.{method}(){note_str}")
        lines.append("")

        # ── Sandwich structure ──────────────────────────────────────────────
        if self.has_sandwich:
            lines += [
                "REQUIRED SANDWICH STRUCTURE:",
                f"  {op.pre_var}  = <measure {ms.label}>   ← BEFORE tool block",
                f"  {ap.tool_var} = {ap.getter_call}",
            ]
            for method in ap.method_sequence:
                lines.append(f"  {ap.tool_var}.{method}()")
            lines += [
                f"  {op.post_var} = <measure {ms.label}>   ← AFTER tool block",
                f"  print(f'Change in {ms.label}: {{{op.pre_var}}} → {{{op.post_var}}}')",
                "",
            ]

        # ── L4 ordering constraints ─────────────────────────────────────────
        constraints = self.ordering_constraints()
        if constraints:
            lines.append("ORDERING CONSTRAINTS (enforced by L4 verifier):")
            for (a, b) in constraints:
                lines.append(f"  {a}() MUST be called BEFORE {b}()")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)

    def summary(self) -> str:
        """One-line summary for logging."""
        sandwich = "sandwich" if self.has_sandwich else "no-sandwich"
        return (
            f"MultiPathCausalGraph | tool={self.action_path.tool_def.tool_id} "
            f"| object={self.object_path.acquisition_summary} "
            f"| action={self.action_path.action_summary} "
            f"| {sandwich}"
        )
