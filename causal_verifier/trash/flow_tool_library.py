"""flow_tool_library.py — Static library of available OpenROAD flow tools.

Each FlowToolDef captures:
  - Python type name       e.g. "gpl.Replace"
  - Getter on design       e.g. design.getReplace()
  - Required method sequence (ordered — each method must appear before the next)
  - Optional method groups (mutually exclusive alternatives)
  - Identification keywords for prompt-level tool selection
  - Measurement specs: what objects to count/query before and after the tool runs

This library is the ground truth for:
  - Tool identification (FlowChainExtractor)
  - L4 sequencing verification (FlowSequencingVerifier)
  - Causal repair hints (FlowRepair)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ── Method-level spec ─────────────────────────────────────────────────────────

@dataclass
class MethodSpec:
    """A single method call in a tool's execution sequence."""
    name:        str            # e.g. "doInitialPlace"
    required:    bool = True    # if False, optional step
    description: str = ""       # human-readable purpose


@dataclass
class MutuallyExclusiveGroup:
    """A group of methods where EXACTLY ONE should be called.

    Example: gpl.Replace has two placement modes:
      - {doInitialPlace + doNesterovPlace}  (two-step analytical)
      - {doIncrementalPlace}               (single-step incremental)
    Only one mode should be used per script.
    """
    group_id:  str         # e.g. "gpl_mode"
    options:   List[List[str]]  # each inner list = one valid mode (ordered methods)
    description: str = ""


# ── Measurement spec ──────────────────────────────────────────────────────────

@dataclass
class MeasurementSpec:
    """Describes a measurable quantity before/after a tool runs.

    Kept deliberately abstract: provides the causal path and a filter hint
    so the FlowChainExtractor can generate task-appropriate measurement code,
    rather than hard-coding a single expression that may not fit every prompt.

    Used to generate and verify the sandwich structure:
      pre = <measure>
      <tool action>
      post = <measure>
      print(delta)
    """
    label:        str         # e.g. "unplaced_instance_count"
    description:  str         # e.g. "Number of unplaced standard-cell instances"
    object_type:  str         # terminal OpenROAD type e.g. "odb.dbInst"
    causal_path:  List[str]   # acquisition path e.g. ["openroad.Design", "odb.dbBlock", "odb.dbInst"]
    filter_hint:  str         # abstract predicate hint e.g. "instances that are not placed"


# ── Tool definition ───────────────────────────────────────────────────────────

@dataclass
class FlowToolDef:
    """Complete specification of one OpenROAD flow tool."""

    tool_id:      str                  # unique key e.g. "global_placement"
    tool_type:    str                  # Python type e.g. "gpl.Replace"
    getter:       str                  # method on design e.g. "getReplace"
    var_name:     str                  # conventional variable name e.g. "placer"

    # Ordered constraints: the default execution sequence.
    # The sequencing verifier enforces that sequence[i] appears before sequence[i+1].
    default_sequence: List[MethodSpec] = field(default_factory=list)

    # Mutually exclusive groups (alternatives to the default sequence).
    exclusive_groups: List[MutuallyExclusiveGroup] = field(default_factory=list)

    # Keywords used to identify this tool from a task prompt.
    keywords: List[str] = field(default_factory=list)

    # Measurement specs (what to count/report before+after this tool).
    measurements: List[MeasurementSpec] = field(default_factory=list)

    # Human-readable description.
    description: str = ""

    # Stage in the physical design flow (for multi-tool dependency ordering).
    flow_stage: int = 0   # 1=floorplan, 2=placement, 3=cts, 4=routing, 5=signoff

    # Tools that must have already run before this tool (causal prerequisites).
    requires_after: List[str] = field(default_factory=list)  # list of tool_ids

    def all_required_methods(self) -> List[str]:
        """Return the names of all required methods in the default sequence."""
        return [m.name for m in self.default_sequence if m.required]

    def ordering_constraints(self) -> List[tuple]:
        """Return (method_a, method_b) pairs where method_a MUST precede method_b."""
        seq = self.all_required_methods()
        return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

    def all_known_methods(self) -> Set[str]:
        """All method names known for this tool (default + exclusive groups)."""
        methods: Set[str] = {m.name for m in self.default_sequence}
        for g in self.exclusive_groups:
            for option in g.options:
                methods.update(option)
        return methods


# ── Tool library ──────────────────────────────────────────────────────────────

TOOL_LIBRARY: Dict[str, FlowToolDef] = {}

def _reg(t: FlowToolDef) -> FlowToolDef:
    TOOL_LIBRARY[t.tool_id] = t
    return t


# ─── 1. Floorplanning ─────────────────────────────────────────────────────────

_reg(FlowToolDef(
    tool_id    = "floorplanning",
    tool_type  = "ifp.InitFloorplan",
    getter     = "getFloorplan",
    var_name   = "floorplan",
    description= "Initialize the floorplan: die/core dimensions and row/track creation.",
    flow_stage = 1,
    keywords   = ["floorplan", "floorplanning", "initialize floorplan",
                  "die area", "core area", "aspect ratio", "utilization", "ifp"],
    default_sequence=[
        MethodSpec("initFloorplan", required=True,
                   description="Set die and core area from utilization or explicit rect"),
        MethodSpec("makeTracks",    required=True,
                   description="Create routing tracks after setting die/core dimensions"),
    ],
    measurements=[
        MeasurementSpec(
            label="core_area_um2",
            description="Core area in square microns",
            object_type="odb.dbBlock",
            causal_path=["openroad.Design", "odb.dbBlock"],
            filter_hint="get the core area rectangle and compute its area",
        ),
    ],
))


# ─── 2. IO Placement ──────────────────────────────────────────────────────────

_reg(FlowToolDef(
    tool_id    = "io_placement",
    tool_type  = "ppl.IOPlacer",
    getter     = "getIOPlacer",
    var_name   = "io_placer",
    description= "Place IO pins on the die boundary.",
    flow_stage = 2,
    requires_after=["floorplanning"],
    keywords   = ["io placement", "io placer", "pin placement", "place pins",
                  "io pins", "port placement", "ppl"],
    default_sequence=[
        MethodSpec("runIOPlacer", required=True,
                   description="Run IO pin placement"),
    ],
    measurements=[
        MeasurementSpec(
            label="placed_bterm_count",
            description="Number of placed IO pins (BTerms)",
            object_type="odb.dbBTerm",
            causal_path=["openroad.Design", "odb.dbBlock", "odb.dbBTerm"],
            filter_hint="BTerms (IO pins) that have been placed on the die boundary",
        ),
    ],
))


# ─── 3. Macro Placement ───────────────────────────────────────────────────────

_reg(FlowToolDef(
    tool_id    = "macro_placement",
    tool_type  = "mpl.MacroPlacer",
    getter     = "getMacroPlacer",
    var_name   = "macro_placer",
    description= "Place hard macros (RAM, IP blocks) before standard-cell placement.",
    flow_stage = 2,
    requires_after=["floorplanning"],
    keywords   = ["macro placement", "macro placer", "place macros",
                  "hard macros", "mpl"],
    default_sequence=[],
    exclusive_groups=[
        MutuallyExclusiveGroup(
            group_id    = "mpl_mode",
            options     = [["placeMacrosCornerMaxWl"], ["placeMacrosCornerMinWL"]],
            description = ("placeMacrosCornerMaxWl: push macros to corners maximising WL; "
                           "placeMacrosCornerMinWL: push macros to corners minimising WL"),
        ),
    ],
    measurements=[
        MeasurementSpec(
            label="placed_macro_count",
            description="Number of placed macro instances",
            object_type="odb.dbInst",
            causal_path=["openroad.Design", "odb.dbBlock", "odb.dbInst"],
            filter_hint="instances whose master cell is a macro block and are placed",
        ),
    ],
))


# ─── 4. Global Placement ──────────────────────────────────────────────────────

_reg(FlowToolDef(
    tool_id    = "global_placement",
    tool_type  = "gpl.Replace",
    getter     = "getReplace",
    var_name   = "placer",
    description= ("Global placement using Replace/Nesterov analytical engine. "
                  "doInitialPlace (quadratic) MUST be called before doNesterovPlace."),
    flow_stage = 2,
    requires_after=["floorplanning"],
    keywords   = ["global placement", "global place", "gpl", "replace",
                  "nesterov", "initial place", "analytical placement",
                  "place cells", "place all cells", "run placement"],
    default_sequence=[
        MethodSpec("doInitialPlace",  required=True,
                   description="Quadratic (initial) placement — MUST precede doNesterovPlace"),
        MethodSpec("doNesterovPlace", required=True,
                   description="Nesterov gradient refinement — MUST follow doInitialPlace"),
    ],
    exclusive_groups=[
        MutuallyExclusiveGroup(
            group_id    = "gpl_mode",
            options     = [
                ["doInitialPlace", "doNesterovPlace"],   # two-step analytical
                ["doIncrementalPlace"],                  # single-step incremental
            ],
            description = ("Use doInitialPlace+doNesterovPlace for full placement, "
                           "OR doIncrementalPlace for incremental — never mix."),
        ),
    ],
    measurements=[
        MeasurementSpec(
            label="unplaced_instance_count",
            description="Number of unplaced standard-cell instances",
            object_type="odb.dbInst",
            causal_path=["openroad.Design", "odb.dbBlock", "odb.dbInst"],
            filter_hint="instances that have not yet been placed (isPlaced() is False)",
        ),
        MeasurementSpec(
            label="placed_instance_count",
            description="Number of placed instances",
            object_type="odb.dbInst",
            causal_path=["openroad.Design", "odb.dbBlock", "odb.dbInst"],
            filter_hint="instances that have been placed (isPlaced() is True)",
        ),
    ],
))


# ─── 5. Detailed Placement (Legalization) ─────────────────────────────────────

_reg(FlowToolDef(
    tool_id    = "detailed_placement",
    tool_type  = "dpl.Opendp",
    getter     = "getOpendp",
    var_name   = "legalizer",
    description= "Detailed placement / legalization using OpenDP.",
    flow_stage = 2,
    requires_after=["global_placement"],
    keywords   = ["detailed placement", "legalization", "legalize", "opendp",
                  "dpl", "filler", "remove overlap"],
    default_sequence=[
        MethodSpec("detailedPlacement", required=True,
                   description="Run detailed placement / legalization"),
    ],
    measurements=[
        MeasurementSpec(
            label="placement_violations",
            description="Number of placement violations (overlaps)",
            object_type="odb.dbInst",
            causal_path=["openroad.Design", "odb.dbBlock", "odb.dbInst"],
            filter_hint="instances with placement overlap or legalization violations",
        ),
    ],
))


# ─── 6. Clock Tree Synthesis ──────────────────────────────────────────────────

_reg(FlowToolDef(
    tool_id    = "cts",
    tool_type  = "cts.TritonCTS",
    getter     = "getTritonCts",
    var_name   = "cts_engine",
    description= "Clock tree synthesis using TritonCTS.",
    flow_stage = 3,
    requires_after=["detailed_placement"],
    keywords   = ["cts", "clock tree synthesis", "clock tree", "tritoncts",
                  "triton cts", "clock buffer", "insert buffers"],
    default_sequence=[
        MethodSpec("runTritonCts", required=True,
                   description="Run clock tree synthesis"),
    ],
    measurements=[
        MeasurementSpec(
            label="instance_count",
            description="Total instance count after CTS (includes inserted clock buffers)",
            object_type="odb.dbInst",
            causal_path=["openroad.Design", "odb.dbBlock", "odb.dbInst"],
            filter_hint="all instances including newly inserted clock buffers",
        ),
    ],
))


# ─── 7. Global Routing ────────────────────────────────────────────────────────

_reg(FlowToolDef(
    tool_id    = "global_routing",
    tool_type  = "grt.GlobalRouter",
    getter     = "getGlobalRouter",
    var_name   = "global_router",
    description= "Global routing using TritonRoute global router.",
    flow_stage = 4,
    requires_after=["cts"],
    keywords   = ["global routing", "global route", "grt", "global router",
                  "route globally", "congestion"],
    default_sequence=[
        MethodSpec("globalRoute", required=True,
                   description="Run global routing"),
    ],
    measurements=[
        MeasurementSpec(
            label="net_count",
            description="Total number of nets routed",
            object_type="odb.dbNet",
            causal_path=["openroad.Design", "odb.dbBlock", "odb.dbNet"],
            filter_hint="all signal nets in the block",
        ),
    ],
))


# ─── 8. Detailed Routing ──────────────────────────────────────────────────────

_reg(FlowToolDef(
    tool_id    = "detailed_routing",
    tool_type  = "drt.TritonRoute",
    getter     = "getTritonRoute",
    var_name   = "detailed_router",
    description= "Detailed routing using TritonRoute.",
    flow_stage = 4,
    requires_after=["global_routing"],
    keywords   = ["detailed routing", "detailed route", "drt", "triton route",
                  "tritonroute", "drc", "route"],
    default_sequence=[
        MethodSpec("detailedRoute", required=True,
                   description="Run detailed routing (Python API entry point)"),
    ],
    measurements=[
        MeasurementSpec(
            label="drc_violation_count",
            description="Number of DRC violations after detailed routing",
            object_type="odb.dbNet",
            causal_path=["openroad.Design", "odb.dbBlock", "odb.dbNet"],
            filter_hint="nets with DRC violations after detailed routing",
        ),
    ],
))


# ─── Lookup helpers ───────────────────────────────────────────────────────────

def identify_tools(task: str) -> List[FlowToolDef]:
    """Return all FlowToolDefs whose keywords match the task string.

    Matches are case-insensitive.  Multiple tools can match for multi-tool tasks.
    Returns list sorted by flow_stage (earliest stage first).
    """
    task_lower = task.lower()
    matched = [
        t for t in TOOL_LIBRARY.values()
        if any(kw in task_lower for kw in t.keywords)
    ]
    return sorted(matched, key=lambda t: t.flow_stage)


def get_tool_by_type(type_name: str) -> Optional[FlowToolDef]:
    """Look up a tool by its Python type name (e.g. 'gpl.Replace')."""
    for t in TOOL_LIBRARY.values():
        if t.tool_type == type_name:
            return t
    return None


def get_tool_by_method(method_name: str) -> Optional[FlowToolDef]:
    """Return the FlowToolDef that owns a given method name."""
    for t in TOOL_LIBRARY.values():
        if method_name in t.all_known_methods():
            return t
    return None
