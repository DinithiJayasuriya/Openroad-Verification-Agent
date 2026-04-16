"""flow_multi_chain_extractor.py — Extracts an independent orange causal chain
for each ActionNode in an ActionGraph.

Two-level flow task pipeline:
  HIGH LEVEL (blue boxes)  : ActionGraph — tool1 → tool2 → metric
  LOW LEVEL  (orange nodes): one ActionChain per ActionNode

Pattern mirrors run_causal_agent.py:
  - No chain object is created.
  - Each path is just a List[str] of type names.
  - Edges are List[Tuple[str, str]] built from consecutive pairs.
  - NodeRetriever._lookup(src, tgt) is called per edge directly.

For tool_execution nodes:
  Path is fixed: ["openroad.Design", tool_def.tool_type]
  Edge API is known from FlowToolDef.getter — no lookup needed.
  Method sequence is known from FlowToolDef.default_sequence.

For metric_computation nodes:
  Path comes from MeasurementSpec.causal_path (looked up by label).
  Edges built from consecutive type pairs in the path.
  NodeRetriever._lookup(src, tgt) called per edge.
  Falls back to generic Design → Block path for unknown labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .flow_task_decomposer import ActionGraph, ActionNode
    from .flow_tool_library    import (
        TOOL_LIBRARY, FlowToolDef, MeasurementSpec, MutuallyExclusiveGroup,
    )
    from .node_retriever       import NodeRetriever, NodeAPIEntry, _normalize, _short
except ImportError:
    from flow_task_decomposer  import ActionGraph, ActionNode              # type: ignore
    from flow_tool_library     import (                                    # type: ignore
        TOOL_LIBRARY, FlowToolDef, MeasurementSpec, MutuallyExclusiveGroup,
    )
    from node_retriever        import NodeRetriever, NodeAPIEntry, _normalize, _short  # type: ignore


# ── Fallback metric label → causal_path table ─────────────────────────────────
# For metric labels the LLM may generate that don't appear verbatim in any
# MeasurementSpec.  Maps common snake_case labels to their acquisition paths.

_METRIC_PATH_TABLE: Dict[str, List[str]] = {
    # wirelength / HPWL
    "hpwl":                    ["openroad.Design", "odb.dbBlock"],
    "wirelength":              ["openroad.Design", "odb.dbBlock"],
    "half_perimeter_wirelength": ["openroad.Design", "odb.dbBlock"],
    "total_wirelength":        ["openroad.Design", "odb.dbBlock"],
    # instance-level
    "instance_count":          ["openroad.Design", "odb.dbBlock", "odb.dbInst"],
    "unplaced_instance_count": ["openroad.Design", "odb.dbBlock", "odb.dbInst"],
    "placed_instance_count":   ["openroad.Design", "odb.dbBlock", "odb.dbInst"],
    "placed_macro_count":      ["openroad.Design", "odb.dbBlock", "odb.dbInst"],
    "placement_violations":    ["openroad.Design", "odb.dbBlock", "odb.dbInst"],
    # port / bterm
    "placed_bterm_count":      ["openroad.Design", "odb.dbBlock", "odb.dbBTerm"],
    "io_pin_count":             ["openroad.Design", "odb.dbBlock", "odb.dbBTerm"],
    # net
    "net_count":               ["openroad.Design", "odb.dbBlock", "odb.dbNet"],
    "routed_net_count":        ["openroad.Design", "odb.dbBlock", "odb.dbNet"],
    "drc_violation_count":     ["openroad.Design", "odb.dbBlock", "odb.dbNet"],
    # area
    "core_area_um2":           ["openroad.Design", "odb.dbBlock"],
    "core_area":               ["openroad.Design", "odb.dbBlock"],
    # timing
    "timing":                  ["openroad.Design"],
    "slack":                   ["openroad.Design"],
    "wns":                     ["openroad.Design"],
    "tns":                     ["openroad.Design"],
}

# Descriptions to attach to unknown metric edges (improves prompt readability)
_METRIC_DESCRIPTION_TABLE: Dict[str, str] = {
    "hpwl":              "Compute HPWL (half-perimeter wirelength) from block",
    "wirelength":        "Compute total wirelength from block",
    "instance_count":    "Count all instances in the block",
    "unplaced_instance_count": "Count unplaced instances",
    "placed_instance_count":   "Count placed instances",
    "placed_macro_count":      "Count placed macro instances",
    "net_count":         "Count nets in the block",
    "drc_violation_count":     "Count DRC violations",
    "core_area_um2":     "Get core area in square microns",
}


# ── ActionChain ───────────────────────────────────────────────────────────────

@dataclass
class ActionChain:
    """Causal chain for one ActionNode in the ActionGraph.

    Attributes
    ----------
    action_node      : The ActionNode this chain corresponds to.
    chain_types      : Ordered type-name sequence (acquisition path).
    node_apis        : One NodeAPIEntry per edge (len = len(chain_types) - 1).
    method_sequence  : For tool nodes — ordered list of methods to call.
    exclusive_groups : For tool nodes — mutually exclusive method groups.
    measurement_spec : For metric nodes — the matched MeasurementSpec (or None).
    constraint_prompt: Ready-to-inject constraint section for this action.
    """
    action_node:       ActionNode
    chain_types:       List[str]
    node_apis:         List[NodeAPIEntry]
    method_sequence:   List[str]
    exclusive_groups:  List[MutuallyExclusiveGroup]
    measurement_spec:  Optional[MeasurementSpec]
    constraint_prompt: str

    @property
    def tool_def(self) -> Optional[FlowToolDef]:
        if self.action_node.tool_id:
            return TOOL_LIBRARY.get(self.action_node.tool_id)
        return None

    def short_chain(self) -> str:
        return " → ".join(_short(t) for t in self.chain_types)


# ── MultiActionChains ─────────────────────────────────────────────────────────

@dataclass
class MultiActionChains:
    """All per-action chains for a flow task, ready for code generation.

    Attributes
    ----------
    action_graph : The source ActionGraph (blue boxes).
    chains       : One ActionChain per ActionNode in execution order.
    sandwich     : True when metric must be measured before AND after tools.
    """
    action_graph: ActionGraph
    chains:       List[ActionChain]
    sandwich:     bool

    @property
    def tool_chains(self) -> List[ActionChain]:
        return [c for c in self.chains if c.action_node.is_tool()]

    @property
    def metric_chains(self) -> List[ActionChain]:
        return [c for c in self.chains if c.action_node.is_metric()]

    def to_full_constraint_prompt(self) -> str:
        """Build the complete constraint block for code generation."""
        lines = [
            "=== FLOW TASK CONSTRAINT BLOCK ===",
            f"Task : {self.action_graph.task}",
            f"Order: {self.action_graph.ordering_summary()}",
            "",
        ]

        sandwich_tag = ""
        if self.sandwich and self.metric_chains:
            m = self.metric_chains[0]
            label = m.action_node.metric_label or "metric"
            sandwich_tag = (
                f"SANDWICH: measure `{label}` BEFORE first tool AND AFTER last tool."
            )
            lines += [sandwich_tag, ""]

        for chain in self.chains:
            lines.append(chain.constraint_prompt)

        lines += [
            "GENERAL RULES:",
            "  • Each action acquires its objects independently from `design`.",
            "  • Do NOT skip or reorder actions.",
            "  • Do NOT call a method on the wrong object type.",
            "  • Follow the sandwich instruction above if present.",
            "===================================",
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        tools   = len(self.tool_chains)
        metrics = len(self.metric_chains)
        sw      = "sandwich" if self.sandwich else "final-report"
        order   = self.action_graph.ordering_summary()
        return f"MultiActionChains [{tools} tool(s), {metrics} metric(s), {sw}]: {order}"


# ── FlowMultiChainExtractor ───────────────────────────────────────────────────

class FlowMultiChainExtractor:
    """Extracts one independent causal chain per ActionNode.

    Parameters
    ----------
    rag_api_path : str
        Path to RAGAPIs.csv (used by NodeRetriever for metric edges).
    """

    def __init__(self, rag_api_path: str):
        self._retriever = NodeRetriever(rag_api_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, action_graph: ActionGraph) -> MultiActionChains:
        """Extract one ActionChain per node in *action_graph*.

        Returns a MultiActionChains ready to pass to the code generator.
        """
        chains: List[ActionChain] = []
        for node in action_graph.nodes:
            if node.is_tool():
                chains.append(self._extract_tool_chain(node))
            else:
                chains.append(self._extract_metric_chain(node, action_graph))

        return MultiActionChains(
            action_graph=action_graph,
            chains=chains,
            sandwich=action_graph.sandwich,
        )

    # ── Tool chain ────────────────────────────────────────────────────────────

    def _extract_tool_chain(self, node: ActionNode) -> ActionChain:
        """Chain for a tool_execution ActionNode.

        The acquisition path (Design → tool_type) and method sequence are
        fully determined by FlowToolDef — no RAG lookup needed.
        """
        tool_def = TOOL_LIBRARY[node.tool_id]
        chain_types = ["openroad.Design", tool_def.tool_type]

        # Edge API: design.<getter>() returns tool_def.tool_type
        api_entry = NodeAPIEntry(
            source_type    = "openroad.Design",
            target_type    = tool_def.tool_type,
            method_name    = tool_def.getter,
            full_signature = f"openroad.Design.{tool_def.getter}(",
            params         = "",
            description    = f"Get {tool_def.tool_type} from design",
            is_list        = False,
            source         = "library",
        )

        constraint_prompt = self._build_tool_constraint(
            action_id=node.action_id,
            tool_def=tool_def,
            api_entry=api_entry,
        )

        return ActionChain(
            action_node      = node,
            chain_types      = chain_types,
            node_apis        = [api_entry],
            method_sequence  = tool_def.all_required_methods(),
            exclusive_groups = tool_def.exclusive_groups,
            measurement_spec = None,
            constraint_prompt= constraint_prompt,
        )

    def _build_tool_constraint(
        self,
        action_id: int,
        tool_def:  FlowToolDef,
        api_entry: NodeAPIEntry,
    ) -> str:
        lines = [
            f"--- Action {action_id}: {tool_def.tool_id} ({tool_def.tool_type}) ---",
            f"  Acquire : {tool_def.var_name} = design.{tool_def.getter}()"
            f"  # returns {tool_def.tool_type}",
        ]

        if tool_def.all_required_methods():
            lines.append("  Call in order:")
            for method in tool_def.all_required_methods():
                spec_desc = next(
                    (m.description for m in tool_def.default_sequence if m.name == method),
                    "",
                )
                comment = f"  # {spec_desc}" if spec_desc else ""
                lines.append(f"    {tool_def.var_name}.{method}(){comment}")

        if tool_def.exclusive_groups:
            for grp in tool_def.exclusive_groups:
                lines.append(f"  NOTE: {grp.description}")

        lines.append("")
        return "\n".join(lines)

    # ── Metric chain ──────────────────────────────────────────────────────────

    def _extract_metric_chain(
        self, node: ActionNode, action_graph: ActionGraph
    ) -> ActionChain:
        """Chain for a metric_computation ActionNode.

        Looks up MeasurementSpec by label (tool measurements → fallback table).
        Falls back to generic Design → Block chain for unknown labels.
        """
        label = node.metric_label or ""
        spec  = self._find_spec(label, action_graph)
        path  = spec.causal_path if spec else self._fallback_path(label)

        # Build edge list and look up each edge directly — same pattern as
        # run_causal_agent.py iterating state.all_edges with _rag_query_for_edge.
        edges     = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        node_apis = [
            self._retriever._lookup(_normalize(src), _normalize(tgt))
            for src, tgt in edges
        ]

        constraint_prompt = self._build_metric_constraint(
            action_id = node.action_id,
            label     = label,
            path      = path,
            node_apis = node_apis,
            spec      = spec,
        )

        return ActionChain(
            action_node      = node,
            chain_types      = path,
            node_apis        = node_apis,
            method_sequence  = [],
            exclusive_groups = [],
            measurement_spec = spec,
            constraint_prompt= constraint_prompt,
        )

    def _find_spec(
        self, label: str, action_graph: ActionGraph
    ) -> Optional[MeasurementSpec]:
        """Find the MeasurementSpec for *label*.

        Search order:
          1. Measurements of tool nodes in this ActionGraph (most relevant)
          2. All tools in TOOL_LIBRARY
        """
        # Step 1: tools in this task
        tool_ids = [n.tool_id for n in action_graph.nodes if n.is_tool() and n.tool_id]
        for tid in tool_ids:
            tool_def = TOOL_LIBRARY.get(tid)
            if not tool_def:
                continue
            for s in tool_def.measurements:
                if s.label == label:
                    return s

        # Step 2: global scan
        for tool_def in TOOL_LIBRARY.values():
            for s in tool_def.measurements:
                if s.label == label:
                    return s

        return None

    def _fallback_path(self, label: str) -> List[str]:
        """Return a causal path for *label* using the fallback table or Design→Block."""
        # Exact match
        if label in _METRIC_PATH_TABLE:
            return _METRIC_PATH_TABLE[label]
        # Partial match (label contains a known keyword)
        for key, path in _METRIC_PATH_TABLE.items():
            if key in label or label in key:
                return path
        # Generic fallback
        return ["openroad.Design", "odb.dbBlock"]

    def _build_metric_constraint(
        self,
        action_id: int,
        label:     str,
        path:      List[str],
        node_apis: List[NodeAPIEntry],
        spec:      Optional[MeasurementSpec],
    ) -> str:
        if spec:
            desc = spec.description
            hint = spec.filter_hint
        else:
            desc = _METRIC_DESCRIPTION_TABLE.get(label, label.replace("_", " "))
            hint = ""

        lines = [
            f"--- Action {action_id}: metric_computation ({label}) ---",
            f"  Measure : {desc}",
            f"  Acquisition path: {' → '.join(_short(t) for t in path)}",
            "  Steps:",
        ]

        for i, api in enumerate(node_apis):
            src = _short(api.source_type)
            tgt = _short(api.target_type)
            if api.method_name.startswith("<UNKNOWN"):
                lines.append(
                    f"    Step {i+1}: {tgt} = ???  "
                    f"# [UNKNOWN] find method on {src} that returns {tgt}"
                )
            else:
                call = (f"{src.lower()}.{api.method_name}({api.params.strip()})"
                        if api.params.strip()
                        else f"{src.lower()}.{api.method_name}()")
                lines.append(f"    Step {i+1}: {tgt.lower()} = {call}")
                if api.description:
                    lines.append(f"             # {api.description}")

        if hint:
            lines.append(f"  Filter  : {hint}")

        lines.append("")
        return "\n".join(lines)


