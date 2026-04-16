"""flow_task_decomposer.py — Decomposes a natural-language flow task into
an ordered Action Graph (the blue boxes in the causal chain diagram).

Each node in the graph is either:
  - tool_execution    : a flow tool that must be acquired and run
                        (e.g. global_placement, detailed_placement)
  - metric_computation: a measurement that must be computed from the DB
                        (e.g. unplaced instance count, HPWL, DRC violations)

The graph also captures whether the metric wraps the tool actions in a
sandwich (pre-measure → tools → post-measure) vs. just a final report.

Example outputs
---------------
Task: "Run Global Placement and report the change in unplaced instance count"
  → ActionGraph:
      Node 1: tool_execution    global_placement
      Node 2: metric_computation unplaced_instance_count
      sandwich: True  (report "change" → needs before AND after)

Task: "Run Global Placement, then Detailed Placement, report final HPWL"
  → ActionGraph:
      Node 1: tool_execution    global_placement
      Node 2: tool_execution    detailed_placement
      Node 3: metric_computation wirelength
      sandwich: False (report "final" → only after)
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from .flow_tool_library import TOOL_LIBRARY, FlowToolDef
except ImportError:
    from flow_tool_library import TOOL_LIBRARY, FlowToolDef          # type: ignore


# ── Action node ───────────────────────────────────────────────────────────────

@dataclass
class ActionNode:
    """One node in the Action Graph.

    Attributes
    ----------
    action_id   : 1-based position in the execution order.
    action_type : "tool_execution" | "metric_computation"
    tool_id     : key into TOOL_LIBRARY (only for tool_execution nodes).
    metric_label: human-readable metric name (only for metric_computation).
    description : original natural-language fragment for this node.
    """
    action_id:    int
    action_type:  str                   # "tool_execution" | "metric_computation"
    tool_id:      Optional[str] = None  # maps to TOOL_LIBRARY key
    metric_label: Optional[str] = None  # e.g. "unplaced_instance_count"
    description:  str = ""

    @property
    def tool_def(self) -> Optional[FlowToolDef]:
        """Return the FlowToolDef for this node, or None if metric node."""
        if self.tool_id:
            return TOOL_LIBRARY.get(self.tool_id)
        return None

    def is_tool(self) -> bool:
        return self.action_type == "tool_execution"

    def is_metric(self) -> bool:
        return self.action_type == "metric_computation"


# ── Action graph ──────────────────────────────────────────────────────────────

@dataclass
class ActionGraph:
    """Ordered sequence of ActionNodes for a flow task.

    Attributes
    ----------
    task       : Original natural-language task string.
    nodes      : Action nodes in execution order (1-based action_id).
    sandwich   : True when the metric node wraps the tool actions —
                 i.e. the task asks for "change" or "before/after",
                 meaning the metric must be measured BEFORE the first
                 tool AND AFTER the last tool.
                 False when the task only asks for a final report (after).
    """
    task:     str
    nodes:    List[ActionNode] = field(default_factory=list)
    sandwich: bool = False

    @property
    def tool_nodes(self) -> List[ActionNode]:
        """Return only tool_execution nodes in order."""
        return [n for n in self.nodes if n.is_tool()]

    @property
    def metric_nodes(self) -> List[ActionNode]:
        """Return only metric_computation nodes in order."""
        return [n for n in self.nodes if n.is_metric()]

    def ordering_summary(self) -> str:
        """Human-readable ordering e.g. 'GlobalPlacement → DetailedPlacement → HPWL'."""
        parts = []
        for n in self.nodes:
            if n.is_tool():
                parts.append(n.tool_id or n.description)
            else:
                parts.append(n.metric_label or "metric")
        return " → ".join(parts)

    def summary(self) -> str:
        sandwich_tag = "sandwich" if self.sandwich else "final-report"
        return (
            f"ActionGraph [{len(self.tool_nodes)} tool(s), "
            f"{len(self.metric_nodes)} metric(s), {sandwich_tag}]: "
            f"{self.ordering_summary()}"
        )


# ── LLM system prompt ─────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    tool_list = "\n".join(
        f'  "{t.tool_id}": {t.description} '
        f'(getter: design.{t.getter}(), type: {t.tool_type})'
        for t in sorted(TOOL_LIBRARY.values(), key=lambda x: x.flow_stage)
    )
    return f"""\
You are an OpenROAD EDA expert. Decompose a natural-language flow task into an
ordered Action Graph.

AVAILABLE TOOLS:
{tool_list}

OUTPUT FORMAT — return ONLY valid JSON, no markdown:
{{
  "nodes": [
    {{
      "action_id": 1,
      "action_type": "tool_execution",
      "tool_id": "<key from AVAILABLE TOOLS>",
      "description": "<fragment from the task>"
    }},
    {{
      "action_id": 2,
      "action_type": "metric_computation",
      "tool_id": null,
      "metric_label": "<short snake_case label e.g. unplaced_instance_count>",
      "description": "<fragment from the task>"
    }}
  ],
  "sandwich": true
}}

RULES:
1. List nodes in the EXACT execution order the task requires.
2. Use "tool_execution" for any EDA tool action (placement, routing, CTS, etc.).
3. Use "metric_computation" for any measurement/report step.
4. "sandwich" = true ONLY when the task asks for a CHANGE, DELTA, or BEFORE/AFTER comparison
   (keywords: "change", "transition", "before and after", "compare", "added",
   "removed", "how many", "were added", "were removed", "difference", "delta").
   "sandwich" = false when the task only asks for a final/post value.
5. tool_id MUST exactly match one of the keys in AVAILABLE TOOLS.
6. metric_label should be a concise snake_case identifier for what is measured.
"""


# ── TaskDecomposer ────────────────────────────────────────────────────────────

class TaskDecomposer:
    """Decomposes a natural-language task into an ActionGraph.

    Parameters
    ----------
    openai_key : str   OpenAI API key.
    model      : str   Model to use (default gpt-4.1-mini).
    """

    def __init__(self, openai_key: str, model: str = "gpt-4.1-mini"):
        self._api_key = openai_key
        self._model   = model
        self._system  = _build_system_prompt()

    def decompose(self, task: str) -> Optional[ActionGraph]:
        """Map *task* to an ActionGraph.

        Returns None if the task is not a flow task or the LLM call fails.
        """
        raw = self._call_llm(task)
        if not raw:
            return None
        return self._parse(task, raw)

    def is_flow_task(self, task: str) -> bool:
        """Quick keyword check — avoids an LLM call for obviously non-flow tasks."""
        task_lower = task.lower()
        return any(
            kw in task_lower
            for t in TOOL_LIBRARY.values()
            for kw in t.keywords
        )

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, task: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user",   "content": f"Task: {task}"},
            ],
            "temperature": 0,
            "max_tokens":  400,
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                return body["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 10 * (2 ** attempt)
                    print(f"    [TaskDecomposer rate-limit] waiting {wait}s...",
                          flush=True)
                    time.sleep(wait)
                else:
                    print(f"    [TaskDecomposer HTTP {e.code}]", flush=True)
                    return ""
            except Exception as exc:
                print(f"    [TaskDecomposer error] {exc}", flush=True)
                return ""
        return ""

    # ── JSON parse ────────────────────────────────────────────────────────────

    def _parse(self, task: str, text: str) -> Optional[ActionGraph]:
        # Strip markdown fences
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(
                ln for ln in text.splitlines()
                if not ln.strip().startswith("```")
            ).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            print(f"    [TaskDecomposer] JSON parse error: {exc}", flush=True)
            return None

        nodes: List[ActionNode] = []
        for raw_node in data.get("nodes", []):
            action_type = raw_node.get("action_type", "")
            tool_id     = raw_node.get("tool_id") or None
            # Validate tool_id against library
            if action_type == "tool_execution":
                if tool_id not in TOOL_LIBRARY:
                    print(
                        f"    [TaskDecomposer] unknown tool_id '{tool_id}' — skipping node",
                        flush=True,
                    )
                    continue

            nodes.append(ActionNode(
                action_id    = int(raw_node.get("action_id", len(nodes) + 1)),
                action_type  = action_type,
                tool_id      = tool_id,
                metric_label = raw_node.get("metric_label") or None,
                description  = str(raw_node.get("description", "")),
            ))

        if not nodes:
            return None

        return ActionGraph(
            task     = task,
            nodes    = nodes,
            sandwich = bool(data.get("sandwich", False)),
        )
