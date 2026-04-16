"""flow_chain_extractor.py — Builds a MultiPathCausalGraph from a task string.

For a Level 2.1 single-tool flow task the extractor answers four questions:

  1. Which tool?         keyword match via identify_tools()  (rule-based)
  2. Which measurement?  keyword match → LLM fallback        (hybrid)
  3. Which methods?      tool library default sequence        (rule-based)
  4. Sandwich needed?    heuristic on task keywords           (rule-based)

The LLM is called ONLY when keyword matching cannot disambiguate which
MeasurementSpec the task is asking about.

Returns None if the task is not recognised as a flow task (no tool match),
so the existing CausalPipeline can handle it unchanged.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import List, Optional

try:
    from .flow_tool_library import (
        FlowToolDef, MeasurementSpec, TOOL_LIBRARY, identify_tools,
    )
    from .flow_causal_graph import MultiPathCausalGraph, ObjectPath, ActionPath
except ImportError:
    from flow_tool_library import (                          # type: ignore
        FlowToolDef, MeasurementSpec, TOOL_LIBRARY, identify_tools,
    )
    from flow_causal_graph import MultiPathCausalGraph, ObjectPath, ActionPath  # type: ignore


# ── Sandwich detection keywords ───────────────────────────────────────────────

_SANDWICH_KEYWORDS = [
    "change", "report", "before and after", "before/after",
    "compare", "difference", "delta", "how many", "count",
    "report the change", "measure", "track",
]


# ── Measurement keyword table ─────────────────────────────────────────────────
# Maps task keywords → MeasurementSpec.label they suggest.
# Used for fast rule-based measurement selection before falling back to LLM.

_MEASUREMENT_KEYWORDS: dict = {
    "unplaced":              "unplaced_instance_count",
    "not placed":            "unplaced_instance_count",
    "placed instance":       "placed_instance_count",
    "placed cell":           "placed_instance_count",
    "instance count":        "unplaced_instance_count",   # default for placement tasks
    "cell count":            "placed_instance_count",
    "macro":                 "placed_macro_count",
    "hard macro":            "placed_macro_count",
    "io pin":                "placed_bterm_count",
    "port":                  "placed_bterm_count",
    "bterm":                 "placed_bterm_count",
    "core area":             "core_area_um2",
    "drc":                   "drc_violation_count",
    "violation":             "drc_violation_count",
    "clock buffer":          "instance_count",
    "buffer":                "instance_count",
    "net count":             "net_count",
    "congestion":            "net_count",
}


# ── FlowChainExtractor ────────────────────────────────────────────────────────

class FlowChainExtractor:
    """Produces a MultiPathCausalGraph from a task string.

    Parameters
    ----------
    openai_key : str, optional
        If provided, enables LLM fallback for ambiguous measurement selection.
    model : str
        OpenAI model for measurement disambiguation (default gpt-4.1-mini).
    """

    def __init__(
        self,
        openai_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
    ):
        self._api_key = openai_key
        self._model   = model

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, task: str) -> Optional[MultiPathCausalGraph]:
        """Build a MultiPathCausalGraph for *task*.

        Returns None if the task is not recognised as a flow task, so the
        caller can fall back to the existing CausalPipeline.
        """
        # Step 1: identify tool
        matched = identify_tools(task)
        if not matched:
            return None
        tool_def = matched[0]   # Level 2.1: single tool

        # Step 2: select measurement spec
        measurement = self._select_measurement(task, tool_def)
        if measurement is None:
            # No measurement found — still build graph without sandwich
            measurement = tool_def.measurements[0] if tool_def.measurements else None
        if measurement is None:
            return None   # tool has no measurement specs at all

        # Step 3: detect sandwich
        has_sandwich = self._detect_sandwich(task)

        # Step 4: assemble graph
        pre_var  = f"pre_{measurement.label}"
        post_var = f"post_{measurement.label}"

        object_path = ObjectPath(
            nodes       = measurement.causal_path,
            measurement = measurement,
            pre_var     = pre_var,
            post_var    = post_var,
        )
        action_path = ActionPath(
            tool_def        = tool_def,
            tool_var        = tool_def.var_name,
            method_sequence = tool_def.all_required_methods(),
        )
        return MultiPathCausalGraph(
            task         = task,
            object_path  = object_path,
            action_path  = action_path,
            has_sandwich = has_sandwich,
        )

    def is_flow_task(self, task: str) -> bool:
        """Quick check: True if *task* contains a recognised flow tool keyword."""
        return len(identify_tools(task)) > 0

    # ── Step 2: measurement selection ─────────────────────────────────────────

    def _select_measurement(
        self, task: str, tool_def: FlowToolDef
    ) -> Optional[MeasurementSpec]:
        """Pick the MeasurementSpec that best matches the task.

        Priority:
          1. Keyword match against _MEASUREMENT_KEYWORDS table
          2. Keyword match against spec labels and descriptions directly
          3. LLM disambiguation (if api_key provided and ambiguous)
          4. Fallback: first spec in tool_def.measurements
        """
        if not tool_def.measurements:
            return None

        # Fast path: single spec — no disambiguation needed
        if len(tool_def.measurements) == 1:
            return tool_def.measurements[0]

        task_lower = task.lower()

        # ── Pass 1: keyword table match ───────────────────────────────────────
        for keyword, label in _MEASUREMENT_KEYWORDS.items():
            if keyword in task_lower:
                for spec in tool_def.measurements:
                    if spec.label == label:
                        return spec

        # ── Pass 2: direct label / description / filter_hint match ───────────
        for spec in tool_def.measurements:
            label_words    = set(spec.label.replace("_", " ").lower().split())
            desc_words     = set(spec.description.lower().split())
            hint_words     = set(spec.filter_hint.lower().split())
            task_words     = set(task_lower.split())
            overlap_score  = len(task_words & (label_words | desc_words | hint_words))
            spec._score    = overlap_score  # temp attribute for sorting

        best = max(tool_def.measurements, key=lambda s: getattr(s, "_score", 0))
        # Clean up temp attribute
        for spec in tool_def.measurements:
            if hasattr(spec, "_score"):
                del spec._score

        # Only accept if there's a non-zero overlap
        if getattr(best, "_score", 0) > 0:
            return best

        # ── Pass 3: LLM disambiguation ────────────────────────────────────────
        if self._api_key and len(tool_def.measurements) > 1:
            llm_choice = self._llm_select_measurement(task, tool_def)
            if llm_choice:
                return llm_choice

        # ── Fallback: first spec ───────────────────────────────────────────────
        return tool_def.measurements[0]

    # ── Step 3: sandwich detection ────────────────────────────────────────────

    def _detect_sandwich(self, task: str) -> bool:
        """Return True if the task asks for a before/after measurement report."""
        task_lower = task.lower()
        return any(kw in task_lower for kw in _SANDWICH_KEYWORDS)

    # ── LLM fallback for measurement disambiguation ───────────────────────────

    def _llm_select_measurement(
        self, task: str, tool_def: FlowToolDef
    ) -> Optional[MeasurementSpec]:
        """Ask the LLM to choose among available MeasurementSpecs.

        Returns None on any API failure (caller falls back to first spec).
        """
        specs_text = "\n".join(
            f'  "{s.label}": {s.description} (filter: {s.filter_hint})'
            for s in tool_def.measurements
        )
        prompt = (
            f"Task: {task}\n\n"
            f"Available measurement specs for tool '{tool_def.tool_id}':\n"
            f"{specs_text}\n\n"
            f"Which spec label best matches what the task asks to measure?\n"
            f'Output ONLY the label string (e.g. "unplaced_instance_count"). No explanation.'
        )
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 30,
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
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body  = json.loads(resp.read().decode())
                label = body["choices"][0]["message"]["content"].strip().strip('"')
                for spec in tool_def.measurements:
                    if spec.label == label:
                        return spec
                return None
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(10 * (2 ** attempt))
                else:
                    return None
            except Exception:
                return None
        return None
