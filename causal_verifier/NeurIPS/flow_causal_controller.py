"""flow_causal_controller.py — LLM controller for the flow causal agent loop.

Mirrors controller.py exactly in structure:
  - Reads FlowCausalAgentState
  - Returns FlowControllerDecision
  - Pre-LLM rule overrides for loop detection
  - LLM call with flow-specific system prompt
  - Rule-based fallback when LLM fails

Flow-specific actions (simpler than the original):
  re_generate     — regenerate code with L4a repair hint injected
  commit_best     — commit best checkpoint and exit
  stop_fail       — give up (budget=0, no passing code)

No re_retrieve_edge or re_extract_chain — the constraint prompt is fixed
from the tool library, so there is nothing to re-retrieve or re-extract.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

try:
    from .flow_causal_state import FlowCausalAgentState
except ImportError:
    from flow_causal_state import FlowCausalAgentState             # type: ignore


# ── Decision dataclass ────────────────────────────────────────────────────────

@dataclass
class FlowControllerDecision:
    diagnosis:      str    # root-cause reasoning
    next_action:    str    # re_generate | re_retrieve_edge | commit_best | stop_fail
    repair_hint:    str    # injected into next generation prompt
    updated_lesson: str    # written to state.lessons before execution
    target_edge:    str  = ""   # for re_retrieve_edge: "<src_short> → <tgt_short>"
    from_fallback:  bool = False


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the Flow Causal Arbiter for an EDA AI Agent.
Your job is to decide the single best next action after verifiers report issues
with the generated OpenROAD Python code.

TWO VERIFIERS are active:

1. STATIC VERIFIER (L1-L3) — checks syntax and API correctness:
   L1 — Syntax error          : code cannot be parsed.
   L2 — Missing DB prerequisite: design.getBlock() or other acquisition calls absent.
   L3 — API hallucination      : code calls a method not in the OpenROAD API
                                 (the api_diffs field lists hallucinated vs. RAG-suggested methods).

2. L4a ACTION ORDERING VERIFIER (A1-A4) — checks tool execution order:
   A1 — Getter present     : design.<getter>() must appear for every tool in the graph.
   A2 — Cross-tool ordering: last method of tool_i must appear before first getter of tool_{i+1}.
   A3 — Sandwich structure : pre_<metric> before first tool, post_<metric> after last tool.
   A4 — Mode integrity     : no mixing of mutually exclusive method groups within one tool.

=== AVAILABLE ACTIONS ===

re_generate
  Re-generate code using the same constraint prompt, injecting repair_hint.
  Use when: L1/L2/A1/A2/A3/A4 issues AND the repair is about logic or ordering.
  repair_hint MUST name the exact issue codes and the fix required.

re_retrieve_edge
  Re-run RAG retrieval for one specific metric chain edge to find the correct API method.
  Use when: L3 (API hallucination) on a METRIC computation edge (not a tool getter).
  target_edge  = "<SourceType_short> → <TargetType_short>" (e.g. "Design → dbBlock").
  repair_hint  = description of the hallucinated method and what kind of method is needed.
  Automatically re-generates after retrieving, costing only 1 budget unit.

commit_best
  Commit the best checkpoint found so far and exit.
  Use when: both verifiers PASS, OR budget = 1 and best checkpoint is reasonable.

stop_fail
  Give up. Use only when: budget = 0 and no passing code was found.

=== DECISION RULES ===

L1 (syntax error):
  → re_generate; repair_hint = "Fix syntax error: <error_message>."

L2 (missing acquisition):
  → re_generate; repair_hint = "Add the missing acquisition call (e.g. design.getBlock())
    before accessing database objects."

L3 (API hallucination on metric edge):
  → re_retrieve_edge; target_edge = "<src> → <tgt>"; repair_hint = "Hallucinated method: <method>.
    Find the correct method that transitions from <src> to <tgt>."

L3 (API hallucination on tool execution code):
  → re_generate; repair_hint = "Method <method> does not exist on <type>. Use <rag_suggestion> instead."

A1 (getter missing):
  → re_generate; repair_hint = "Add design.<getter>() to acquire the tool before calling methods."

A2 (wrong action order):
  → re_generate; repair_hint = "Move all <tool_j> code to AFTER the last method of <tool_i>."

A3 (sandwich missing/wrong):
  → re_generate; repair_hint = "Add pre_<label>=<measure> before first tool, post_<label>=<measure> after last tool."

A4 (mode mixing):
  → re_generate; repair_hint = "Choose exactly ONE mode for <group_id> — remove all methods from the other mode."

Priority when multiple issues:
  L1 > L2 > L3 > A1 > A2 > A3 > A4.
  List ALL issues in repair_hint, most critical first.

Repeating same check after 2+ attempts:
  → Include a stronger warning: "You made this error before — do not repeat it."

Budget = 1:
  → commit_best unless there is exactly one clear fix remaining.

Budget = 0:
  → stop_fail.

=== OUTPUT FORMAT ===
Output ONLY valid JSON — no explanation, no markdown:
{
  "diagnosis":      "concise root-cause referencing specific check codes (L1/L2/L3/A1/A2/A3/A4)",
  "next_action":    "re_generate | re_retrieve_edge | commit_best | stop_fail",
  "target_edge":    "<SourceType> → <TargetType>  (only for re_retrieve_edge, else empty string)",
  "repair_hint":    "specific instruction for the code generator",
  "updated_lesson": "one lesson to remember for future steps in this run"
}
"""


# ── FlowCausalController ──────────────────────────────────────────────────────

class FlowCausalController:
    """LLM-powered controller for flow task agent loop.

    Parameters
    ----------
    api_key : str   OpenAI key
    model   : str   Model to use (default gpt-4.1-mini)
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.api_key = api_key
        self.model   = model

    def decide(self, state: FlowCausalAgentState) -> FlowControllerDecision:
        """Read state → return decision. Falls back to rules if LLM call fails."""

        # ── Pre-LLM override: repeating same check ────────────────────────────
        repeat_override = self._check_repeat_loop(state)
        if repeat_override:
            return repeat_override

        user_msg = state.to_controller_string()
        text     = self._call_llm(user_msg)
        if text:
            decision = self._parse_decision(text)
            if decision:
                return decision

        return self._rule_fallback(state)

    # ── Loop detection ────────────────────────────────────────────────────────

    def _check_repeat_loop(
        self, state: FlowCausalAgentState
    ) -> Optional[FlowControllerDecision]:
        """If the same L4a check has failed 3+ consecutive times, strengthen the hint."""
        snap = state.l4a_result
        if snap is None or snap.passed or not snap.issue_checks:
            return None

        non_boot = [o for o in state.observations if not o.is_bootstrap]
        if len(non_boot) < 3:
            return None

        # Count how many consecutive non-boot steps reported the same check failing
        current_checks = set(snap.issue_checks)
        consecutive = 0
        for obs in reversed(non_boot):
            # Observation result contains check codes e.g. "FAIL checks=['A2']"
            if any(c in obs.result for c in current_checks):
                consecutive += 1
            else:
                break

        if consecutive < 3:
            return None

        hint = (
            f"CRITICAL: The following L4a checks have failed {consecutive} times in a row: "
            f"{snap.issue_checks}.\n"
            f"You MUST fix these before anything else.\n\n"
            f"{snap.feedback}"
        )

        print(
            f"  [flow-controller] REPEAT LOOP: checks {snap.issue_checks} "
            f"failed {consecutive} consecutive times — strengthening hint",
            flush=True,
        )

        return FlowControllerDecision(
            diagnosis=(
                f"L4a checks {snap.issue_checks} have repeated {consecutive} times. "
                f"Forcing re_generate with an explicit critical warning."
            ),
            next_action    = "re_generate",
            repair_hint    = hint,
            updated_lesson = (
                f"Checks {snap.issue_checks} keep failing — pay close attention to "
                f"the action ordering and sandwich structure."
            ),
            from_fallback  = True,
        )

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, user_msg: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0,
            "max_tokens":  400,
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
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
                    print(f"    [flow-controller rate-limit] waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    [flow-controller HTTP {e.code}]", flush=True)
                    return ""
            except Exception as exc:
                print(f"    [flow-controller error] {exc}", flush=True)
                return ""
        return ""

    # ── JSON parse ────────────────────────────────────────────────────────────

    def _parse_decision(self, text: str) -> Optional[FlowControllerDecision]:
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.splitlines() if not l.strip().startswith("```")
            ).strip()
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return None

        action = d.get("next_action", "")
        if action not in ("re_generate", "re_retrieve_edge", "commit_best", "stop_fail"):
            return None

        return FlowControllerDecision(
            diagnosis      = str(d.get("diagnosis",      "")),
            next_action    = action,
            repair_hint    = str(d.get("repair_hint",    "")),
            updated_lesson = str(d.get("updated_lesson", "")),
            target_edge    = str(d.get("target_edge",    "")),
            from_fallback  = False,
        )

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_fallback(
        self, state: FlowCausalAgentState
    ) -> FlowControllerDecision:
        l4a  = state.l4a_result
        sv   = state.static_result

        # Both verifiers pass → commit
        l4a_ok = l4a is None or l4a.passed
        sv_ok  = sv is None or sv.passed
        if l4a_ok and sv_ok:
            return FlowControllerDecision(
                diagnosis     = "Both verifiers passed — committing.",
                next_action   = "commit_best",
                repair_hint   = "",
                updated_lesson= "",
                from_fallback = True,
            )

        if state.budget_remaining == 0:
            return FlowControllerDecision(
                diagnosis     = "Budget exhausted, no passing code.",
                next_action   = "stop_fail",
                repair_hint   = "",
                updated_lesson= "",
                from_fallback = True,
            )

        if state.budget_remaining <= 1:
            return FlowControllerDecision(
                diagnosis     = "Budget nearly exhausted — committing best.",
                next_action   = "commit_best",
                repair_hint   = "",
                updated_lesson= "",
                from_fallback = True,
            )

        # Static verifier failed at L3 (API hallucination) and there are metric edges
        if sv and not sv.passed and sv.layer_failed == 3 and state.all_edges:
            return FlowControllerDecision(
                diagnosis     = f"L3 API hallucination — re-retrieving first metric edge.",
                next_action   = "re_retrieve_edge",
                target_edge   = (f"{state.all_edges[0][0].split('.')[-1]} → "
                                 f"{state.all_edges[0][1].split('.')[-1]}"),
                repair_hint   = sv.feedback[:400] if sv.feedback else "",
                updated_lesson= "",
                from_fallback = True,
            )

        # Static verifier failed at L1/L2 — re-generate
        if sv and not sv.passed:
            return FlowControllerDecision(
                diagnosis     = f"Static verifier FAIL(L{sv.layer_failed}) — re-generating.",
                next_action   = "re_generate",
                repair_hint   = sv.feedback[:400] if sv.feedback else "",
                updated_lesson= "",
                from_fallback = True,
            )

        # L4a failed — re-generate
        return FlowControllerDecision(
            diagnosis     = f"L4a FAIL checks={l4a.issue_checks if l4a else '?'} — re-generating.",
            next_action   = "re_generate",
            repair_hint   = l4a.feedback[:600] if (l4a and l4a.feedback) else "",
            updated_lesson= "",
            from_fallback = True,
        )
