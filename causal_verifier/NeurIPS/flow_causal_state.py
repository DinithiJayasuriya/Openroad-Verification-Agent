"""flow_causal_state.py — Mutable runtime state for the flow causal agent loop.

Mirrors causal_state.py but carries flow-task specific fields:
  action_graph  : ActionGraph (blue boxes) from TaskDecomposer
  multi_chains  : MultiActionChains (orange nodes) from FlowMultiChainExtractor
  l4a_result    : L4SequencingResult from FlowL4aVerifier

Bootstrap steps (free, no budget cost):
  1. flow_decompose  — task → ActionGraph  (TaskDecomposer LLM call)
  2. flow_extract    — ActionGraph → MultiActionChains  (FlowMultiChainExtractor)
  3. flow_generate   — MultiActionChains → code  (LLM with constraint prompt)
  4. flow_l4a_verify — code → L4SequencingResult  (FlowL4aVerifier AST check)

Controller loop (budgeted):
  FlowCausalController.decide(state) → FlowControllerDecision
  FlowCausalDispatcher.dispatch(state, decision) → execute action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from .flow_task_decomposer      import ActionGraph
    from .flow_multi_chain_extractor import MultiActionChains
    from .flow_l4a_verifier          import L4SequencingResult
except ImportError:
    from flow_task_decomposer       import ActionGraph              # type: ignore
    from flow_multi_chain_extractor import MultiActionChains        # type: ignore
    from flow_l4a_verifier          import L4SequencingResult       # type: ignore

# VerifierSnapshot reused from causal_state (same interface)
try:
    from .causal_state import VerifierSnapshot
except ImportError:
    from causal_state import VerifierSnapshot                       # type: ignore


# ── Single action-observation record ─────────────────────────────────────────

@dataclass
class FlowToolObservation:
    step:         int
    action:       str    # which action was executed
    result:       str    # one-line summary
    detail:       str  = ""
    is_bootstrap: bool = False


# ── L4a verifier snapshot ─────────────────────────────────────────────────────

@dataclass
class L4aSnapshot:
    """Compact record of one L4a verification run."""
    passed:      bool
    issues:      List[str]   # [check: description] strings
    feedback:    str
    issue_checks: List[str]  # just the check codes e.g. ["A1", "A2"]

    @classmethod
    def from_result(cls, result: L4SequencingResult) -> "L4aSnapshot":
        return cls(
            passed       = result.passed,
            issues       = [f"{i.check}: {i.description}" for i in result.issues],
            feedback     = result.feedback,
            issue_checks = [i.check for i in result.issues],
        )


# ── Main state object ─────────────────────────────────────────────────────────

@dataclass
class FlowCausalAgentState:
    """Runtime state for one flow task in the agent loop.

    The bootstrap populates the flow fields (action_graph, multi_chains).
    The controller reads everything via to_controller_string().
    """
    task:       str
    max_budget: int

    # ── step / budget counters ─────────────────────────────────────────────────
    step:        int = 0
    budget_used: int = 0

    # ── flow-specific fields (filled by bootstrap) ────────────────────────────
    action_graph:  Optional[ActionGraph]       = None
    multi_chains:  Optional[MultiActionChains] = None

    # ── code ───────────────────────────────────────────────────────────────────
    current_code: str       = ""
    code_history: List[str] = field(default_factory=list)

    # best checkpoint
    best_code:  str   = ""
    best_score: float = -1.0
    best_step:  int   = -1
    best_snap:  Optional[L4aSnapshot] = None

    # ── verifier state (latest) ───────────────────────────────────────────────
    static_result: Optional[VerifierSnapshot] = None   # L1-L3 from OpenROADStaticVerifier
    l4a_result:    Optional[L4aSnapshot]      = None   # A1-A4 from FlowL4aVerifier

    # ── metric edges for re_retrieve_edge ─────────────────────────────────────
    # Only metric chain edges are tracked here — tool edges are fixed from library
    all_edges: List[Tuple[str, str]] = field(default_factory=list)
    # one RAG hit dict per metric edge (or None if not found)
    edge_apis: List[Optional[dict]]  = field(default_factory=list)

    # ── accumulated knowledge ─────────────────────────────────────────────────
    lessons:      List[str]               = field(default_factory=list)
    observations: List[FlowToolObservation] = field(default_factory=list)

    # ── commit flag ────────────────────────────────────────────────────────────
    committed:      bool = False
    committed_code: str  = ""

    # ── budget ────────────────────────────────────────────────────────────────
    @property
    def budget_remaining(self) -> int:
        return self.max_budget - self.budget_used

    # ── observation recording ─────────────────────────────────────────────────

    def add_bootstrap_obs(self, action: str, result: str, detail: str = "") -> None:
        self.observations.append(FlowToolObservation(
            step=self.step, action=action, result=result,
            detail=detail, is_bootstrap=True,
        ))
        self.step += 1

    def add_observation(self, action: str, result: str, detail: str = "") -> None:
        self.observations.append(FlowToolObservation(
            step=self.step, action=action, result=result,
            detail=detail, is_bootstrap=False,
        ))
        self.step        += 1
        self.budget_used += 1

    def add_lesson(self, lesson: str) -> None:
        lesson = lesson.strip()
        if lesson and lesson not in self.lessons:
            self.lessons.append(lesson)

    # ── best checkpoint ────────────────────────────────────────────────────────

    def maybe_update_best(self) -> bool:
        if not self.current_code:
            return False
        score = _score_snap(self.l4a_result)
        if score > self.best_score:
            self.best_code  = self.current_code
            self.best_score = score
            self.best_step  = self.step
            self.best_snap  = self.l4a_result
            return True
        return False

    # ── controller context string ─────────────────────────────────────────────

    def to_controller_string(self) -> str:
        lines = [
            f"TASK: {self.task}",
            f"STEP: {self.step}  BUDGET_REMAINING: {self.budget_remaining}/{self.max_budget}",
            "",
        ]

        # Action graph summary
        if self.action_graph:
            lines += [
                "ACTION GRAPH (blue boxes):",
                f"  Order    : {self.action_graph.ordering_summary()}",
                f"  Sandwich : {self.action_graph.sandwich}",
                "",
            ]
        else:
            lines += ["ACTION GRAPH: not extracted", ""]

        # Multi-chain summary
        if self.multi_chains:
            lines += [
                "MULTI-CHAIN CONSTRAINT PROMPT (injected into generator):",
            ]
            for ln in self.multi_chains.to_full_constraint_prompt().splitlines():
                lines.append(f"  {ln}")
            lines.append("")
        else:
            lines += ["MULTI-CHAIN: not extracted", ""]

        # Action-observation history
        lines.append("ACTION-OBSERVATION HISTORY:")
        if not self.observations:
            lines.append("  (none)")
        for obs in self.observations:
            tag = "[boot]" if obs.is_bootstrap else f"[{obs.step}]"
            lines.append(f"  {tag} {obs.action:30s} → {obs.result}")
        lines.append("")

        # Current code preview
        if self.current_code:
            preview = self.current_code[:300].replace("\n", "↵")
            lines.append(f"CURRENT CODE ({len(self.current_code)} chars): {preview}...")
        else:
            lines.append("CURRENT CODE: none")
        lines.append("")

        # Static verifier (L1-L3)
        lines.append("LATEST STATIC VERIFIER (L1-L3):")
        if self.static_result:
            sv = self.static_result
            s  = "PASS" if sv.passed else f"FAIL(L{sv.layer_failed})"
            lines.append(f"  {s}  issues={sv.issues[:2]}")
            if not sv.passed and sv.feedback:
                for ln in sv.feedback.splitlines()[:6]:
                    lines.append(f"  {ln}")
        else:
            lines.append("  not run yet")
        lines.append("")

        # Metric edges for re_retrieve_edge context
        if self.all_edges:
            lines.append("METRIC CHAIN EDGES (re_retrieve_edge targets):")
            for i, (src, tgt) in enumerate(self.all_edges):
                api = self.edge_apis[i] if i < len(self.edge_apis) else None
                method = api["function_name"].split("(")[0].split(".")[-1] if api else "UNKNOWN"
                lines.append(f"  [{i}] {src.split('.')[-1]} → {tgt.split('.')[-1]} via {method}()")
            lines.append("")

        # L4a verifier result
        lines.append("LATEST L4a VERIFIER (A1-A4):")
        if self.l4a_result:
            snap = self.l4a_result
            status = "PASS" if snap.passed else f"FAIL checks={snap.issue_checks}"
            lines.append(f"  {status}")
            if not snap.passed and snap.feedback:
                for ln in snap.feedback.splitlines()[:10]:
                    lines.append(f"  {ln}")
        else:
            lines.append("  not run yet")
        lines.append("")

        # Best checkpoint
        if self.best_score >= 0:
            bs = "PASS" if (self.best_snap and self.best_snap.passed) else "FAIL"
            lines.append(
                f"BEST CHECKPOINT: step={self.best_step}  "
                f"score={self.best_score:.3f}  {bs}"
            )
        else:
            lines.append("BEST CHECKPOINT: none")
        lines.append("")

        # Accumulated lessons
        if self.lessons:
            lines.append("ACCUMULATED LESSONS:")
            for i, l in enumerate(self.lessons, 1):
                lines.append(f"  {i}. {l}")
        else:
            lines.append("ACCUMULATED LESSONS: none yet")

        return "\n".join(lines)


# ── Scoring helper ────────────────────────────────────────────────────────────

def _score_snap(snap: Optional[L4aSnapshot]) -> float:
    """Map an L4a snapshot to a scalar ∈ [-0.1, 1.0]. Higher = better."""
    if snap is None:
        return -0.1
    if snap.passed:
        return 1.0
    # Partial credit: fewer issues = higher score
    n = len(snap.issues)
    return max(0.0, 0.8 - n * 0.1)
