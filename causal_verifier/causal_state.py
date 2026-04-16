"""causal_state.py — Mutable runtime state for the causal agent loop.

One CausalAgentState is created per task and updated by the bootstrap
steps and (later) the controller. The controller reads it to decide the
next action; it never writes directly.

Bootstrap steps (always run, no budget cost):
  1. causal_extract   — task text → causal chain
  2. causal_rag       — per-edge RAG retrieval → edge_apis
  3. causal_generate  — skeleton build + GPT completion → current_code
  4. static_verify    — static verifier on generated code

After bootstrap the controller loop takes over (to be implemented).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── single action-observation record ──────────────────────────────────────────

@dataclass
class ToolObservation:
    step:       int
    action:     str    # which action was executed
    result:     str    # one-line summary
    detail:     str = ""   # full detail (code, verifier output, etc.)
    is_bootstrap: bool = False   # True for free bootstrap steps


# ── verifier snapshot ─────────────────────────────────────────────────────────

@dataclass
class VerifierSnapshot:
    passed:       bool
    layer_failed: int       # 0=pass  1=syntax  2=prereq  3=api  4=swig_attr
    issues:       List[str]
    feedback:     str
    confidence:   float = 0.0   # reserved for LLM verifier; static gives 0.0/1.0
    api_diffs:    List[Any]  = field(default_factory=list)  # List[APIEdgeDiff] from causal verifier
    is_soft_fail: bool = False  # SOFT_FAIL: passed=True but near-miss API detected; controller may ACCEPT with LLM diagnosis


# ── main state object ──────────────────────────────────────────────────────────

@dataclass
class CausalAgentState:
    """Full runtime state for one task in the causal agent loop.

    The bootstrap populates the causal fields (chain, edge_apis, skeleton).
    The controller (TBD) reads everything via to_controller_string().
    """
    task:        str
    max_budget:  int     # controller action budget (bootstrap is free)

    # ── step / budget counters ─────────────────────────────────────────────────
    step:         int = 0
    budget_used:  int = 0

    # ── causal chain fields (filled by bootstrap) ──────────────────────────────
    chain:        List[str]              = field(default_factory=list)
    # chain = first (primary) path for display/logging, e.g. ["openroad.Design", "odb.dbBlock", "odb.dbInst"]

    # Multi-path fields (set when chain extractor returns more than one path)
    paths:        List[List[str]]        = field(default_factory=list)
    # paths[i] = full type sequence for acquisition path i
    # e.g. [["openroad.Design","odb.dbBlock","odb.dbBTerm"],
    #        ["openroad.Design","odb.dbBlock","odb.dbNet"]]

    all_edges:    List[Any]              = field(default_factory=list)
    # all_edges = unique (src, tgt) tuples across all paths, in order
    # edge_apis[i] corresponds to all_edges[i]

    action_node:  str = ""   # Action Node string from extractor, e.g. "connect(dbBTerm, dbNet)"

    edge_apis:    List[Optional[Dict]]   = field(default_factory=list)
    # one dict per edge in all_edges (function_name, parameters, return_type, description, score)
    # or None if RAG found nothing

    skeleton:     str = ""   # schema-correct acquisition skeleton
    api_summary:  str = ""   # human-readable "src->tgt: method" per edge

    # ── code ───────────────────────────────────────────────────────────────────
    current_code:  str        = ""
    code_history:  List[str]  = field(default_factory=list)

    # best checkpoint
    best_code:     str   = ""
    best_score:    float = -1.0
    best_step:     int   = -1
    best_snapshot: Optional[VerifierSnapshot] = None

    # ── verifier state (latest) ────────────────────────────────────────────────
    static_result: Optional[VerifierSnapshot] = None
    llm_result:    Optional[VerifierSnapshot] = None   # set only when static passes

    # ── accumulated knowledge ──────────────────────────────────────────────────
    lessons:      List[str]          = field(default_factory=list)
    observations: List[ToolObservation] = field(default_factory=list)

    # ── commit flag ────────────────────────────────────────────────────────────
    committed:      bool = False
    committed_code: str  = ""

    # ── generator self-diagnosis ───────────────────────────────────────────────
    llm_diagnosis:  str  = ""   # latest [Diagnosis] block from generator LLM

    # ── budget ────────────────────────────────────────────────────────────────
    @property
    def budget_remaining(self) -> int:
        return self.max_budget - self.budget_used

    # ── observation recording ─────────────────────────────────────────────────

    def add_bootstrap_obs(self, action: str, result: str, detail: str = "") -> None:
        """Record a bootstrap step — step counter advances but budget is NOT charged."""
        self.observations.append(ToolObservation(
            step=self.step, action=action, result=result,
            detail=detail, is_bootstrap=True,
        ))
        self.step += 1   # step counter advances for history ordering

    def add_observation(self, action: str, result: str, detail: str = "") -> None:
        """Record a controller action — charges one unit of budget."""
        self.observations.append(ToolObservation(
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
        """Update best checkpoint if current code is better. Returns True if updated."""
        if not self.current_code:
            return False
        score = _score_snapshot(self.static_result)
        if score > self.best_score:
            self.best_code     = self.current_code
            self.best_score    = score
            self.best_step     = self.step
            self.best_snapshot = self.static_result
            return True
        return False

    # ── controller context string ─────────────────────────────────────────────

    def to_controller_string(self) -> str:
        """Compact information-dense summary for the controller LLM."""
        lines = [
            f"TASK: {self.task}",
            f"STEP: {self.step}  BUDGET_REMAINING: {self.budget_remaining}/{self.max_budget}",
            "",
            "CAUSAL PATHS:",
        ]
        if self.paths:
            for j, p in enumerate(self.paths, 1):
                lines.append(f"  Path {j}: {' -> '.join(p)}")
        elif self.chain:
            lines.append(f"  Path 1: {' -> '.join(self.chain)}")
        else:
            lines.append("  (not extracted)")
        if self.action_node:
            lines.append(f"  Action: {self.action_node}")
        lines.append(f"  APIs: {self.api_summary or '(none)'}")
        lines.append("")

        lines.append("SKELETON:")
        if self.skeleton:
            for ln in self.skeleton.splitlines():
                lines.append(f"  {ln}")
        else:
            lines.append("  (not built)")
        lines.append("")

        lines.append("ACTION-OBSERVATION HISTORY:")
        if not self.observations:
            lines.append("  (none)")
        for obs in self.observations:
            tag = "[boot]" if obs.is_bootstrap else f"[{obs.step}]"
            lines.append(f"  {tag} {obs.action:25s} → {obs.result}")
        lines.append("")

        if self.current_code:
            preview = self.current_code[:200].replace("\n", "↵")
            lines.append(f"CURRENT CODE ({len(self.current_code)} chars): {preview}...")
        else:
            lines.append("CURRENT CODE: none")
        lines.append("")

        lines.append("LATEST STATIC VERIFIER:")
        if self.static_result:
            sv = self.static_result
            if sv.passed and sv.is_soft_fail:
                s = "SOFT_FAIL (near-miss API — see LLM Diagnosis)"
            elif sv.passed:
                s = "PASS"
            else:
                s = f"FAIL(layer={sv.layer_failed})"
            lines.append(f"  {s}  issues={sv.issues[:2]}")
            if sv.feedback and (not sv.passed or sv.is_soft_fail):
                lines.append(f"  feedback: {sv.feedback[:400]}")
            # Show API diffs if present (from causal verifier L3)
            if sv.api_diffs:
                lines.append("  API DIFFS (hallucinated vs RAG-retrieved):")
                for d in sv.api_diffs:
                    src_s = d.src_type.split(".")[-1]
                    tgt_s = d.tgt_type.split(".")[-1]
                    nm    = " [NEAR-MISS]" if getattr(d, "is_near_miss", False) else ""
                    lines.append(f"    {src_s}→{tgt_s}{nm}: code used {d.code_methods}, "
                                 f"RAG says '{d.rag_method}'")
        else:
            lines.append("  not run yet")
        lines.append("")

        lines.append("LLM DIAGNOSIS (from generator):")
        if self.llm_diagnosis:
            lines.append(f"  {self.llm_diagnosis[:400]}")
        else:
            lines.append("  (none provided)")
        lines.append("")

        lines.append("LATEST LLM SEMANTIC VERIFIER:")
        if self.llm_result:
            lv = self.llm_result
            s  = "PASS" if lv.passed else f"FAIL(layer={lv.layer_failed})"
            lines.append(f"  {s}  conf={lv.confidence:.2f}  issues={lv.issues[:2]}")
            if not lv.passed and lv.feedback:
                lines.append(f"  feedback: {lv.feedback[:300]}")
        else:
            lines.append("  not run yet (runs only after static PASS)")
        lines.append("")

        if self.best_score >= 0:
            bs = self.best_snapshot
            bs_str = "PASS" if (bs and bs.passed) else f"FAIL(layer={bs.layer_failed if bs else '?'})"
            lines.append(f"BEST CHECKPOINT: step={self.best_step}  score={self.best_score:.3f}  {bs_str}")
        else:
            lines.append("BEST CHECKPOINT: none")
        lines.append("")

        if self.lessons:
            lines.append("ACCUMULATED LESSONS:")
            for i, l in enumerate(self.lessons, 1):
                lines.append(f"  {i}. {l}")
        else:
            lines.append("ACCUMULATED LESSONS: none yet")

        return "\n".join(lines)


# ── scoring helper ─────────────────────────────────────────────────────────────

def _score_snapshot(snap: Optional[VerifierSnapshot]) -> float:
    """Map a verifier snapshot to a scalar ∈ [-0.1, 1.0]. Higher = better."""
    if snap is None:
        return -0.1
    if snap.passed:
        return 0.9 + 0.1 * snap.confidence
    layer_score = snap.layer_failed / 5.0   # 0..0.8 (layers 1-4)
    return layer_score * 0.5 + snap.confidence * 0.3
