"""flow_pipeline.py — FlowPipeline: orchestrator for single-tool flow tasks.

Ties together:
  1. FlowChainExtractor      — task → MultiPathCausalGraph
  2. LLM code generator      — graph.to_constraint_prompt() → Python code
  3. FlowSequencingVerifier  — L4 ordering + sandwich check
  4. Surgical repair         — L4SequencingResult.repaired_code → re-verify

Entry points
------------
  pipeline = FlowPipeline(openai_key="sk-...")

  # Just get the graph + constraint prompt (no generation)
  result = pipeline.plan(task)
  print(result.constraint_prompt)

  # Full loop: generate → verify → repair, up to `budget` attempts
  result = pipeline.run(task, budget=3)
  if result.passed:
      print(result.final_code)
  else:
      print(result.feedback)

This file does NOT modify any existing pipeline files.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from .flow_chain_extractor    import FlowChainExtractor
    from .flow_causal_graph       import MultiPathCausalGraph
    from .flow_sequencing_verifier import FlowSequencingVerifier, L4SequencingResult
except ImportError:
    from flow_chain_extractor     import FlowChainExtractor          # type: ignore
    from flow_causal_graph        import MultiPathCausalGraph         # type: ignore
    from flow_sequencing_verifier import FlowSequencingVerifier, L4SequencingResult  # type: ignore


# ── Per-attempt record ────────────────────────────────────────────────────────

@dataclass
class FlowAttempt:
    """Record of one generate-verify cycle."""
    attempt_number: int
    code:           str
    l4_result:      L4SequencingResult
    was_repaired:   bool = False   # True if this code came from the repairer


# ── Pipeline result ───────────────────────────────────────────────────────────

@dataclass
class FlowPipelineResult:
    """Returned by FlowPipeline.run() and FlowPipeline.plan().

    Attributes
    ----------
    task              : Original task string.
    graph             : The MultiPathCausalGraph (None if not a flow task).
    constraint_prompt : The prompt section injected into the generator.
    final_code        : Best code produced (passing or best attempt).
    passed            : True only when L4 passes.
    l4_result         : Most recent L4SequencingResult.
    attempts          : All generate-verify cycles in order.
    feedback          : Human-readable summary for the caller.
    """
    task:              str
    graph:             Optional[MultiPathCausalGraph]
    constraint_prompt: str
    final_code:        str        = ""
    passed:            bool       = False
    l4_result:         Optional[L4SequencingResult] = None
    attempts:          List[FlowAttempt] = field(default_factory=list)
    feedback:          str        = ""

    def summary(self) -> str:
        n     = len(self.attempts)
        badge = "PASS" if self.passed else "FAIL"
        tool  = self.graph.action_path.tool_def.tool_id if self.graph else "unknown"
        return f"[FlowPipeline] {badge} | tool={tool} | attempts={n}"


# ── Generation system prompt ──────────────────────────────────────────────────

_GENERATION_SYSTEM_PROMPT = """\
You are an expert OpenROAD Python API programmer.
Generate a complete Python script that runs inside the OpenROAD interactive shell.

RULES:
1. `design` (openroad.Design) and `tech` (openroad.Tech) are pre-available — do NOT import or re-create them.
2. Follow the FLOW TASK CONSTRAINT BLOCK below EXACTLY:
   - Acquire objects in the stated object path order.
   - Call the tool methods in the stated action path order.
   - Implement the sandwich structure if specified (pre-measure → tool → post-measure).
3. Do NOT call any method on the wrong object type.
4. Output ONLY the Python code. No explanation, no markdown fences.
"""


# ── FlowPipeline ──────────────────────────────────────────────────────────────

class FlowPipeline:
    """Orchestrates extraction → generation → L4 verification → repair.

    Parameters
    ----------
    openai_key   : OpenAI API key (required for generation).
    model        : Model to use for code generation (default gpt-4.1-mini).
    """

    def __init__(
        self,
        openai_key: str,
        model: str = "gpt-4.1-mini",
    ):
        self._api_key  = openai_key
        self._model    = model
        self._extractor = FlowChainExtractor(openai_key=openai_key, model=model)
        self._verifier  = FlowSequencingVerifier()

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(self, task: str) -> FlowPipelineResult:
        """Extract graph and build constraint prompt — no generation.

        Returns a FlowPipelineResult with graph and constraint_prompt populated.
        result.graph is None if the task is not a recognised flow task.
        """
        graph = self._extractor.extract(task)
        if graph is None:
            return FlowPipelineResult(
                task=task, graph=None,
                constraint_prompt="",
                feedback=f"Not a flow task — no tool keyword matched: '{task}'",
            )
        prompt = graph.to_constraint_prompt()
        return FlowPipelineResult(
            task=task, graph=graph,
            constraint_prompt=prompt,
        )

    def run(self, task: str, budget: int = 3) -> FlowPipelineResult:
        """Full pipeline: extract → generate → L4 check → repair, up to budget.

        Each budget step is one generate-verify cycle.
        On L4 failure, if the repairer produces repaired code it is verified
        immediately (counts as one extra attempt) before regenerating.

        Returns FlowPipelineResult with passed=True as soon as L4 passes.
        On budget exhaustion returns the best result seen (fewest issues).
        """
        # Step 1: extract graph
        plan_result = self.plan(task)
        if plan_result.graph is None:
            return plan_result

        graph    = plan_result.graph
        prompt   = plan_result.constraint_prompt
        attempts: List[FlowAttempt] = []
        best_code = ""
        best_l4: Optional[L4SequencingResult] = None

        repair_hint = ""   # injected into next generation when L4 fails

        for attempt_n in range(1, budget + 1):
            print(f"  [FlowPipeline] Attempt {attempt_n}/{budget}...", flush=True)

            # Step 2: generate code
            code = self._generate(task, prompt, repair_hint)
            if not code:
                print(f"  [FlowPipeline] Generation failed on attempt {attempt_n}", flush=True)
                continue

            # Step 3: L4 verify
            l4 = self._verifier.verify(code, graph)
            attempts.append(FlowAttempt(attempt_n, code, l4, was_repaired=False))

            # Track best (fewest issues)
            if best_l4 is None or len(l4.issues) < len(best_l4.issues):
                best_code = code
                best_l4   = l4

            if l4.passed:
                print(f"  [FlowPipeline] PASS on attempt {attempt_n}", flush=True)
                return FlowPipelineResult(
                    task=task, graph=graph,
                    constraint_prompt=prompt,
                    final_code=code,
                    passed=True,
                    l4_result=l4,
                    attempts=attempts,
                    feedback="PASS",
                )

            # Step 4: try surgical repair before next generation
            if l4.repaired_code:
                print(f"  [FlowPipeline] Applying repair...", flush=True)
                l4_repaired = self._verifier.verify(l4.repaired_code, graph)
                attempts.append(FlowAttempt(
                    attempt_n, l4.repaired_code, l4_repaired, was_repaired=True
                ))
                if best_l4 is None or len(l4_repaired.issues) < len(best_l4.issues):
                    best_code = l4.repaired_code
                    best_l4   = l4_repaired

                if l4_repaired.passed:
                    print(f"  [FlowPipeline] PASS after repair (attempt {attempt_n})", flush=True)
                    return FlowPipelineResult(
                        task=task, graph=graph,
                        constraint_prompt=prompt,
                        final_code=l4.repaired_code,
                        passed=True,
                        l4_result=l4_repaired,
                        attempts=attempts,
                        feedback="PASS (after repair)",
                    )

            # Build repair hint for next generation from current L4 issues
            repair_hint = self._build_repair_hint(l4, graph)
            print(
                f"  [FlowPipeline] FAIL — {len(l4.issues)} issue(s): "
                + "; ".join(i.check for i in l4.issues),
                flush=True,
            )

        # Budget exhausted
        feedback = (
            f"Budget exhausted after {len(attempts)} attempt(s). "
            f"Best result had {len(best_l4.issues) if best_l4 else '?'} L4 issue(s).\n"
            + (best_l4.feedback if best_l4 else "")
        )
        return FlowPipelineResult(
            task=task, graph=graph,
            constraint_prompt=prompt,
            final_code=best_code,
            passed=False,
            l4_result=best_l4,
            attempts=attempts,
            feedback=feedback,
        )

    # ── Code generation ───────────────────────────────────────────────────────

    def _generate(self, task: str, constraint_prompt: str, repair_hint: str = "") -> str:
        """Call the LLM to generate code for the task.

        constraint_prompt  : injected from graph.to_constraint_prompt()
        repair_hint        : appended when a previous attempt failed L4
        """
        user_parts = [
            f"Task: {task}",
            "",
            "FLOW TASK CONSTRAINT BLOCK (follow exactly):",
            constraint_prompt,
        ]
        if repair_hint:
            user_parts += [
                "",
                "REPAIR HINT (fix these issues from the previous attempt):",
                repair_hint,
            ]
        user_parts.append("\nOutput the complete Python script:")
        user_msg = "\n".join(user_parts)

        payload = json.dumps({
            "model":    self._model,
            "messages": [
                {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0,
            "max_tokens":  800,
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
                text = body["choices"][0]["message"]["content"].strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = "\n".join(
                        ln for ln in text.splitlines()
                        if not ln.strip().startswith("```")
                    )
                return text.strip()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 10 * (2 ** attempt)
                    print(f"    [rate limit] waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    [HTTP {e.code}] generation failed", flush=True)
                    return ""
            except Exception as exc:
                print(f"    [generation error] {exc}", flush=True)
                return ""
        return ""

    # ── Repair hint builder ───────────────────────────────────────────────────

    def _build_repair_hint(
        self, l4: L4SequencingResult, graph: MultiPathCausalGraph
    ) -> str:
        """Summarise L4 failures into a concise repair instruction."""
        lines = ["Fix these L4 sequencing issues:"]
        for iss in l4.issues:
            lines.append(f"  [{iss.check}] {iss.description}")
            if iss.repair_hint:
                lines.append(f"         → {iss.repair_hint}")
        lines += [
            "",
            f"Required method order: {graph.action_path.action_summary}",
        ]
        if graph.has_sandwich:
            op = graph.object_path
            lines += [
                f"Required sandwich: assign {op.pre_var} BEFORE tool, "
                f"{op.post_var} AFTER tool.",
            ]
        return "\n".join(lines)
