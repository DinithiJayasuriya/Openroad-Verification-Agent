"""controller.py — LLM controller for the causal agent loop.

Reads the full CausalAgentState and decides the next action.

The controller is NOT a rule engine — it reasons over the full trajectory
(all past attempts, verifier feedback, lessons, budget remaining) and
outputs a structured decision. This lets it handle cases that rules cannot:
  - Recognise when re-retrieval has been tried and failed → switch strategy
  - Interpret L3 failures that are actually L2 chain problems
  - Be budget-aware: cheap action when budget is low
  - Accumulate lessons within an episode

Fallback: if the LLM call fails, a rule-based fallback fires so the loop
          never stalls.
"""

import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from causal_state import CausalAgentState


# ─────────────────────────────────────────────────────────────────────────────
# Decision dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ControllerDecision:
    diagnosis:      str            # root-cause reasoning
    next_action:    str            # re_generate | re_retrieve_edge | re_generate_tcl | re_extract_chain | commit_best | stop_fail
    target_edge:    str            # "src -> tgt" (only for re_retrieve_edge)
    repair_hint:    str            # injected into the next generation prompt
    updated_lesson: str            # written to state.lessons before execution
    from_fallback:  bool = False   # True if rule-based fallback fired
    rag_query:      str  = ""      # free-text RAG query (used when target_edge not in chain)


# ─────────────────────────────────────────────────────────────────────────────
# Controller system prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the Causal Arbiter for an EDA AI Agent. Your job is to resolve conflicts \
between the Verifier (who enforces strict documentation) and the LLM Generator \
(who provides the implementation), and decide the single best next action.

=== AVAILABLE ACTIONS ===

ACCEPT  (maps to commit_best)
  The LLM's reasoning in [Diagnosis] is technically sound; the code is likely correct.
  Use when: verifier is SOFT_FAIL (near-miss) AND the LLM [Diagnosis] provides a valid
  technical justification (e.g., "Singular vs Plural mismatch", "getNet returns one net
  not a list — cardinality correct for this task"). Commit the best code immediately.

re_generate
  Re-generate code using the same causal chain + APIs, injecting repair_hint to fix
  the specific issue.
  Use when: L1 syntax error, L2 flow error, or L3 hard hallucination when RAG
  re-retrieval has already been attempted and still failed.

re_retrieve_edge
  Re-run targeted RAG retrieval for a specific chain edge where the API was wrong or
  missing. Dispatcher updates the edge's API, then automatically re-generates.
  target_edge MUST be filled: "src_type -> tgt_type" if the edge is already in the
  causal chain. If the required edge is NOT in the current chain (e.g., you need
  dbBlock→dbInst but the chain only has dbBlock→dbNet), leave target_edge empty and
  instead fill rag_query with a natural-language description of what API you need
  (e.g., "create a new instance on dbBlock given a dbMaster and name").
  Use when: L3 hard hallucination AND re-retrieval not yet tried for that edge.

re_generate_tcl
  Re-generate code using design.evalTclString("...") as the fallback for operations
  the Python API cannot perform. The dispatcher will prompt the generator with the
  missing method name and a list of equivalent Tcl commands.
  Use when: the same Python method has been reported as "not a method of" in 2+
  consecutive attempts AND RAG retrieval has already failed to find a replacement.
  This is the last-resort before commit_best when the Python API has a genuine gap.

re_extract_chain
  Re-run the causal chain extractor for the task, injecting a correction hint that
  names the bad intermediate node (e.g. "odb.dbSeg does not exist"). The dispatcher
  re-extracts the chain, re-runs per-edge RAG, and re-generates code from scratch.
  Use when: L2 HIERARCHY VIOLATION or MISSING errors repeat 2+ times on the same
  node, suggesting the chain extractor invented a type that doesn't exist in the API.
  repair_hint MUST contain the bad node name (e.g. "dbSeg").

commit_best
  Commit the best checkpoint found so far and exit the loop.
  Use when: verifier PASS, OR budget = 1 and best checkpoint is reasonable.

stop_fail
  Give up. Use only when: budget = 0 and no passing code was ever found.

=== DECISION RULES ===

Structural Priority (RE_GENERATE):
  If the LLM skipped a required Node in the Causal Chain (L2: missing getBlock,
  getDb, getTech etc.), you MUST choose re_generate — no negotiation.

Semantic Flexibility (ACCEPT):
  If the verifier is SOFT_FAIL and the LLM used a method that is a singular/plural
  variation of the RAG suggestion (e.g., getNet vs getNets) AND the [Diagnosis]
  provides a valid justification based on OpenDB object model logic, choose ACCEPT.
  Example valid diagnosis: "dbBTerm.getNet() returns a single net — getNets() does
  not exist; cardinality is correct for a block terminal."

Hierarchical Truth (RE_RETRIEVE):
  If the LLM claims a method belongs to a different object type from what the Causal
  Chain specifies (e.g., findMaster on dbDatabase instead of dbBlock), and the chain
  hierarchy may be wrong, choose re_retrieve_edge for that specific edge to get
  better RAG grounding.

Hard Hallucination (RE_RETRIEVE then RE_GENERATE):
  If the LLM used a completely non-existent method with no near-miss and no valid
  diagnosis, choose re_retrieve_edge (if not already tried) or re_generate with
  explicit correct method from known-methods list.

Python API Gap (RE_GENERATE_TCL):
  If the same "is not a method of" error recurs after 2+ attempts and RAG has
  already failed to find a Python replacement, the method simply does not exist
  in the Python bindings. Choose re_generate_tcl so the dispatcher prompts the
  generator to use design.evalTclString("...") with the equivalent Tcl command.

Wrong Causal Chain (RE_EXTRACT_CHAIN):
  If L2 "HIERARCHY VIOLATION" or "MISSING" errors repeat 2+ times on the same
  chain node (e.g. verifier keeps saying "findLayer belongs to dbNet" even after
  re_generate), the chain extractor invented a non-existent type. Choose
  re_extract_chain and set repair_hint to the bad node name so the dispatcher
  can re-extract with a correction hint. This runs BEFORE re_generate_tcl because
  fixing the chain is cheaper than abandoning Python.

Budget = 1 → commit_best unless a single clear fix is available.
Repeating same error after 2 attempts → change strategy.

=== OUTPUT FORMAT ===
Output ONLY valid JSON — no explanation, no markdown:
{
  "diagnosis":      "concise root-cause reasoning, referencing LLM Diagnosis if relevant",
  "next_action":    "ACCEPT | re_generate | re_retrieve_edge | re_generate_tcl | re_extract_chain | commit_best | stop_fail",
  "target_edge":    "src_type -> tgt_type  (only for re_retrieve_edge when edge IS in chain, else empty string)",
  "rag_query":      "natural-language RAG query (only for re_retrieve_edge when edge NOT in chain, else empty string)",
  "repair_hint":    "specific instruction for the code generator (empty for ACCEPT)",
  "updated_lesson": "concise lesson to remember for future steps in this run"
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# CausalController
# ─────────────────────────────────────────────────────────────────────────────

class CausalController:
    """
    LLM-powered controller. Reads CausalAgentState, returns ControllerDecision.

    Parameters
    ----------
    api_key : str   OpenAI key
    model   : str   Model to use (default gpt-4.1-mini)
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.api_key = api_key
        self.model   = model

    def decide(self, state: CausalAgentState) -> ControllerDecision:
        """Read state → return decision. Falls back to rules if LLM call fails."""

        # ── RETRIEVE LOOP RULE (pre-LLM override) ─────────────────────────────
        # If 2+ consecutive re_retrieve_edge actions without an intervening
        # re_generate, the model is stuck in a retrieve loop. Override: force
        # re_generate with the known-methods list from the static verifier.
        loop_override = self._check_retrieve_loop(state)
        if loop_override:
            return loop_override

        # ── CHAIN VIOLATION LOOP RULE (pre-LLM override) ──────────────────────
        # If L2 HIERARCHY VIOLATION / MISSING errors repeat 2+ consecutive times
        # on the same node, the causal chain likely contains a non-existent type.
        # Override: re-extract chain with a correction hint before trying Tcl.
        chain_override = self._check_chain_violation_loop(state)
        if chain_override:
            return chain_override

        # ── MISSING METHOD LOOP RULE (pre-LLM override) ───────────────────────
        # If the same "not a method of" error recurs across 2+ non-bootstrap
        # observations, the method genuinely doesn't exist in the Python API.
        # Override: switch to evalTclString() fallback.
        tcl_override = self._check_missing_method_loop(state)
        if tcl_override:
            return tcl_override

        user_msg = state.to_controller_string()

        text = self._call_llm(user_msg)
        if text:
            decision = self._parse_decision(text)
            if decision:
                # ── POST-LLM OVERRIDE: L1 SyntaxError with budget remaining ──
                # The LLM sometimes returns commit_best on an L1 SyntaxError
                # (e.g. because it thinks the code was already fixed).  An L1
                # failure means the code is syntactically broken and CANNOT be
                # executed — committing it is always wrong when budget allows.
                snap = state.static_result
                if (
                    decision.next_action == "commit_best"
                    and snap is not None
                    and not snap.passed
                    and snap.layer_failed == 1
                    and state.budget_remaining > 1
                ):
                    syntax_issue = snap.issues[0] if snap.issues else "syntax error"
                    return ControllerDecision(
                        diagnosis=f"L1 SyntaxError override — cannot commit broken code; forcing re_generate. Issue: {syntax_issue[:100]}",
                        next_action="re_generate",
                        target_edge="",
                        repair_hint=f"Fix the syntax error: {syntax_issue}",
                        updated_lesson="",
                        from_fallback=True,
                    )
                return decision

        # LLM failed or returned garbage — use rule fallback
        return self._rule_fallback(state)

    # ── Retrieve loop detection ────────────────────────────────────────────────

    def _check_retrieve_loop(self, state: CausalAgentState) -> Optional[ControllerDecision]:
        """Detect 2+ consecutive re_retrieve_edge without intervening re_generate.

        When detected, returns a forced re_generate decision that injects the
        known-methods list from static verifier feedback so the model has
        enough information to pick the right method without another RAG call.
        """
        non_boot = [o for o in state.observations if not o.is_bootstrap]
        if len(non_boot) < 2:
            return None

        # Count consecutive re_retrieve_edge from the tail of history
        consecutive = 0
        for obs in reversed(non_boot):
            if obs.action == "re_retrieve_edge":
                consecutive += 1
            else:
                break   # chain broken by any other action

        if consecutive < 2:
            return None

        # Build hint from the static verifier's full known-methods feedback
        sv = state.static_result
        hint = ""
        if sv and sv.feedback:
            hint = sv.feedback[:600]
        else:
            hint = ("RAG retrieval has not resolved the issue. "
                    "Use only the methods listed in the static verifier output above.")

        # Build a prohibitive lesson from the first issue
        lesson = ""
        if sv and sv.issues:
            # Extract the hallucinated method name if possible
            issue_text = sv.issues[0]
            lesson = (
                f"do NOT use the method that caused: {issue_text[:120]} "
                f"— RAG retrieval failed to find a replacement; use the known-methods list instead"
            )

        print(f"  [controller] RETRIEVE LOOP detected ({consecutive} consecutive "
              f"re_retrieve_edge) — forcing re_generate with known-methods list",
              flush=True)

        return ControllerDecision(
            diagnosis=(
                f"RETRIEVE LOOP: {consecutive} consecutive re_retrieve_edge actions "
                f"without progress. Switching to re_generate using the known-methods "
                f"list from the static verifier feedback."
            ),
            next_action    = "re_generate",
            target_edge    = "",
            repair_hint    = (
                f"RAG retrieval has been attempted {consecutive} times and failed. "
                f"DO NOT call any method that was rejected in prior attempts.\n\n"
                f"The static verifier lists ALL KNOWN VALID methods for each type. "
                f"Pick ONLY from those:\n\n{hint}"
            ),
            updated_lesson = lesson,
            from_fallback  = True,
        )

    # ── Chain violation loop detection ────────────────────────────────────────

    def _check_chain_violation_loop(self, state: CausalAgentState) -> Optional[ControllerDecision]:
        """Detect when L2 HIERARCHY VIOLATION or MISSING errors repeat 2+
        consecutive non-bootstrap observations on the same chain node, indicating
        the chain extractor invented a non-existent type.
        Triggers re_extract_chain to rebuild the chain with a correction hint.
        """
        snap = state.static_result
        if snap is None or snap.passed or snap.layer_failed != 2:
            return None

        current_issue = snap.issues[0] if snap.issues else ""
        if "HIERARCHY VIOLATION" not in current_issue and "MISSING" not in current_issue:
            return None

        # Count consecutive L2 failures from the tail of non-bootstrap history
        non_boot = [o for o in state.observations if not o.is_bootstrap]
        consecutive_l2 = 0
        for obs in reversed(non_boot):
            if "FAIL(L2)" in obs.result:
                consecutive_l2 += 1
            else:
                break

        if consecutive_l2 < 2:
            return None

        # Extract the bad node name from the issue text.
        # Patterns: "to acquire dbSeg"  /  "'type dbTech'"  /  "belongs to dbNet"
        bad_node = ""
        m = re.search(r"to acquire (\w+)", current_issue)
        if m:
            bad_node = m.group(1)          # e.g. "dbSeg"
        else:
            m = re.search(r"type (\w+)\b", current_issue)
            if m:
                bad_node = m.group(1)      # e.g. "dbTech"

        # Verify the bad node is in the chain AND has a low-confidence RAG hit
        # (score < 0.8 on its edge → extractor was guessing)
        if bad_node:
            confirmed = False
            all_edges = state.all_edges or list(zip(state.chain[:-1], state.chain[1:]))
            for i, (src, tgt) in enumerate(all_edges):
                tgt_short = tgt.split(".")[-1]
                if bad_node in tgt_short or tgt_short in bad_node:
                    api = state.edge_apis[i] if i < len(state.edge_apis) else None
                    score = api.get("score", 0.0) if api else 0.0
                    if score < 0.8:
                        confirmed = True
                        break
            if not confirmed:
                # Node exists with a confident RAG hit → chain is probably right,
                # the L2 violation is a genuine method ownership error, not a bad node.
                return None

        print(
            f"  [controller] CHAIN VIOLATION LOOP: {consecutive_l2} consecutive L2 failures "
            f"— re-extracting chain (bad node='{bad_node}')",
            flush=True,
        )

        return ControllerDecision(
            diagnosis=(
                f"L2 violation repeated {consecutive_l2} times on node '{bad_node}'. "
                f"The chain extractor likely invented a non-existent type. "
                f"Re-extracting causal chain with correction hint."
            ),
            next_action="re_extract_chain",
            target_edge="",
            repair_hint=bad_node,
            updated_lesson=(
                f"'{bad_node}' may not exist in the OpenROAD Python API — "
                f"chain re-extraction was triggered after {consecutive_l2} L2 failures."
            ),
            from_fallback=True,
        )

    # ── Missing method loop detection ─────────────────────────────────────────

    def _check_missing_method_loop(self, state: CausalAgentState) -> Optional[ControllerDecision]:
        """Detect when the same Python method keeps failing as 'not a method of'
        across 2+ non-bootstrap observations, indicating the method simply does not
        exist in the Python API.  Triggers re_generate_tcl (evalTclString fallback).
        """
        snap = state.static_result
        if snap is None or snap.passed or snap.layer_failed != 3:
            return None

        current_issue = snap.issues[0] if snap.issues else ""
        if "is not a method of" not in current_issue:
            return None

        # Extract the hallucinated method name, e.g. 'createInst()' → 'createInst'
        m = re.search(r"'(\w+)\(\)'", current_issue)
        if not m:
            return None
        bad_method = m.group(1)

        # Count previous non-bootstrap observations that also reported this method failing
        non_boot = [o for o in state.observations if not o.is_bootstrap]
        prior_count = sum(1 for o in non_boot if bad_method in o.result)

        if prior_count < 2:
            return None

        total = prior_count + 1  # include current failure
        print(
            f"  [controller] MISSING METHOD LOOP: '{bad_method}()' failed {total} times "
            f"— forcing re_generate_tcl (evalTclString fallback)",
            flush=True,
        )

        return ControllerDecision(
            diagnosis=(
                f"'{bad_method}()' does not exist in the OpenROAD Python API. "
                f"RAG retrieval failed to find a replacement after {prior_count} prior "
                f"attempt(s). Switching to evalTclString() Tcl fallback."
            ),
            next_action="re_generate_tcl",
            target_edge="",
            repair_hint=bad_method,
            updated_lesson=(
                f"'{bad_method}()' is not in the OpenROAD Python API. "
                f"Use design.evalTclString('<tcl_command>') for operations not "
                f"supported by the Python bindings."
            ),
            from_fallback=True,
        )

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, user_msg: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0,
            "max_tokens":  500,
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
                    print(f"    [controller rate-limit] waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"    [controller HTTP {e.code}]", flush=True)
                    return ""
            except Exception as exc:
                print(f"    [controller error] {exc}", flush=True)
                return ""
        return ""

    # ── JSON parse ────────────────────────────────────────────────────────────

    def _parse_decision(self, text: str) -> Optional[ControllerDecision]:
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            text  = "\n".join(lines).strip()
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return None

        action = d.get("next_action", "")
        # ACCEPT is the Causal Arbiter's semantic-flexibility action → commit_best
        if action == "ACCEPT":
            action = "commit_best"
        if action not in ("re_generate", "re_retrieve_edge", "re_generate_tcl",
                          "re_extract_chain", "commit_best", "stop_fail"):
            return None

        return ControllerDecision(
            diagnosis      = str(d.get("diagnosis",      "")),
            next_action    = action,
            target_edge    = str(d.get("target_edge",    "")),
            repair_hint    = str(d.get("repair_hint",    "")),
            updated_lesson = str(d.get("updated_lesson", "")),
            from_fallback  = False,
            rag_query      = str(d.get("rag_query",      "")),
        )

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _rule_fallback(self, state: CausalAgentState) -> ControllerDecision:
        """Simple rules — fire when LLM is unavailable."""
        snap = state.static_result

        if snap is None or (snap.passed and not snap.is_soft_fail):
            return ControllerDecision(
                diagnosis="Verifier passed — committing.",
                next_action="commit_best",
                target_edge="", repair_hint="", updated_lesson="",
                from_fallback=True,
            )

        if snap is not None and snap.passed and snap.is_soft_fail:
            # SOFT_FAIL with a near-miss: if LLM provided a diagnosis, accept it
            diag = getattr(state, "llm_diagnosis", "")
            if diag:
                return ControllerDecision(
                    diagnosis=f"SOFT_FAIL near-miss — LLM diagnosis present, accepting: {diag[:80]}",
                    next_action="commit_best",
                    target_edge="", repair_hint="", updated_lesson="",
                    from_fallback=True,
                )
            # No diagnosis — re-retrieve the mismatched edge
            if snap.api_diffs and state.chain:
                d = snap.api_diffs[0]
                edge = f"{d.src_type} -> {d.tgt_type}"
                return ControllerDecision(
                    diagnosis=f"SOFT_FAIL near-miss but no LLM diagnosis — re-retrieving {edge}",
                    next_action="re_retrieve_edge",
                    target_edge=edge,
                    repair_hint=snap.issues[0] if snap.issues else "",
                    updated_lesson="",
                    from_fallback=True,
                )

        if state.budget_remaining <= 1:
            return ControllerDecision(
                diagnosis="Budget nearly exhausted — committing best.",
                next_action="commit_best",
                target_edge="", repair_hint="", updated_lesson="",
                from_fallback=True,
            )

        layer = snap.layer_failed
        issue = snap.issues[0] if snap.issues else ""

        # L3: try re-retrieve if not already done recently
        if layer == 3:
            already_retrieved = any(
                o.action == "re_retrieve_edge"
                for o in state.observations[-3:]
            )
            if not already_retrieved and state.chain:
                edge = _guess_edge_from_issue(issue, state.chain)
                if edge:
                    return ControllerDecision(
                        diagnosis=f"L3 API hallucination — {issue[:80]}",
                        next_action="re_retrieve_edge",
                        target_edge=edge,
                        repair_hint=issue,
                        updated_lesson="",
                        from_fallback=True,
                    )

        # Default: re-generate with verifier feedback
        return ControllerDecision(
            diagnosis=f"L{layer} failure — re-generating with feedback.",
            next_action="re_generate",
            target_edge="",
            repair_hint=snap.feedback[:400] if snap.feedback else issue,
            updated_lesson="",
            from_fallback=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _guess_edge_from_issue(issue: str, chain: list) -> str:
    """
    Try to extract a 'src -> tgt' edge from a verifier issue string.
    Falls back to the first chain edge if nothing found.
    """
    # Look for type names mentioned in the issue
    for i, (src, tgt) in enumerate(zip(chain[:-1], chain[1:])):
        src_short = src.split(".")[-1].lower()
        tgt_short = tgt.split(".")[-1].lower()
        if src_short in issue.lower() or tgt_short in issue.lower():
            return f"{src} -> {tgt}"
    # Fallback: first edge
    if len(chain) >= 2:
        return f"{chain[0]} -> {chain[1]}"
    return ""
