"""llm_verifier.py — LLM semantic verifier for the causal pipeline.

Runs ONLY after static + causal verifiers both pass (L1-L3 clean).
Uses gpt-4.1-mini to check 5 semantic angles:

  A. Task requirements  — does the code address everything the task asks?
  B. Object acquisition — are getBlock/getTech/getDb called when needed?
  C. API correctness    — methods on correct types? no hallucinated methods?
  D. Logic correctness  — does the logic match the task semantics?
  E. Output             — are results printed / returned as required?

Returns VerifierSnapshot with:
  passed       = True iff ≥ 3/5 sub-checks are YES
  confidence   = n_YES / 5  (continuous PRM score for controller)
  issues       = failing sub-check descriptions
  layer_failed = 5 (semantic) when failed

fail_open=True: API errors → PASS-through (non-blocking).
"""

import json
import time
import urllib.request
import urllib.error
from typing import List, Optional

from causal_state import VerifierSnapshot


_SYSTEM = """\
You are an expert OpenROAD Python API reviewer.

Given a task and a Python script, check the code on EXACTLY these 5 angles:
  A. TASK REQUIREMENTS   — does the code address everything the task asks?
  B. OBJECT ACQUISITION  — are all prerequisite objects acquired?
                           (design.getBlock() before block ops,
                            design.getDb() before database ops,
                            design.getTech() before tech ops, etc.)
  C. API CORRECTNESS     — are methods called on the correct object types?
                           No hallucinated method names?
  D. LOGIC CORRECTNESS   — does the code logic correctly implement the task
                           semantics? (correct iteration, correct conditions,
                           correct attribute access — e.g. getLocation()
                           returns a tuple (x,y), NOT a Point object with .x)
  E. OUTPUT              — does the code print or return the required results?

For each angle respond YES or NO, then one short sentence explaining.
Reply ONLY with valid JSON in this exact format (no markdown):
{
  "A": {"verdict": "YES", "note": "..."},
  "B": {"verdict": "NO",  "note": "..."},
  "C": {"verdict": "YES", "note": "..."},
  "D": {"verdict": "YES", "note": "..."},
  "E": {"verdict": "YES", "note": "..."}
}
"""

_USER_TMPL = """\
Task: {task}

Expected acquisition chain (for context):
{chain_context}

Code to review:
```python
{code}
```
"""


class CausalLLMVerifier:
    """LLM semantic verifier using gpt-4.1-mini.

    Parameters
    ----------
    api_key   : OpenAI API key
    model     : model name (default gpt-4.1-mini)
    fail_open : if True, API errors → PASS-through (default True)
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini",
                 fail_open: bool = True):
        self.api_key   = api_key
        self.model     = model
        self.fail_open = fail_open

    def verify(self, task: str, code: str,
               chain_context: str = "") -> VerifierSnapshot:
        """Run LLM semantic check. Returns VerifierSnapshot (layer 5 if failed)."""
        user_msg = _USER_TMPL.format(
            task          = task,
            chain_context = chain_context or "(not provided)",
            code          = code,
        )
        raw = self._call(user_msg)

        if raw is None:
            if self.fail_open:
                return VerifierSnapshot(
                    passed=True, layer_failed=0,
                    issues=["LLM verifier skipped (API error)"],
                    feedback="PASS (fail_open — API error)", confidence=0.5,
                )
            return VerifierSnapshot(
                passed=False, layer_failed=5,
                issues=["LLM verifier API call failed"],
                feedback="LLM verifier API call failed", confidence=0.0,
            )

        return self._parse(raw)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call(self, user_msg: str) -> Optional[str]:
        payload = json.dumps({
            "model":       self.model,
            "messages":    [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0,
            "max_tokens":  400,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data    = payload,
            headers = {"Content-Type":  "application/json",
                       "Authorization": f"Bearer {self.api_key}"},
            method  = "POST",
        )
        wait = 10
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                return body["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"  [LLMVerifier] 429 — retry in {wait}s", flush=True)
                    time.sleep(wait); wait = min(wait * 2, 120)
                else:
                    print(f"  [LLMVerifier] HTTP {e.code}", flush=True)
                    return None
            except Exception as exc:
                print(f"  [LLMVerifier] error: {exc}", flush=True)
                return None
        return None

    # ── response parser ────────────────────────────────────────────────────────

    def _parse(self, raw: str) -> VerifierSnapshot:
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(
                ln for ln in text.splitlines()
                if not ln.strip().startswith("```")
            ).strip()

        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            if self.fail_open:
                return VerifierSnapshot(
                    passed=True, layer_failed=0,
                    issues=["LLM verifier: unparseable response"],
                    feedback="PASS (fail_open, parse error)", confidence=0.5,
                )
            return VerifierSnapshot(
                passed=False, layer_failed=5,
                issues=["LLM verifier returned unparseable response"],
                feedback=raw[:200], confidence=0.0,
            )

        _LABELS = {
            "A": "Task requirements",
            "B": "Object acquisition",
            "C": "API correctness",
            "D": "Logic correctness",
            "E": "Output",
        }
        yes_count = 0
        issues: List[str] = []
        for key in ("A", "B", "C", "D", "E"):
            entry   = d.get(key, {})
            verdict = str(entry.get("verdict", "NO")).strip().upper()
            note    = str(entry.get("note", "")).strip()
            if verdict == "YES":
                yes_count += 1
            else:
                issues.append(f"[{key}] {_LABELS[key]}: {note}")

        confidence = yes_count / 5
        passed     = yes_count >= 3   # majority ≥ 3/5

        if passed:
            return VerifierSnapshot(
                passed=True, layer_failed=0,
                issues=issues,          # soft warnings (non-failing checks)
                feedback=f"LLM PASS (conf={confidence:.2f})",
                confidence=confidence,
            )

        feedback_lines = [f"LLM semantic check FAILED (conf={confidence:.2f}):"]
        feedback_lines += [f"  {iss}" for iss in issues]
        return VerifierSnapshot(
            passed=False, layer_failed=5,
            issues=issues,
            feedback="\n".join(feedback_lines),
            confidence=confidence,
        )
