"""
llm_adversary.py — LLM-A: Adversarial task generator (GAN generator role).

Generates long-horizon EDA tasks grounded in the real OpenDB API, then
hardens them using solver failure signals (layer_failed, issues, api_diffs).

Two modes:
  generate_initial(n, difficulty) — cold start: N tasks at difficulty level 2-4
  harden(task_dict, failure_log)  — hot start: 2 harder variants from a failed task
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial hallucination traps — injected into system prompt so LLM-A can
# craft tasks that specifically stress these edges.
# ─────────────────────────────────────────────────────────────────────────────

_TRAPS = [
    {
        "trap":    "iterm.getDirection()",
        "correct": "iterm.isInputSignal() / iterm.isOutputSignal()",
        "type":    "odb.dbITerm",
        "use_in":  "tasks that filter pins by direction (INPUT/OUTPUT)",
    },
    {
        "trap":    "iterm.getName()",
        "correct": "iterm.getMTerm().getName()",
        "type":    "odb.dbITerm",
        "use_in":  "tasks that sort or print pin names on an ITerm",
    },
    {
        "trap":    "inst.getMasterName()",
        "correct": "inst.getMaster().getName()",
        "type":    "odb.dbInst",
        "use_in":  "tasks that group instances by master cell name",
    },
    {
        "trap":    "mterm.getDirection()",
        "correct": "mterm.getIoType()",
        "type":    "odb.dbMTerm",
        "use_in":  "tasks that traverse getMTerm() and then check direction string",
    },
    {
        "trap":    "net.getFanout() / net.getDriver()",
        "correct": "count net.getITerms(); find OUTPUT via isOutputSignal()",
        "type":    "odb.dbNet",
        "use_in":  "tasks that find max-fanout nets or locate driver instances",
    },
]


class LLMAdversary:
    """LLM-A: generates and hardens adversarial EDA tasks."""

    def __init__(self, api_key: str, model: str, rag_api_path: str):
        self.api_key = api_key
        self.model   = model
        self._type_hierarchy = self._build_type_hierarchy(rag_api_path)
        self._system_prompt  = self._make_system_prompt()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_initial(self, n: int = 10, difficulty: int = 4) -> List[Dict]:
        """Generate n fresh adversarial tasks at the given step-count difficulty."""
        domains = ["nets+fanout", "instances+master", "pins+direction",
                   "placement+status", "clock+isInClock", "mixed nets+instances",
                   "ITerms+sorting", "nets+disconnect"]
        domain_assignments = ", ".join(
            f"task {i+1}={domains[i % len(domains)]}" for i in range(n)
        )
        user_msg = (
            f"Generate exactly {n} adversarial EDA tasks.\n"
            f"Each task must have exactly {difficulty} sequential steps.\n\n"
            f"Domain assignments (each task MUST use its assigned domain):\n"
            f"  {domain_assignments}\n\n"
            f"Difficulty rules:\n"
            f"  - NEVER mention method names in step descriptions. Describe intent only.\n"
            f"  - Steps must have cross-step co-reference: an object from step N reused in step N+2.\n"
            f"  - At least {n//2} tasks must include a step that requires filtering pins by direction\n"
            f"    (describe as 'keep only input/output pins' — do NOT say isInputSignal).\n"
            f"  - The final step of each task must call design.evalTclString(...) with a \n"
            f"    realistic Tcl command appropriate to the domain.\n"
            f"  - All object types and operations must be achievable with the OpenDB API table.\n\n"
            f"Return a JSON array of {n} objects with keys: "
            f"complex_prompt, step_1, step_2, step_3"
            + (", step_4" if difficulty >= 4 else "")
            + ".\nNo text outside the JSON array."
        )
        raw = self._call_llm([
            {"role": "system",  "content": self._system_prompt},
            {"role": "user",    "content": user_msg},
        ])
        return self._parse_tasks(raw, difficulty)

    def harden(self, task_dict: Dict, failure_log: Dict) -> List[Dict]:
        """Generate 2 harder variants of a failed task using the failure signal."""
        layer   = failure_log.get("layer_failed", 0)
        issues  = failure_log.get("issues", [])
        diffs   = failure_log.get("api_diffs", [])
        n_steps = sum(1 for k in task_dict if k.startswith("step_"))

        # Build failure summary for LLM-A
        failure_desc = f"Layer {layer} failure.\nIssues:\n"
        for iss in issues[:3]:
            failure_desc += f"  - {iss}\n"
        if diffs:
            failure_desc += "API mismatches:\n"
            for d in diffs[:3]:
                failure_desc += (
                    f"  - Edge [{d.get('src','')}]->[{d.get('tgt','')}]: "
                    f"code used {d.get('code_methods','?')}, "
                    f"correct is {d.get('rag_method','?')}\n"
                )

        target_steps = min(n_steps + 1, 4)

        user_msg = (
            f"The following task was attempted and FAILED:\n\n"
            f"Task: {json.dumps(task_dict, indent=2)}\n\n"
            f"Failure signal:\n{failure_desc}\n"
            f"Your job: generate 2 harder variants of this task that:\n"
            f"  1. Add one extra acquisition step that deepens the causal chain.\n"
            f"  2. Deliberately include the operation that caused the failure "
            f"     (so the solver must handle it correctly).\n"
            f"  3. Add a cross-step co-reference dependency not present in the original.\n"
            f"  4. All methods must exist in the OpenDB API table.\n\n"
            f"Each variant must have {target_steps} steps.\n"
            f"Return a JSON array of 2 objects with keys: "
            f"complex_prompt, step_1, step_2, step_3"
            + (", step_4" if target_steps >= 4 else "")
            + ".\nDo not include any text outside the JSON array."
        )
        raw = self._call_llm([
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_msg},
        ])
        return self._parse_tasks(raw, target_steps)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_type_hierarchy(self, rag_api_path: str) -> Dict[str, List[str]]:
        """Parse RAGAPIs.csv into {type: [method_signature, ...]} map."""
        df = pd.read_csv(rag_api_path)
        hierarchy: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            fn   = str(row.get("Function Name:", "")).strip()
            rt   = str(row.get("Return Type:", "")).strip()
            desc = str(row.get("Description:", "")).strip()
            if not fn or fn.lower() == "nan":
                continue
            # Extract owner type from function name: "odb.dbBlock.getNets(" → "odb.dbBlock"
            parts = fn.split(".")
            if len(parts) >= 3:
                owner = ".".join(parts[:2])   # e.g. "odb.dbBlock"
            elif len(parts) == 2:
                owner = parts[0]              # e.g. "openroad.Design"
            else:
                continue
            entry = fn.rstrip("(") + ("" if not rt or rt == "nan" else f" → {rt}")
            hierarchy.setdefault(owner, []).append(entry)
        return hierarchy

    def _make_system_prompt(self) -> str:
        # Build compact API table string
        api_lines = []
        for type_name, methods in sorted(self._type_hierarchy.items()):
            api_lines.append(f"\nType: {type_name}")
            for m in methods[:12]:   # cap per type to keep prompt manageable
                api_lines.append(f"  {m}")

        traps_lines = []
        for t in _TRAPS:
            traps_lines.append(
                f"  TRAP: {t['trap']}  →  CORRECT: {t['correct']}  "
                f"[{t['type']}]  USE IN: {t['use_in']}"
            )

        return (
            "You are LLM-A, the Adversary in an adversarial EDA task generation framework.\n\n"
            "Your role: generate multi-step OpenROAD Python tasks that stress-test a solver.\n\n"
            "== CRITICAL RULE: NEVER mention method names in task descriptions ==\n"
            "Task steps must describe INTENT and DATA OBJECTS only — not how to implement them.\n"
            "BAD:  'Filter ITerms by checking isOutputSignal() and get names via getMTerm().getName()'\n"
            "GOOD: 'From those pins, keep only the outputs and collect their signal names.'\n"
            "BAD:  'Call inst.getMaster().getName() to get the master cell name'\n"
            "GOOD: 'Find which cell type each instance uses and count by cell type.'\n\n"
            "== DOMAIN VARIETY: each task must use a DIFFERENT primary object type ==\n"
            "Rotate across: nets (dbNet/ITerms), instances (dbInst/master), pins (dbITerm/dbMTerm),\n"
            "placement (origin/status), timing (dbBlock/dbNet fanout), clock (isInClock), mixed.\n"
            "Do NOT generate two tasks with the same structure.\n\n"
            "== ADVERSARIAL TRAPS: design tasks that naturally lead solvers into these mistakes ==\n"
            + "\n".join(traps_lines) + "\n\n"
            "Example of a GOOD adversarial task (notice: no method names, cross-step dependency,\n"
            "forces solver to figure out direction-checking and pin naming independently):\n"
            "  step_1: Iterate all nets in the block and find the net connected to the most pins.\n"
            "  step_2: From that net, collect only the pins whose direction is INPUT and sort them\n"
            "          by the name of the instance they belong to.\n"
            "  step_3: Disconnect every INPUT pin after position 3 in that sorted list.\n"
            "  step_4: Call design.evalTclString(\"set_wire_rc -signal -resistance 0.01 -capacitance 0.02\") and print the result.\n\n"
            "OpenDB API Table (for your reference when designing valid task chains):\n"
            + "\n".join(api_lines) + "\n\n"
            "Output format: JSON array of task objects.\n"
            "Keys: complex_prompt, step_1, step_2, step_3 [, step_4]\n"
            "  complex_prompt : one natural-language sentence describing the overall goal\n"
            "  step_N         : intent + data objects only, NO method names, NO code\n\n"
            "Additional constraints:\n"
            "  - Steps must have cross-step co-reference: an object found in step N is used in step N+2.\n"
            "  - Every task must be solvable using only the API table above.\n"
            "  - Tasks must be physically meaningful on a placed/routed design.\n"
            "  - No two tasks may share the same primary object type or same action pattern.\n"
        )

    def _call_llm(self, messages: List[Dict], retries: int = 4) -> str:
        payload = json.dumps({
            "model":       self.model,
            "messages":    messages,
            "temperature": 0.9,
            "max_tokens":  4096,
        }).encode()
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = "https://api.openai.com/v1/chat/completions"
        wait = 10
        for attempt in range(retries):
            try:
                req  = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode())
                    return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"  [adversary] rate-limited, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    wait = min(wait * 2, 120)
                else:
                    raise
        raise RuntimeError("LLM-A: max retries exceeded")

    def _parse_tasks(self, raw: str, n_steps: int) -> List[Dict]:
        """Extract JSON array from LLM response, validate keys."""
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        # Find the first [...] block
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            tasks = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [adversary] JSON parse error: {e} — returning []", flush=True)
            return []
        if not isinstance(tasks, list):
            tasks = [tasks]
        required_keys = {"complex_prompt", "step_1", "step_2", "step_3"}
        if n_steps >= 4:
            required_keys.add("step_4")
        valid = []
        for t in tasks:
            if isinstance(t, dict) and required_keys.issubset(t.keys()):
                valid.append(t)
            else:
                print(f"  [adversary] skipping task missing keys: {t.keys()}", flush=True)
        return valid
