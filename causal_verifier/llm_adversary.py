"""
llm_adversary.py — LLM-A: Adversarial task generator (dataset5-style mutations).

Tasks follow a find → mutate → verify → report structure:
  Step 1: Query design to find a target object (most-used master, highest fanout net, etc.)
  Step 2: Mutate design state (setPlacementStatus, disconnect, swapMaster, connect to net)
  Step 3: Verify the mutation worked (re-query the same object and confirm change)
  Step 4: Report a side effect via design.evalTclString (clock, RC, timing)

GAN/Actor-Critic inspired pipeline:
  LLM-A (Adversary)  generates complex multi-step EDA mutation tasks.
  LLM-B (Solver)     attempts each task via the causal pipeline (verifier + controller).
  Critic signal      = LLM-B's failure logs (layer_failed, issues, api_diffs).
  Hardening          = LLM-A generates harder variants from failed tasks.

Two modes:
  generate_initial(n, difficulty) — cold start: N tasks at difficulty level 2-4
  harden(task_dict, failure_log)  — hot start: 2 harder variants from a failed task
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Dict, List
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Mutation-specific adversarial traps
# ─────────────────────────────────────────────────────────────────────────────

_TRAPS = [
    {
        "trap":    "inst.setPlacementStatus(FIRM) — passing bare word instead of string",
        "correct": "inst.setPlacementStatus('FIRM') — always pass a quoted string",
        "type":    "odb.dbInst",
        "use_in":  "tasks that set instance placement to FIRM, PLACED, or UNPLACED",
    },
    {
        "trap":    "inst.swapMaster('BUF_X2') — passing master name as string",
        "correct": "master = design.getDb().findMaster('BUF_X2'); inst.swapMaster(master)",
        "type":    "odb.dbInst",
        "use_in":  "tasks that swap instance master cells to a different variant",
    },
    {
        "trap":    "net.addITerm(iterm) or iterm.setNet(net) to connect a pin",
        "correct": "iterm.connect(net) — this is the only correct way to connect a pin to a net",
        "type":    "odb.dbITerm",
        "use_in":  "tasks that connect floating input pins to VSS or VDD",
    },
    {
        "trap":    "iterm.getDirection() to check if a pin is input or output",
        "correct": "iterm.isInputSignal() or iterm.isOutputSignal()",
        "type":    "odb.dbITerm",
        "use_in":  "tasks that filter pins by direction before mutating them",
    },
    {
        "trap":    "inst.getMasterName() to get the cell type name",
        "correct": "inst.getMaster().getName()",
        "type":    "odb.dbInst",
        "use_in":  "tasks that group or filter instances by master cell name",
    },
    {
        "trap":    "net.getFanout() to count connected pins",
        "correct": "len(list(net.getITerms())) — iterate and count ITerms",
        "type":    "odb.dbNet",
        "use_in":  "tasks that find highest fanout nets",
    },
    {
        "trap":    "design.evalTclString('set_wire_rc') without -signal or -clock flag",
        "correct": "evalTclString('set_wire_rc -signal -resistance R -capacitance C') or -clock",
        "type":    "openroad.Design",
        "use_in":  "tasks that configure wire parasitics after a mutation",
    },
    {
        "trap":    "block.findNet('VSS') may return None — not checking before connect",
        "correct": "always check: vss = block.findNet('VSS'); assert vss is not None",
        "type":    "odb.dbBlock",
        "use_in":  "tasks that connect floating pins to power/ground nets",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Mutation task domains — mirrors dataset5 structure
# ─────────────────────────────────────────────────────────────────────────────

_DOMAINS = [
    "most_used_master+FIRM_placement",
    "highest_fanout_net+disconnect_inputs",
    "most_floating_inputs+connect_to_VSS",
    "most_output_pins+FIRM+clock_check",
    "master_swap+upsized_variant",
    "highest_Y_instance+FIRM+clock_assign",
    "most_connected_instance+FIRM+clock_propagate",
    "non_power_net+disconnect+clock_assign",
    "BUF_prefix_instances+FIRM+wire_RC",
    "lowest_X_instance+FIRM+clock_check",
    "second_most_used_master+FIRM+CTS_buffer",
    "net_touching_most_instances+prune+propagate",
]


class LLMAdversary:
    """LLM-A: generates and hardens adversarial dataset5-style mutation tasks."""

    def __init__(self, api_key: str, model: str, rag_api_path: str):
        self.api_key = api_key
        self.model   = model
        self._type_hierarchy = self._build_type_hierarchy(rag_api_path)
        self._system_prompt  = self._make_system_prompt()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_initial(self, n: int = 10, difficulty: int = 4) -> List[Dict]:
        """Generate n dataset5-style adversarial mutation tasks."""
        domain_assignments = ", ".join(
            f"task {i+1}={_DOMAINS[i % len(_DOMAINS)]}" for i in range(n)
        )
        user_msg = (
            f"Generate exactly {n} adversarial EDA mutation tasks.\n"
            f"Each task must have exactly {difficulty} sequential steps.\n\n"
            f"Domain assignments (each task MUST use its assigned domain):\n"
            f"  {domain_assignments}\n\n"
            f"MANDATORY structure for every task:\n"
            f"  step_1: FIND — query the design to locate a specific target object\n"
            f"          (e.g. instance with most ITerms, net with highest fanout,\n"
            f"           instances matching a master name prefix)\n"
            f"  step_2: MUTATE — change the design state for the found object(s)\n"
            f"          (e.g. set placement status, disconnect pins, connect to net,\n"
            f"           swap master cell)\n"
            f"  step_3: VERIFY — re-query the same object and confirm the mutation\n"
            f"          (e.g. check placement status changed, pin is now connected,\n"
            f"           new master name matches expected)\n"
            f"  step_4: REPORT — use design.evalTclString to report a side effect\n"
            f"          (clock membership, wire RC setup, clock propagation,\n"
            f"           CTS buffer registration)\n\n"
            f"Difficulty rules:\n"
            f"  - NEVER mention method names in step descriptions. Describe intent only.\n"
            f"  - step_1 must use a non-trivial selection criterion (not just first instance).\n"
            f"  - step_2 must mutate MORE than one object when the domain calls for it\n"
            f"    (e.g. first 3 instances of that master, all INPUT pins beyond index 5).\n"
            f"  - step_3 must confirm the mutation by re-reading the changed property.\n"
            f"  - step_4 must use design.evalTclString with a realistic Tcl command.\n"
            f"  - Cross-step co-reference: the object found in step_1 must be reused\n"
            f"    in both step_2 and step_3.\n\n"
            f"Return a JSON array of {n} objects with keys: "
            f"complex_prompt, step_1, step_2, step_3"
            + (", step_4" if difficulty >= 4 else "")
            + ".\nNo text outside the JSON array."
        )
        raw = self._call_llm([
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_msg},
        ])
        return self._parse_tasks(raw, difficulty)

    def harden(self, task_dict: Dict, failure_log: Dict) -> List[Dict]:
        """Generate 2 harder variants of a failed task using the failure signal."""
        layer  = failure_log.get("layer_failed", 0)
        issues = failure_log.get("issues", [])
        diffs  = failure_log.get("api_diffs", [])
        n_steps = sum(1 for k in task_dict if k.startswith("step_"))

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
            f"The following mutation task was attempted and FAILED:\n\n"
            f"Task: {json.dumps(task_dict, indent=2)}\n\n"
            f"Failure signal:\n{failure_desc}\n"
            f"Your job: generate 2 harder variants that:\n"
            f"  1. Chain two mutations — the result of the first mutation\n"
            f"     becomes the selection criterion for the second mutation.\n"
            f"  2. Deliberately include the operation that caused the failure\n"
            f"     so the solver must handle it correctly.\n"
            f"  3. Add a conditional mutation — only mutate objects that pass\n"
            f"     an additional filter (e.g. only instances NOT already FIRM,\n"
            f"     only pins that are currently floating).\n"
            f"  4. All methods must exist in the API table.\n\n"
            f"Keep the find → mutate → verify → report structure.\n"
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
            fn  = str(row.get("Function Name:", "")).strip()
            rt  = str(row.get("Return Type:", "")).strip()
            if not fn or fn.lower() == "nan":
                continue
            parts = fn.split(".")
            if len(parts) >= 3:
                owner = ".".join(parts[:2])
            elif len(parts) == 2:
                owner = parts[0]
            else:
                continue
            entry = fn.rstrip("(") + ("" if not rt or rt == "nan" else f" → {rt}")
            hierarchy.setdefault(owner, []).append(entry)
        return hierarchy

    def _make_system_prompt(self) -> str:
        # Split into read methods and write/mutation methods
        read_lines  = []
        write_lines = []
        write_keywords = {"set", "connect", "disconnect", "swap", "create",
                          "remove", "add", "place", "evalTcl"}

        for type_name, methods in sorted(self._type_hierarchy.items()):
            read_entries  = []
            write_entries = []
            for m in methods[:15]:
                name_part = m.split(".")[-1].lower()
                if any(name_part.startswith(w) for w in write_keywords):
                    write_entries.append(f"  {m}")
                else:
                    read_entries.append(f"  {m}")

            if read_entries:
                read_lines.append(f"\nType: {type_name}")
                read_lines.extend(read_entries)
            if write_entries:
                write_lines.append(f"\nType: {type_name}")
                write_lines.extend(write_entries)

        traps_lines = []
        for t in _TRAPS:
            traps_lines.append(
                f"  TRAP: {t['trap']}\n"
                f"  CORRECT: {t['correct']}\n"
                f"  [{t['type']}] USE IN: {t['use_in']}\n"
            )

        return (
            "You are LLM-A, the Adversary in a dataset5-style adversarial EDA task generation framework.\n\n"
            "Your role: generate multi-step OpenROAD tasks that MUTATE design state and\n"
            "stress-test a solver's ability to correctly find, change, verify, and report\n"
            "on design objects.\n\n"
            "== TASK STRUCTURE (mandatory: find → mutate → verify → report) ==\n"
            "  step_1: FIND   — locate the target object using a non-trivial criterion\n"
            "  step_2: MUTATE — change its design state (placement, connectivity, master)\n"
            "  step_3: VERIFY — re-read the changed property and confirm the mutation\n"
            "  step_4: REPORT — call design.evalTclString with a clock/RC/timing command\n\n"
            "== CRITICAL RULE: NEVER mention method names in task descriptions ==\n"
            "Describe INTENT and DATA OBJECTS only — not how to implement.\n"
            "BAD:  'Call inst.setPlacementStatus(\"FIRM\") on each instance'\n"
            "GOOD: 'Set the placement status of those instances to fixed/firm'\n"
            "BAD:  'Use iterm.connect(vss_net) to tie floating pins'\n"
            "GOOD: 'Connect all floating input pins on that instance to the ground net'\n\n"
            "== EXAMPLE dataset5-style task ==\n"
            "  complex_prompt: Find the most-used master cell, set the first three\n"
            "                  instances to FIRM placement, then check clock membership.\n"
            "  step_1: Count how many instances use each master cell type and identify\n"
            "          the master cell that appears most frequently in the design.\n"
            "  step_2: From all instances using that master cell, take the first three\n"
            "          and change their placement status to fixed/firm.\n"
            "  step_3: Re-read the placement status of those three instances and confirm\n"
            "          all three are now reported as firmly placed.\n"
            "  step_4: For each of those three instances, check whether it belongs to\n"
            "          the clock tree and report the result using design.evalTclString.\n\n"
            "== ADVERSARIAL TRAPS: design tasks that lead solvers into these mistakes ==\n"
            + "\n".join(traps_lines) + "\n"
            "== READ API (use for step_1 and step_3 — finding and verifying) ==\n"
            + "\n".join(read_lines) + "\n\n"
            "== MUTATION API (use for step_2 — changing design state) ==\n"
            + "\n".join(write_lines) + "\n\n"
            "Output format: JSON array of task objects.\n"
            "Keys: complex_prompt, step_1, step_2, step_3 [, step_4]\n"
            "  complex_prompt : one natural-language sentence describing the overall goal\n"
            "  step_N         : intent + data objects only, NO method names, NO code\n\n"
            "Additional constraints:\n"
            "  - step_1 selection criterion must be non-trivial (max, min, prefix filter,\n"
            "    count-based ranking — not just 'get the first instance').\n"
            "  - step_2 must mutate a specific subset (first N, all matching filter, etc.).\n"
            "  - step_3 must re-read the exact property changed in step_2.\n"
            "  - step_4 evalTclString command must be physically meaningful.\n"
            "  - No two tasks may share the same mutation operation.\n"
            "  - Tasks must be executable on a placed/routed design.\n"
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

    def generate_variants(self, seed_prompt: str, n_variants: int = 3,
                          target_steps: int = 4) -> List[Dict]:
        """Generate n_variants harder but causally valid variants of a seed prompt.

        Each call produces a small batch (3-5 tasks) so the output easily fits
        within the 4096-token max_tokens budget — no truncation risk.

        Args:
            seed_prompt:  The complex_prompt string from the seed task.
            n_variants:   Number of harder variants to generate (default 3).
            target_steps: Number of steps in each generated task (default 4).
        """
        step_keys = ", ".join(f"step_{i}" for i in range(1, target_steps + 1))
        user_msg = (
            f"Given this seed OpenROAD task:\n\n\"{seed_prompt}\"\n\n"
            f"Generate {n_variants} harder but causally valid variants using these strategies:\n"
            f"  Strategy A — more subgoals: increase dependent steps to {target_steps}, "
            f"each step's output feeding the next.\n"
            f"  Strategy B — branching dependencies: add a step that depends on results "
            f"from two prior steps simultaneously (e.g. find instances AND nets, then act "
            f"on the combination).\n"
            f"  Strategy C — cross-stage actions: span multiple design stages in one task "
            f"(e.g. query placed instances, check routing connectivity, report timing).\n"
            f"  Strategy D — combined subtasks: merge two related but distinct operations "
            f"(e.g. find most-used master AND highest-fanout net, then mutate both).\n\n"
            f"Hard rules:\n"
            f"  - NEVER mention method names — describe intent and data objects only.\n"
            f"  - All steps must be executable on a placed/routed design.\n"
            f"  - Objects found in earlier steps must be reused in later steps.\n"
            f"  - Each variant must be genuinely different from the seed and from each other.\n"
            f"  - Each variant must have exactly {target_steps} steps.\n\n"
            f"Return a JSON array of {n_variants} objects.\n"
            f"Keys: complex_prompt, {step_keys}.\n"
            f"No text outside the JSON array."
        )
        raw = self._call_llm([
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_msg},
        ])
        return self._parse_tasks(raw, target_steps)

    def _parse_tasks(self, raw: str, n_steps: int) -> List[Dict]:
        """Extract JSON array from LLM response, validate keys."""
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
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
