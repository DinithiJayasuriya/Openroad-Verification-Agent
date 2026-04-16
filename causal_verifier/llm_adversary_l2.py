"""
llm_adversary.py — LLM-A: Adversarial task generator (Level 2).

Level 2 tasks follow a before→invoke→after→compare structure:
  Step 1: Query design state BEFORE using odb/Timing API
  Step 2: Invoke a flow tool to change the design
  Step 3: Query design state AFTER the tool ran
  Step 4: Compare before/after and report what changed

GAN/Actor-Critic inspired pipeline:
  LLM-A (Adversary)  generates complex multi-step EDA tasks grounded in real flow tools.
  LLM-B (Solver)     attempts each task via the causal pipeline (verifier + controller).
  Critic signal      = LLM-B's failure logs (layer_failed, issues, api_diffs).
  Hardening          = LLM-A generates harder variants adding cross-tool dependencies.

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
# Level 2 adversarial traps — common mistakes when invoking flow tools
# ─────────────────────────────────────────────────────────────────────────────

_TRAPS = [
    {
        "trap":    "gpl.Replace.doNesterovPlace() before doInitialPlace()",
        "correct": "always call doInitialPlace() first, then doNesterovPlace()",
        "type":    "gpl.Replace",
        "use_in":  "tasks that run global placement",
    },
    {
        "trap":    "design.getTiming() — Timing is constructed as openroad.Timing(design)",
        "correct": "timing = openroad.Timing(design)  — it is a constructor, not a getter",
        "type":    "openroad.Timing",
        "use_in":  "tasks that query pin slack, arrival, or power after a flow step",
    },
    {
        "trap":    "psm.PDNSim.analyzePowerGrid() without calling setNet() first",
        "correct": "call pdnsim.setNet(net) before pdnsim.analyzePowerGrid()",
        "type":    "psm.PDNSim",
        "use_in":  "tasks that analyze IR drop on a power net",
    },
    {
        "trap":    "grt.GlobalRouter.globalRoute() without setting layer bounds first",
        "correct": "call setMinRoutingLayer() and setMaxRoutingLayer() before globalRoute()",
        "type":    "grt.GlobalRouter",
        "use_in":  "tasks that run global routing",
    },
    {
        "trap":    "cts.TritonCTS.runTritonCts() without setBufferList() and setRootBuffer()",
        "correct": "configure buffer list and root buffer before runTritonCts()",
        "type":    "cts.TritonCTS",
        "use_in":  "tasks that run clock tree synthesis",
    },
    {
        "trap":    "pdn.PdnGen.buildGrids() without setCoreDomain() first",
        "correct": "call setCoreDomain() before buildGrids(), then writeToDb() to commit",
        "type":    "pdn.PdnGen",
        "use_in":  "tasks that build power delivery networks",
    },
    {
        "trap":    "dpl.Opendp.detailedPlacement() without running global placement first",
        "correct": "detailed placement must follow global placement (gpl.Replace)",
        "type":    "dpl.Opendp",
        "use_in":  "tasks that run detailed placement or check legalization",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 flow tool domains
# ─────────────────────────────────────────────────────────────────────────────

_DOMAINS = [
    "global_placement+density",
    "detailed_placement+legality",
    "clock_tree_synthesis+skew",
    "global_routing+overflow",
    "power_delivery+IR_drop",
    "timing_analysis+slack",
    "floorplan+IO_placement",
    "macro_placement+wirelength",
]


class LLMAdversary:
    """LLM-A: generates and hardens adversarial Level-2 EDA tasks."""

    def __init__(self, api_key: str, model: str, rag_api_path: str):
        self.api_key = api_key
        self.model   = model
        self._flow_api   = self._build_flow_api(rag_api_path)
        self._type_hierarchy = self._build_type_hierarchy(rag_api_path)
        self._system_prompt  = self._make_system_prompt()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_initial(self, n: int = 10, difficulty: int = 4) -> List[Dict]:
        """Generate n Level-2 adversarial tasks at the given step-count difficulty."""
        domain_assignments = ", ".join(
            f"task {i+1}={_DOMAINS[i % len(_DOMAINS)]}" for i in range(n)
        )
        user_msg = (
            f"Generate exactly {n} adversarial Level-2 EDA tasks.\n"
            f"Each task must have exactly {difficulty} sequential steps.\n\n"
            f"Domain assignments (each task MUST use its assigned domain):\n"
            f"  {domain_assignments}\n\n"
            f"MANDATORY structure for every task:\n"
            f"  step_1: Query design state BEFORE the flow tool runs\n"
            f"          (use odb.* or openroad.Timing API to measure something)\n"
            f"  step_2: Invoke the flow tool for the assigned domain\n"
            f"          (configure required parameters first, then run it)\n"
            f"  step_3: Query design state AFTER the tool ran\n"
            f"          (measure the same or related metric as step_1)\n"
            f"  step_4: Compare before/after values and report the delta\n"
            f"          (use design.evalTclString(...) for the final report)\n\n"
            f"Difficulty rules:\n"
            f"  - NEVER mention method names in step descriptions. Describe intent only.\n"
            f"  - step_1 must measure something specific (e.g. unplaced count, WNS, fanout).\n"
            f"  - step_2 must configure the tool correctly before invoking it.\n"
            f"  - step_3 must re-measure the same metric so a delta can be computed.\n"
            f"  - step_4 must use the before/after values from steps 1 and 3.\n"
            f"  - Cross-step co-reference: object or value from step 1 reused in step 4.\n\n"
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
            f"The following Level-2 task was attempted and FAILED:\n\n"
            f"Task: {json.dumps(task_dict, indent=2)}\n\n"
            f"Failure signal:\n{failure_desc}\n"
            f"Your job: generate 2 harder variants that:\n"
            f"  1. Add a cross-tool dependency — the result from one flow tool\n"
            f"     feeds as input to a second flow tool.\n"
            f"  2. Deliberately include the operation that caused the failure\n"
            f"     so the solver must handle it correctly this time.\n"
            f"  3. Deepen the before/after comparison — require computing a ratio\n"
            f"     or percentage improvement, not just a raw delta.\n"
            f"  4. All methods must exist in the flow tool API table.\n\n"
            f"Keep the mandatory before→invoke→after→compare structure.\n"
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

    def _build_flow_api(self, rag_api_path: str) -> Dict[str, List[str]]:
        """Parse RAGAPIs.csv — return only flow tool methods (non-odb, non-openroad.Design)."""
        df = pd.read_csv(rag_api_path)
        flow_tools = {"gpl", "dpl", "grt", "drt", "cts", "pdn", "ppl", "ifp", "mpl", "psm"}
        api: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            fn  = str(row.get("Function Name:", "")).strip()
            rt  = str(row.get("Return Type:", "")).strip()
            if not fn or fn.lower() == "nan":
                continue
            ns = fn.split(".")[0]
            if ns not in flow_tools:
                continue
            parts = fn.split(".")
            owner = ".".join(parts[:2]) if len(parts) >= 3 else parts[0]
            entry = fn.rstrip("(") + ("" if not rt or rt == "nan" else f" → {rt}")
            api.setdefault(owner, []).append(entry)
        return api

    def _build_type_hierarchy(self, rag_api_path: str) -> Dict[str, List[str]]:
        """Parse RAGAPIs.csv into {type: [method_signature, ...]} map (all types)."""
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
        # Flow tool API section
        flow_lines = []
        for type_name, methods in sorted(self._flow_api.items()):
            flow_lines.append(f"\nFlow Tool: {type_name}")
            for m in methods:
                flow_lines.append(f"  {m}")

        # odb + Timing API section (for before/after queries)
        odb_lines = []
        for type_name, methods in sorted(self._type_hierarchy.items()):
            if type_name.startswith("odb.") or type_name.startswith("openroad."):
                odb_lines.append(f"\nType: {type_name}")
                for m in methods[:10]:
                    odb_lines.append(f"  {m}")

        traps_lines = []
        for t in _TRAPS:
            traps_lines.append(
                f"  TRAP: {t['trap']}  →  CORRECT: {t['correct']}  "
                f"[{t['type']}]  USE IN: {t['use_in']}"
            )

        return (
            "You are LLM-A, the Adversary in a Level-2 adversarial EDA task generation framework.\n\n"
            "Your role: generate multi-step OpenROAD tasks that invoke FLOW TOOLS and stress-test\n"
            "a solver's ability to correctly configure, run, and measure the result of each tool.\n\n"
            "== LEVEL 2 TASK STRUCTURE (mandatory) ==\n"
            "Every task must follow this before→invoke→after→compare pattern:\n"
            "  step_1: Measure a design metric BEFORE the flow tool runs (odb/Timing API)\n"
            "  step_2: Configure and invoke the flow tool for the assigned domain\n"
            "  step_3: Re-measure the same metric AFTER the tool ran\n"
            "  step_4: Compute and report the delta (before vs after) via evalTclString\n\n"
            "== CRITICAL RULE: NEVER mention method names in task descriptions ==\n"
            "Describe INTENT and DATA OBJECTS only — not how to implement.\n"
            "BAD:  'Call gpl.Replace.doInitialPlace() then doNesterovPlace()'\n"
            "GOOD: 'Run the global placement engine with a target density of 0.7'\n"
            "BAD:  'Use openroad.Timing(design).getPinSlack() on each endpoint'\n"
            "GOOD: 'Measure the worst setup slack across all timing endpoints'\n\n"
            "== ADVERSARIAL TRAPS: design tasks that lead solvers into these mistakes ==\n"
            + "\n".join(traps_lines) + "\n\n"
            "== EXAMPLE Level-2 task ==\n"
            "  complex_prompt: Run global placement and measure how the number of\n"
            "                  illegal instances changes after placement.\n"
            "  step_1: Count all instances that are currently unplaced or illegally\n"
            "          placed, and record the total as the baseline.\n"
            "  step_2: Configure the global placer with a target utilization of 0.7\n"
            "          and run the full placement pass.\n"
            "  step_3: After placement completes, recount the instances that remain\n"
            "          unplaced or have an illegal placement status.\n"
            "  step_4: Compute the percentage reduction in illegal instances and\n"
            "          report it using design.evalTclString.\n\n"
            "== FLOW TOOL API (use these for step_2 — configuration + invocation) ==\n"
            + "\n".join(flow_lines) + "\n\n"
            "== ODB / TIMING API (use these for step_1 and step_3 — before/after queries) ==\n"
            + "\n".join(odb_lines) + "\n\n"
            "Output format: JSON array of task objects.\n"
            "Keys: complex_prompt, step_1, step_2, step_3 [, step_4]\n"
            "  complex_prompt : one natural-language sentence describing the overall goal\n"
            "  step_N         : intent + data objects only, NO method names, NO code\n\n"
            "Additional constraints:\n"
            "  - step_2 MUST configure required parameters before invoking the tool.\n"
            "  - step_3 MUST measure something that can be compared to step_1.\n"
            "  - step_4 MUST use values from both step_1 and step_3.\n"
            "  - No two tasks may use the same flow tool or same measurement metric.\n"
            "  - Tasks must be physically meaningful on a placed/routed design.\n"
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
