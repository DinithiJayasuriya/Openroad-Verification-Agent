"""test_causal.py — Unit tests for the causal chain pipeline.

Run:
  cd /mnt/ssd1/dinithi/OpenROAD-Agent
  python -m causal_verifier.test_causal
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_verifier.chain_extractor  import CausalChainExtractor
from causal_verifier.node_retriever   import NodeRetriever, _normalize
from causal_verifier.chain_validator  import CausalChainValidator
from causal_verifier.chain_cache      import ChainCache
from causal_verifier.pipeline         import CausalPipeline

RAG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "RAGData", "RAGAPIs.csv")

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"

def run(name, cond, detail=""):
    tag = _PASS if cond else _FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


# ══════════════════════════════════════════════════════════════════════════════
# 1. Chain extraction
# ══════════════════════════════════════════════════════════════════════════════

def test_chain_extractor():
    print("\n── CausalChainExtractor ──")
    ok = 0
    ex = CausalChainExtractor()   # no LLM key

    cases = [
        ("Find instance 486",
         ["openroad.Design", "odb.dbBlock", "odb.dbInst"]),
        ("get all nets in the design",
         ["openroad.Design", "odb.dbBlock", "odb.dbNet"]),
        ("print top-level port names",
         ["openroad.Design", "odb.dbBlock", "odb.dbBTerm"]),
        ("run global placement",
         ["openroad.Design", "gpl.Replace"]),
        ("set up floorplan utilization",
         ["openroad.Design", "ifp.InitFloorplan"]),
        ("run clock tree synthesis",
         ["openroad.Design", "cts.TritonCTS"]),
        ("check timing slack for all paths",
         ["openroad.Design", "openroad.Timing"]),
        ("get the metal layer M1",
         ["openroad.Design", "odb.dbBlock", "odb.dbTech", "odb.dbTechLayer"]),
        ("place I/O pins on die boundary",
         ["openroad.Design", "ppl.IOPlacer"]),
    ]

    for task, expected in cases:
        chain = ex.extract(task)
        ok += run(f"  '{task}'",
                  chain.type_names == expected,
                  f"got {chain.type_names}")
    print(f"  Passed {ok}/{len(cases)}")
    return ok, len(cases)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Node retriever
# ══════════════════════════════════════════════════════════════════════════════

def test_node_retriever():
    print("\n── NodeRetriever ──")
    ok = 0

    retriever = NodeRetriever(RAG_PATH)
    from causal_verifier.chain_extractor import CausalChain, ChainNode as N

    cases = [
        # (chain_type_names, expected_methods_per_edge)
        (["openroad.Design", "odb.dbBlock", "odb.dbInst"],
         ["getBlock", "findInst"]),
        (["openroad.Design", "odb.dbBlock", "odb.dbNet"],
         ["getBlock", "getNets"]),     # CSV: getNets (list) preferred over supplement findNet
        (["openroad.Design", "odb.dbBlock", "odb.dbBTerm"],
         ["getBlock", "getBTerms"]),   # CSV: getBTerms (list) preferred over supplement findBTerm
        (["openroad.Design", "gpl.Replace"],
         ["getReplace"]),
        (["openroad.Design", "ifp.InitFloorplan"],
         ["getFloorplan"]),
        (["openroad.Design", "cts.TritonCTS"],
         ["getTritonCts"]),
        (["openroad.Design", "odb.dbBlock", "odb.dbTech", "odb.dbTechLayer"],
         ["getBlock", "getTech", "findLayer"]),
    ]

    for type_names, expected_methods in cases:
        nodes = [N.from_type(t) for t in type_names]
        chain = CausalChain(nodes=nodes, task="test", extraction_method="rule",
                            confidence=0.9)
        apis  = retriever.get_chain_apis(chain)
        methods = [a.method_name for a in apis]
        # Allow partial match: just check expected methods appear in order
        match = all(m in methods for m in expected_methods)
        ok += run(
            f"  chain {' → '.join(t.split('.')[-1] for t in type_names)}",
            match,
            f"got {methods}, expected {expected_methods}",
        )
    print(f"  Passed {ok}/{len(cases)}")
    return ok, len(cases)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Causal validator
# ══════════════════════════════════════════════════════════════════════════════

def test_chain_validator():
    print("\n── CausalChainValidator ──")
    ok = 0

    from causal_verifier.chain_extractor import CausalChain, ChainNode as N
    from causal_verifier.node_retriever  import NodeAPIEntry

    validator = CausalChainValidator()

    def make_chain(*types):
        nodes = [N.from_type(t) for t in types]
        return CausalChain(nodes=nodes, task="t", extraction_method="rule",
                           confidence=0.9)

    def make_api(src, tgt, method, params=""):
        return NodeAPIEntry(source_type=src, target_type=tgt,
                            method_name=method, full_signature=f"{src}.{method}(",
                            params=params, description="test")

    # ── Test 1: correct code passes ──
    code1 = """
design = openroad.Design(tech)
block = design.getBlock()
inst = block.findInst("u_cpu")
print(inst.getName())
"""
    chain1   = make_chain("openroad.Design", "odb.dbBlock", "odb.dbInst")
    apis1    = [make_api("openroad.Design", "odb.dbBlock", "getBlock"),
                make_api("odb.dbBlock",     "odb.dbInst",  "findInst", "str(name)")]
    result1  = validator.validate(code1, chain1, apis1)
    ok += run("correct code passes", result1.passed, str(result1))

    # ── Test 2: skipped getBlock → fail ──
    code2 = """
design = openroad.Design(tech)
inst = design.findInst("u_cpu")   # wrong: findInst is on dbBlock, not Design
"""
    result2 = validator.validate(code2, chain1, apis1)
    ok += run("skipped getBlock → FAIL", not result2.passed,
              f"issues={result2.issues[:1]}")

    # ── Test 3: correct net chain ──
    code3 = """
block = design.getBlock()
nets = block.getNets()
for net in nets:
    print(net.getName())
"""
    chain3 = make_chain("openroad.Design", "odb.dbBlock", "odb.dbNet")
    apis3  = [make_api("openroad.Design", "odb.dbBlock", "getBlock"),
              make_api("odb.dbBlock",     "odb.dbNet",   "getNets")]
    result3 = validator.validate(code3, chain3, apis3)
    ok += run("correct net chain passes", result3.passed, str(result3))

    # ── Test 4: syntax error caught ──
    code4 = "def foo(\n  x ="
    result4 = validator.validate(code4, chain1, apis1)
    ok += run("syntax error → FAIL", not result4.passed, result4.issues[:1])

    # ── Test 5: flow-tool chain ──
    code5 = """
placer = design.getReplace()
placer.doNesterovPlace()
"""
    chain5 = make_chain("openroad.Design", "gpl.Replace")
    apis5  = [make_api("openroad.Design", "gpl.Replace", "getReplace")]
    result5 = validator.validate(code5, chain5, apis5)
    ok += run("flow-tool (Replace) chain passes", result5.passed, str(result5))

    print(f"  Passed {ok}/5")
    return ok, 5


# ══════════════════════════════════════════════════════════════════════════════
# 4. Chain cache and borrowing
# ══════════════════════════════════════════════════════════════════════════════

def test_chain_cache():
    print("\n── ChainCache ──")
    ok = 0

    from causal_verifier.chain_extractor import CausalChain, ChainNode as N
    from causal_verifier.node_retriever  import NodeAPIEntry

    cache = ChainCache()   # in-memory only

    def make_chain(*types):
        nodes = [N.from_type(t) for t in types]
        return CausalChain(nodes=nodes, task="t", extraction_method="rule",
                           confidence=0.9)

    def make_api(src, tgt, method):
        return NodeAPIEntry(source_type=src, target_type=tgt,
                            method_name=method, full_signature=f"{src}.{method}(",
                            params="", description="test")

    # Record a successful chain for "Find instance"
    chain_inst = make_chain("openroad.Design", "odb.dbBlock", "odb.dbInst")
    apis_inst  = [make_api("openroad.Design", "odb.dbBlock", "getBlock"),
                  make_api("odb.dbBlock",     "odb.dbInst",  "findInst")]
    cache.record_success("case_3", "Find inst u1", chain_inst, apis_inst)

    ok += run("edge count after record", cache.edge_count() == 2,
              f"got {cache.edge_count()}")

    # New chain that shares the Design→Block edge
    chain_net = make_chain("openroad.Design", "odb.dbBlock", "odb.dbNet")
    borrowed  = cache.find_matching_edges(chain_net)
    ok += run("Design→Block edge borrowed for net chain",
              ("openroad.Design", "odb.dbBlock") in borrowed
              or ("openroad.Design", "odb.dbBlock") in
                  {(k[0], k[1]) for k in borrowed},
              f"borrowed keys: {list(borrowed.keys())}")

    # The borrowed entry should be marked as "cache"
    if borrowed:
        entry = list(borrowed.values())[0]
        ok += run("borrowed entry has source='cache'",
                  entry.source == "cache", f"got {entry.source}")
    else:
        ok += run("borrowed entry has source='cache'", False, "no borrowed entries")

    # has_edge check
    ok += run("has_edge Design→Block", cache.has_edge("openroad.Design", "odb.dbBlock"))
    ok += run("has_edge Block→Net not cached yet",
              not cache.has_edge("odb.dbBlock", "odb.dbNet"))

    print(f"  Passed {ok}/5")
    return ok, 5


# ══════════════════════════════════════════════════════════════════════════════
# 5. End-to-end pipeline
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_e2e():
    print("\n── CausalPipeline (end-to-end) ──")
    ok = 0

    pipeline = CausalPipeline(rag_api_path=RAG_PATH)

    # ── plan ──
    result = pipeline.plan("Find instance named u_cpu in the design")
    ok += run("plan returns chain",
              result.chain.terminal_type == "odb.dbInst",
              f"terminal={result.chain.terminal_type}")
    ok += run("constraint_prompt non-empty",
              len(result.constraint_prompt) > 50)
    ok += run("MANDATORY in prompt",
              "MANDATORY" in result.constraint_prompt)

    # ── validate correct code ──
    good_code = """
block = design.getBlock()
inst  = block.findInst("u_cpu")
print(inst.getName())
"""
    val = pipeline.validate("Find instance named u_cpu", good_code, result)
    ok += run("good code → validation passes", val.passed, str(val))

    # ── record and borrow ──
    pipeline.record_success("case_1", "Find instance named u_cpu", result)
    ok += run("cache has Design→Block after record",
              pipeline.cache.has_edge("openroad.Design", "odb.dbBlock"))

    # New task that shares the Design→Block edge
    result2 = pipeline.plan("Get all nets in the design")
    ok += run("Design→Block borrowed in net chain",
              result2.has_cached_edges,
              f"borrowed={result2.borrowed_edges}")

    ok += run("CACHED in net constraint prompt",
              "CACHED" in result2.constraint_prompt,
              result2.constraint_prompt[:300])

    # ── validate wrong code ──
    bad_code = """
db = design.getDb()
inst = db.findInst("u_cpu")   # wrong: findInst is not on dbDatabase
"""
    val_bad = pipeline.validate("Find instance named u_cpu", bad_code, result)
    ok += run("bad code (wrong receiver) → validation fails",
              not val_bad.passed, str(val_bad))

    print(f"  Passed {ok}/8")
    return ok, 8


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Causal Chain Pipeline — Test Suite")
    print("=" * 60)

    total_ok = total_n = 0
    for fn in [test_chain_extractor, test_node_retriever,
               test_chain_validator, test_chain_cache, test_pipeline_e2e]:
        ok, n = fn()
        total_ok += ok
        total_n  += n

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_ok}/{total_n} passed")
    if total_ok == total_n:
        print("\033[32mAll tests passed.\033[0m")
    else:
        print(f"\033[31m{total_n - total_ok} test(s) failed.\033[0m")
    print("=" * 60)
    return total_ok == total_n


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
