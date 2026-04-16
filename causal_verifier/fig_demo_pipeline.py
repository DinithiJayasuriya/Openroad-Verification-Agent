#!/usr/bin/env python3
"""fig_demo_pipeline.py — Paper-ready demo figure for the Causal Agentic Pipeline.

Output: fig_demo_pipeline.pdf + .png
Usage:  python3 causal_verifier_4_2/fig_demo_pipeline.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Colours ─────────────────────────────────────────────────────────────────
C = dict(
    INPUT="#EBF5FB", LLM="#FEF9E7", RAG="#E8F8F5", VERIFY="#FDEDEC",
    CTRL="#F4ECF7", PASS="#D5F5E3", BORDER="#AEB6BF", TITLE="#2C3E50",
    RED="#E74C3C", GREEN="#27AE60", ARROW="#5D6D7E",
)

# ── Canvas ──────────────────────────────────────────────────────────────────
W_PAGE, H_PAGE = 16, 28
fig, ax = plt.subplots(figsize=(W_PAGE, H_PAGE))
ax.set_xlim(0, W_PAGE)
ax.set_ylim(0, H_PAGE)
ax.axis("off")

X0 = 0.5          # left margin
BW = 14.8         # box width
CX = X0 + BW / 2  # centre x
LX = X0 + 0.5     # left-col text x
RX = X0 + BW / 2 + 0.4  # right-col text x
LINE = 0.24        # line spacing
FS = 7.2           # body font
FST = 9.5          # title font
GAP = 0.28         # gap between boxes

# ── Helpers ─────────────────────────────────────────────────────────────────

def _box(y, h, bg):
    ax.add_patch(FancyBboxPatch(
        (X0, y - h), BW, h, boxstyle="round,pad=0.15",
        lw=1.0, edgecolor=C["BORDER"], facecolor=bg))
    return y - h

def _title(y, text, bg=None):
    bg = bg or C["TITLE"]
    ax.add_patch(FancyBboxPatch(
        (X0 + 0.15, y - 0.40), BW - 0.3, 0.34,
        boxstyle="round,pad=0.05", lw=0, facecolor=bg))
    ax.text(CX, y - 0.23, text, ha="center", va="center",
            fontsize=FST, fontweight="bold", color="white", family="sans-serif")

def _lines(x, y, lines, mono=False, fs=None):
    """Draw a list of (text, {props}) tuples. Returns y after last line."""
    fs = fs or FS
    fam = "monospace" if mono else "sans-serif"
    for txt, p in lines:
        ax.text(x, y, txt, fontsize=fs, family=p.get("fam", fam),
                color=p.get("c", "black"), fontstyle=p.get("s", "normal"),
                fontweight=p.get("w", "normal"), ha="left", va="top")
        y -= LINE
    return y

def _arrow(y1, y2, x=None):
    x = x or CX
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["ARROW"], lw=1.3))

P = {}   # plain props
B = {"w": "bold"}
I = {"s": "italic"}
M = {"fam": "monospace"}
MR = {"fam": "monospace", "c": C["RED"]}
MG = {"fam": "monospace", "c": C["GREEN"]}
RED = {"c": C["RED"]}
GRN = {"c": C["GREEN"]}

# ════════════════════════════════════════════════════════════════════════════
# 0. TASK INPUT
# ════════════════════════════════════════════════════════════════════════════
y = 27.5
h = 0.7
_box(y, h, C["INPUT"])
ax.text(CX, y - 0.18, "User Task (NLP Prompt)", ha="center", va="center",
        fontsize=FST, fontweight="bold", family="sans-serif", color=C["TITLE"])
ax.text(CX, y - 0.50, '\u201cFind which net is connected to pin A of instance inv_1\u201d',
        ha="center", va="center", fontsize=8.5, family="sans-serif", style="italic")
y -= h
_arrow(y - 0.02, y - GAP + 0.02)
y -= GAP

# ════════════════════════════════════════════════════════════════════════════
# 1. CHAIN EXTRACTION
# ════════════════════════════════════════════════════════════════════════════
h1 = 3.2
bootstrap_top = y
_box(y, h1, C["LLM"])
_title(y, "Stage 1: Causal Chain Extraction (LLM)")

ly = y - 0.6
ax.text(LX, ly, "Prompt", fontsize=FS+0.5, fontweight="bold", family="sans-serif"); ly -= LINE
_lines(LX, ly, [
    ("System: You are an OpenROAD Python API expert.", B),
    ("Identify the mandatory sequence of object types", I),
    ("that must be acquired step-by-step.", I),
    ("", P),
    ("Rules:", B),
    ("  \u2022 Entry point = openroad.Design", P),
    ("  \u2022 No dbChip (C++ internal, not in Python API)", P),
    ("  \u2022 Use exact Python type names", P),
    ("  \u2022 Output ONLY a JSON array", P),
    ("", P),
    ("User:", B),
    ("  Task: Find which net is connected to pin A", P),
    ("        of instance inv_1", P),
])

ry = y - 0.6
ax.text(RX, ry, "Output", fontsize=FS+0.5, fontweight="bold", family="sans-serif"); ry -= LINE
_lines(RX, ry, [
    ("LLM Output:", B),
    ("", P),
    ('["openroad.Design",', M),
    (' "odb.dbBlock",', M),
    (' "odb.dbInst",', M),
    (' "odb.dbITerm",', M),
    (' "odb.dbNet"]', M),
])

y -= h1
_arrow(y - 0.02, y - GAP + 0.02)
y -= GAP

# ════════════════════════════════════════════════════════════════════════════
# 2. RAG RETRIEVAL
# ════════════════════════════════════════════════════════════════════════════
h2 = 2.4
_box(y, h2, C["RAG"])
_title(y, "Stage 2: Graph-Guided RAG Retrieval (per-edge, no LLM)")

ax.text(LX, y - 0.6,
    "For each edge in the chain, query the structured API database (RAGAPIs.csv):",
    fontsize=FS, family="sans-serif")

rows = [
    ("Source", "Target", "Retrieved API Method", "Score"),
    ("Design", "dbBlock", "design.getBlock() \u2192 dbBlock", "1.00 (strict)"),
    ("dbBlock", "dbInst", "block.findInst(inst_name) \u2192 dbInst", "1.00 (strict)"),
    ("dbInst", "dbITerm", "inst.findITerm(mterm_name) \u2192 dbITerm", "0.92 (loose)"),
    ("dbITerm", "dbNet", "iterm.getNet() \u2192 dbNet", "1.00 (strict)"),
]
ty = y - 0.95
cols = [LX + 0.2, LX + 2.2, LX + 4.5, LX + 11.5]
for i, (a, b, c, d) in enumerate(rows):
    fw = "bold" if i == 0 else "normal"
    fam = "sans-serif" if i == 0 else "monospace"
    ax.text(cols[0], ty, a, fontsize=FS, fontweight=fw, family=fam)
    ax.text(cols[0] + 1.5, ty, "\u2192", fontsize=FS, family="sans-serif")
    ax.text(cols[1], ty, b, fontsize=FS, fontweight=fw, family=fam)
    ax.text(cols[2], ty, c, fontsize=FS, fontweight=fw, family=fam if i == 0 else "monospace")
    ax.text(cols[3], ty, d, fontsize=FS, fontweight=fw, family="sans-serif")
    ty -= LINE

y -= h2
_arrow(y - 0.02, y - GAP + 0.02)
y -= GAP

# ════════════════════════════════════════════════════════════════════════════
# 3. CODE GENERATION (Attempt 1)
# ════════════════════════════════════════════════════════════════════════════
h3 = 3.8
bootstrap_bot = y - h3
_box(y, h3, C["LLM"])
_title(y, "Stage 3: Constraint-Injected Code Generation (LLM \u2014 Attempt 1)")

ly = y - 0.6
ax.text(LX, ly, "Prompt", fontsize=FS+0.5, fontweight="bold", family="sans-serif"); ly -= LINE
_lines(LX, ly, [
    ("System: You are an expert OpenROAD Python API", B),
    ("programmer. Generate a script for the shell.", I),
    ("", P),
    ("Rules:", B),
    ("  \u2022 design / tech pre-available", P),
    ("  \u2022 Acquire objects in chain order", P),
    ("  \u2022 ALL methods are instance methods", P),
    ("  \u2022 Flat procedural code (no classes/functions)", P),
    ("  \u2022 Null-check all find*() calls", P),
    ("  \u2022 Provide [Diagnosis] if deviating from RAG", P),
    ("", P),
    ("User:", B),
    ('  Task: Find which net is connected to', P),
    ('        pin A of instance inv_1', P),
    ("", P),
    ("  Chain: Design \u2192 dbBlock \u2192 dbInst \u2192 dbITerm \u2192 dbNet", P),
    ("", P),
    ("  Edge APIs (from RAG):", B),
    ('    design.getBlock() \u2192 dbBlock', P),
    ('    block.findInst("inv_1") \u2192 dbInst', P),
    ('    inst.findITerm("A") \u2192 dbITerm', P),
    ('    iterm.getNet() \u2192 dbNet', P),
])

ry = y - 0.6
ax.text(RX, ry, "Output", fontsize=FS+0.5, fontweight="bold", family="sans-serif"); ry -= LINE
_lines(RX, ry, [
    ("LLM Output:", B),
    ("", P),
    ("[Diagnosis]: None", M),
    ("[Code]:", M),
    ("block = design.getBlock()", M),
    ('inst = block.findInst("inv_1")', M),
    ('pin = inst.getITerm("A")', MR),
    ("net = pin.getNet()", M),
    ("print(net.getName())", M),
    ("", P),
    ("", P),
    ("\u2191 Bug: getITerm() does not exist", RED),
    ("  on odb.dbInst. Should be findITerm().", RED),
    ("  Also: no None-check on findInst().", RED),
])

y -= h3
_arrow(y - 0.02, y - GAP + 0.02)
y -= GAP

# ════════════════════════════════════════════════════════════════════════════
# 4. VERIFICATION (FAIL)
# ════════════════════════════════════════════════════════════════════════════
h4 = 2.6
loop_top = y
_box(y, h4, C["VERIFY"])
_title(y, "Stage 4: Multi-Layer Causal Verification \u2014 FAIL")

checks = [
    ("L1 \u2014 Syntax", "ast.parse() succeeds", C["GREEN"], "\u2713 PASS"),
    ("L2 \u2014 Causal Flow", "Chain continuity OK; methods on correct receivers", C["GREEN"], "\u2713 PASS"),
    ("L2c \u2014 Null Safety", "findInst() result used without None-check", C["RED"], "\u2717 FAIL"),
    ("L3 \u2014 API Diff", 'getITerm("A") not a method of odb.dbInst; RAG says findITerm()', C["RED"], "\u2717 FAIL"),
]
ty = y - 0.65
for layer, desc, col, result in checks:
    ax.text(LX + 0.1, ty, layer, fontsize=FS, fontweight="bold", family="sans-serif")
    ax.text(LX + 3.5, ty, desc, fontsize=FS - 0.3, family="sans-serif")
    ax.text(X0 + BW - 1.0, ty, result, fontsize=FS, fontweight="bold", color=col, family="sans-serif")
    ty -= LINE * 1.15

ty -= 0.15
ax.text(LX, ty, "Structured Feedback \u2192 Controller:", fontsize=FS, fontweight="bold", family="sans-serif")
ty -= LINE * 1.1
ax.text(LX, ty,
    '"FAIL(L3): hallucinated method getITerm() on odb.dbInst. RAG suggests inst.findITerm(mterm_name).',
    fontsize=FS - 0.5, family="sans-serif", style="italic", color="#555555")
ty -= LINE
ax.text(LX, ty,
    'Also: findInst() result must be None-checked before dereferencing."',
    fontsize=FS - 0.5, family="sans-serif", style="italic", color="#555555")

y -= h4
_arrow(y - 0.02, y - GAP + 0.02)
y -= GAP

# ════════════════════════════════════════════════════════════════════════════
# 5. CONTROLLER DECISION
# ════════════════════════════════════════════════════════════════════════════
h5 = 3.3
_box(y, h5, C["CTRL"])
_title(y, "Controller Decision (LLM Arbiter)")

ly = y - 0.6
ax.text(LX, ly, "Prompt", fontsize=FS+0.5, fontweight="bold", family="sans-serif"); ly -= LINE
_lines(LX, ly, [
    ("System: You are the Causal Arbiter.", B),
    ("Resolve conflicts between the Verifier", I),
    ("and the Generator, and decide the single", I),
    ("best next action.", I),
    ("", P),
    ("Available actions:", B),
    ("  re_generate, re_retrieve_edge,", P),
    ("  re_extract_chain, re_generate_tcl,", P),
    ("  commit_best, stop_fail", P),
    ("", P),
    ("Decision rule: L3 hard hallucination +", P),
    ("RAG already has the correct method", P),
    ("\u21d2 re_generate with repair hint.", P),
    ("", P),
    ("User: Full agent state (chain, APIs,", B),
    ("  generated code, FAIL(L3), attempt 1/6)", P),
])

ry = y - 0.6
ax.text(RX, ry, "Output", fontsize=FS+0.5, fontweight="bold", family="sans-serif"); ry -= LINE
_lines(RX, ry, [
    ("Controller Output:", B),
    ("", P),
    ('{', M),
    ('  "diagnosis": "getITerm()', M),
    ('     hallucinated; correct method', M),
    ('     is findITerm()",', M),
    ('', P),
    ('  "next_action": "re_generate",', M),
    ('', P),
    ('  "repair_hint": "Use', M),
    ("     inst.findITerm('A'), not", M),
    ("     inst.getITerm('A'). Add", M),
    ('     None-check after findInst()."', M),
    ('', P),
    ('  "updated_lesson": "dbInst has', M),
    ('     findITerm, not getITerm"', M),
    ('}', M),
])

y -= h5
_arrow(y - 0.02, y - GAP + 0.02)
y -= GAP

# ════════════════════════════════════════════════════════════════════════════
# 6. RE-GENERATION (Attempt 2)
# ════════════════════════════════════════════════════════════════════════════
h6 = 3.2
_box(y, h6, C["LLM"])
_title(y, "Re-Generation (LLM \u2014 Attempt 2, with repair hint injected)")

ly = y - 0.6
ax.text(LX, ly, "Prompt (appended to conversation)", fontsize=FS+0.5, fontweight="bold", family="sans-serif"); ly -= LINE
_lines(LX, ly, [
    ("", P),
    ("CRITICAL PROHIBITIONS:", B),
    ("  \u2022 dbInst has findITerm, NOT getITerm", P),
    ("", P),
    ("REPAIR HINT:", B),
    ('  Use inst.findITerm("A") instead of', P),
    ('  inst.getITerm("A").', P),
    ("  Add None-check after block.findInst().", P),
])

ry = y - 0.6
ax.text(RX, ry, "Output", fontsize=FS+0.5, fontweight="bold", family="sans-serif"); ry -= LINE
_lines(RX, ry, [
    ("LLM Output (corrected):", B),
    ("", P),
    ("[Diagnosis]: None", M),
    ("[Code]:", M),
    ("block = design.getBlock()", M),
    ('inst = block.findInst("inv_1")', M),
    ("if inst is None:", MG),
    ('    print("inv_1 not found")', MG),
    ("else:", MG),
    ('    pin = inst.findITerm("A")', MG),
    ("    if pin is not None:", MG),
    ("        net = pin.getNet()", MG),
    ("        if net is not None:", MG),
    ("            print(net.getName())", MG),
])

y -= h6
_arrow(y - 0.02, y - GAP + 0.02)
y -= GAP

# ════════════════════════════════════════════════════════════════════════════
# 7. RE-VERIFICATION (PASS)
# ════════════════════════════════════════════════════════════════════════════
h7 = 2.2
_box(y, h7, C["PASS"])
_title(y, "Re-Verification \u2014 PASS \u2713", bg=C["GREEN"])
loop_bot = y - h7

pass_checks = [
    ("L1 \u2014 Syntax", "ast.parse() OK"),
    ("L2 \u2014 Causal Flow", "All 4 edges acquired in correct order"),
    ("L2c \u2014 Null Safety", "findInst, findITerm both None-checked"),
    ("L3 \u2014 API Diff", "findITerm matches RAG; getNet matches RAG"),
    ("L5 \u2014 Semantic (LLM)", "A\u2713 B\u2713 C\u2713 D\u2713 E\u2713  (5/5, confidence = 1.0)"),
]
ty = y - 0.65
for layer, desc in pass_checks:
    ax.text(LX + 0.1, ty, layer, fontsize=FS, fontweight="bold", family="sans-serif")
    ax.text(LX + 3.8, ty, desc, fontsize=FS, family="sans-serif")
    ax.text(X0 + BW - 0.7, ty, "\u2713", fontsize=FS + 1, fontweight="bold",
            color=C["GREEN"], family="sans-serif")
    ty -= LINE * 1.1

ty -= 0.2
ax.text(LX, ty,
    "Controller:  commit_best  \u2192  Final code committed.   Budget used: 1 of 6.",
    fontsize=FS + 0.5, fontweight="bold", family="sans-serif", color=C["TITLE"])

y -= h7

# ════════════════════════════════════════════════════════════════════════════
# SIDE BRACKETS
# ════════════════════════════════════════════════════════════════════════════
bx = X0 + BW + 0.35

# Bootstrap
ax.plot([bx, bx + 0.15], [bootstrap_top, bootstrap_top], color=C["ARROW"], lw=1.5)
ax.plot([bx, bx + 0.15], [bootstrap_bot, bootstrap_bot], color=C["ARROW"], lw=1.5)
ax.plot([bx + 0.15, bx + 0.15], [bootstrap_top, bootstrap_bot], color=C["ARROW"], lw=1.5)
mid = (bootstrap_top + bootstrap_bot) / 2
ax.text(bx + 0.35, mid + 0.15, "Bootstrap", fontsize=8, fontweight="bold",
        color=C["ARROW"], family="sans-serif", rotation=90, va="center")
ax.text(bx + 0.35, mid - 0.35, "(free)", fontsize=7,
        color=C["ARROW"], family="sans-serif", rotation=90, va="center")

# Agentic loop
ax.plot([bx, bx + 0.15], [loop_top, loop_top], color="#8E44AD", lw=1.5)
ax.plot([bx, bx + 0.15], [loop_bot, loop_bot], color="#8E44AD", lw=1.5)
ax.plot([bx + 0.15, bx + 0.15], [loop_top, loop_bot], color="#8E44AD", lw=1.5)
mid = (loop_top + loop_bot) / 2
ax.text(bx + 0.35, mid + 0.35, "Agentic", fontsize=8, fontweight="bold",
        color="#8E44AD", family="sans-serif", rotation=90, va="center")
ax.text(bx + 0.35, mid - 0.2, "Loop", fontsize=8, fontweight="bold",
        color="#8E44AD", family="sans-serif", rotation=90, va="center")
ax.text(bx + 0.35, mid - 0.75, "(max B", fontsize=7,
        color="#8E44AD", family="sans-serif", rotation=90, va="center")
ax.text(bx + 0.35, mid - 1.15, "iters)", fontsize=7,
        color="#8E44AD", family="sans-serif", rotation=90, va="center")

# ════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════
ly = y - 0.55
items = [
    (C["INPUT"], "User Input"), (C["LLM"], "LLM Call"), (C["RAG"], "RAG Retrieval"),
    (C["VERIFY"], "Verification (Fail)"), (C["CTRL"], "Controller"), (C["PASS"], "Verification (Pass)"),
]
lx = LX
for color, label in items:
    ax.add_patch(FancyBboxPatch(
        (lx, ly - 0.08), 0.3, 0.2, boxstyle="round,pad=0.02",
        lw=0.5, edgecolor=C["BORDER"], facecolor=color))
    ax.text(lx + 0.4, ly + 0.02, label, fontsize=6.5, family="sans-serif",
            va="center", color=C["ARROW"])
    lx += 2.5

# ── Save ────────────────────────────────────────────────────────────────────
plt.tight_layout()
out_dir = "causal_verifier_4_2"
plt.savefig(f"{out_dir}/fig_demo_pipeline.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{out_dir}/fig_demo_pipeline.png", bbox_inches="tight", dpi=300)
print(f"Saved: {out_dir}/fig_demo_pipeline.pdf")
print(f"Saved: {out_dir}/fig_demo_pipeline.png")
