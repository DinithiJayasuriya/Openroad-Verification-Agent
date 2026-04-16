#!/usr/bin/env python3
"""fig_demo_e2e.py — End-to-end demo: one task through the full pipeline.

Shows: Chain extraction (with hallucination catch) → RAG → Code gen (with L3 catch)
       → Controller decision → Re-gen → PASS.

Output: fig_demo_e2e.pdf / .png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Colours ─────────────────────────────────────────────────────────────────
BG_TASK  = "#EBF5FB"
BG_LLM   = "#FEF9E7"
BG_FAIL  = "#FDEDEC"
BG_FIX   = "#F4ECF7"
BG_RAG   = "#E8F8F5"
BG_PASS  = "#D5F5E3"
BG_CODE  = "#F8F9F9"
C_T      = "#2C3E50"
C_B      = "#AEB6BF"
C_R      = "#C0392B"
C_G      = "#27AE60"
C_A      = "#5D6D7E"
C_P      = "#8E44AD"

fig, ax = plt.subplots(figsize=(13.5, 19))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 19)
ax.axis("off")

# ── Helpers ─────────────────────────────────────────────────────────────────
def box(x, y, w, h, bg, ec=C_B, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                 lw=lw, ec=ec, fc=bg))

def tbar(x, y, w, text, bg=C_T, fs=8.5):
    box(x, y, w, 0.32, bg, ec="none", lw=0)
    ax.text(x + w/2, y + 0.16, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white", family="sans-serif")

def t(x, y, s, fs=7.2, **kw):
    d = dict(ha="left", va="top", family="sans-serif", fontsize=fs)
    d.update(kw)
    ax.text(x, y, s, **d)

def m(x, y, s, fs=7, **kw):
    t(x, y, s, fs=fs, family="monospace", **kw)

def arr_d(x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_A, lw=1.2))

def arr_r(x1, y, x2, c=C_A):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))

L = 0.24  # line height

# ════════════════════════════════════════════════════════════════════════════
# 1  TASK
# ════════════════════════════════════════════════════════════════════════════
Y = 18.6
box(0.3, Y, 12.9, 0.55, BG_TASK)
ax.text(6.75, Y + 0.28, 'Task:  "Find which net is connected to pin A of instance inv_1"',
        ha="center", va="center", fontsize=9, style="italic", family="sans-serif")
arr_d(6.75, Y - 0.03, Y - 0.22)

# ════════════════════════════════════════════════════════════════════════════
# 2  CHAIN EXTRACTION  →  HALLUCINATION  →  CORRECTION
# ════════════════════════════════════════════════════════════════════════════
Y = 15.85
H = 2.45
box(0.3, Y, 12.9, H, BG_LLM)
tbar(0.45, Y + H - 0.38, 12.6, "\u2460  Chain Extraction  \u2192  Validation Gate  \u2192  Corrected Chain")

# -- initial chain (left)
cx = 0.6; cy = Y + H - 0.6
t(cx, cy, "LLM extracts chain:", fontweight="bold"); cy -= L * 1.1
m(cx + 0.2, cy, '["openroad.Design",'); cy -= L * 0.85
m(cx + 0.2, cy, ' "odb.dbBlock",'); cy -= L * 0.85
m(cx + 0.2, cy, ' "odb.dbInst",'); cy -= L * 0.85
m(cx + 0.2, cy, ' "odb.dbITerm",'); cy -= L * 0.85
m(cx + 0.2, cy, ' "odb.dbNet"]')

# -- validation (middle)
vx = 4.5; vy = Y + H - 0.6
t(vx, vy, "Validation gate checks each type:", fontweight="bold", fs=7); vy -= L * 1.1
checks = [
    ("Design",  "\u2713 VALID", C_G),
    ("dbBlock",  "\u2713 VALID", C_G),
    ("dbInst",   "\u2713 VALID", C_G),
    ("dbITerm",  "\u2713 VALID", C_G),
    ("dbNet",    "\u2713 VALID", C_G),
]
for typ, stat, c in checks:
    m(vx + 0.1, vy, typ, fs=6.8)
    t(vx + 2.0, vy, stat, fs=6.8, color=c, fontweight="bold")
    vy -= L * 0.8

t(vx, vy - 0.05, "All types exist in API database \u2713", fs=7, color=C_G, fontweight="bold")

# -- what-if callout (right side)
rx = 8.8; ry = Y + H - 0.6
t(rx, ry, "What if LLM had hallucinated?", fontweight="bold", fs=7, color="#888"); ry -= L * 1.1
box(rx - 0.1, ry - 1.6, 4.3, 1.8, "#FFF5F5", ec=C_R, lw=0.8)
t(rx + 0.1, ry - 0.05, "If chain included", fs=6.8, color=C_R)
m(rx + 0.1, ry - 0.28, "odb.dbChip", fs=7, color=C_R, fontweight="bold")
t(rx + 2.2, ry - 0.28, "(C++ only):", fs=6.8, color=C_R)
ry -= 0.55
t(rx + 0.1, ry, "\u2192 Gate flags HALLUCINATION", fs=6.5, color=C_R)
ry -= L * 0.85
t(rx + 0.1, ry, "\u2192 Edit-distance suggests", fs=6.5, color=C_R)
m(rx + 0.1, ry - L * 0.85, "   odb.dbBlock", fs=6.8, color=C_G, fontweight="bold")
t(rx + 2.8, ry - L * 0.85, "(73%)", fs=6, color="#999")
ry -= L * 1.7
t(rx + 0.1, ry, "\u2192 LLM re-extracts with fix", fs=6.5, color=C_R)

arr_d(6.75, Y - 0.03, Y - 0.22)

# ════════════════════════════════════════════════════════════════════════════
# 3  RAG RETRIEVAL
# ════════════════════════════════════════════════════════════════════════════
Y = 14.35
H = 1.3
box(0.3, Y, 12.9, H, BG_RAG)
tbar(0.45, Y + H - 0.38, 12.6, "\u2461  Per-Edge RAG Retrieval")

ey = Y + H - 0.55
edges = [
    ("Design \u2192 dbBlock",  "design.getBlock()",          "1.00"),
    ("dbBlock \u2192 dbInst",  'block.findInst(name)',       "1.00"),
    ("dbInst \u2192 dbITerm",  'inst.findITerm(mterm_name)', "0.92"),
    ("dbITerm \u2192 dbNet",   'iterm.getNet()',             "1.00"),
]
xs = [0.8, 3.5, 8.5]
t(xs[0], ey, "Edge", fontweight="bold", fs=6.5)
t(xs[1], ey, "Retrieved Method", fontweight="bold", fs=6.5)
t(xs[2], ey, "Score", fontweight="bold", fs=6.5)
ey -= L * 0.85
for edge, method, score in edges:
    t(xs[0], ey, edge, fs=6.8)
    m(xs[1], ey, method, fs=6.8)
    t(xs[2], ey, score, fs=6.8, color="#888")
    ey -= L * 0.75

arr_d(6.75, Y - 0.03, Y - 0.22)

# ════════════════════════════════════════════════════════════════════════════
# 4  CODE GENERATION — ATTEMPT 1  (with bug)
# ════════════════════════════════════════════════════════════════════════════
Y = 11.9
H = 2.05
box(0.3, Y, 6.15, H, BG_LLM)
tbar(0.45, Y + H - 0.38, 5.85, "\u2462  Code Generation (Attempt 1)")

cy = Y + H - 0.58
m(0.6, cy, "block = design.getBlock()"); cy -= L * 0.9
m(0.6, cy, 'inst = block.findInst("inv_1")'); cy -= L * 0.9
m(0.6, cy, 'pin = inst.getITerm("A")', color=C_R, fontweight="bold"); cy -= L * 0.9
m(0.6, cy, "net = pin.getNet()"); cy -= L * 0.9
m(0.6, cy, "print(net.getName())"); cy -= L * 1.2
t(0.6, cy, "\u26a0 getITerm() does not exist on dbInst", fs=6.8, color=C_R, fontweight="bold")
t(0.6, cy - L * 0.85, "  Also: no None-check after findInst()", fs=6.5, color=C_R)

# ── VERIFIER (right of attempt 1)
box(7.05, Y, 6.15, H, BG_FAIL)
tbar(7.2, Y + H - 0.38, 5.85, "\u2463  Verifier Catches Error", bg=C_R)

vy = Y + H - 0.6
layers = [
    ("L1 Syntax",     "\u2713 PASS", C_G),
    ("L2 Causal Flow", "\u2713 PASS", C_G),
    ("L2c Null Safety", "\u2717 FAIL \u2014 findInst() unchecked", C_R),
    ("L3 API Diff",    '\u2717 FAIL \u2014 getITerm() hallucinated', C_R),
]
for layer, result, c in layers:
    t(7.3, vy, layer, fs=6.8, fontweight="bold")
    t(9.5, vy, result, fs=6.5, color=c, fontweight="bold")
    vy -= L * 0.95

vy -= L * 0.5
t(7.3, vy, "L3 feedback:", fontweight="bold", fs=7)
vy -= L * 0.95
t(7.3, vy, '"getITerm() not a method of dbInst.', fs=6.3, color="#555", style="italic")
vy -= L * 0.8
t(7.3, vy, ' RAG suggests: findITerm(mterm_name)"', fs=6.3, color="#555", style="italic")

arr_d(6.75, Y - 0.03, Y - 0.22)

# ════════════════════════════════════════════════════════════════════════════
# 5  CONTROLLER DECISION
# ════════════════════════════════════════════════════════════════════════════
Y = 10.0
H = 1.6
box(0.3, Y, 12.9, H, BG_FIX)
tbar(0.45, Y + H - 0.38, 12.6, "\u2464  Controller Decision (LLM Arbiter)", bg=C_P)

cy = Y + H - 0.6
t(0.6, cy, "Input:", fontweight="bold", fs=7)
t(1.5, cy, "Verifier result FAIL(L3) + full agent state (chain, code, attempt 1 of 6)", fs=7)
cy -= L * 1.1
t(0.6, cy, "Rule:", fontweight="bold", fs=7)
t(1.5, cy, "L3 hallucination + RAG already has correct method  \u21d2  re_generate with repair hint", fs=7)
cy -= L * 1.2

m(0.8, cy, '{ "next_action": "re_generate",', fs=7.2)
m(0.8, cy - L * 0.9,
  '  "repair_hint": "Use inst.findITerm(\'A\'), not getITerm(). Add None-check after findInst().",', fs=7)
m(0.8, cy - L * 1.8,
  '  "lesson": "dbInst has findITerm(), not getITerm()" }', fs=7.2)

arr_d(6.75, Y - 0.03, Y - 0.22)

# ════════════════════════════════════════════════════════════════════════════
# 6  RE-GENERATION — ATTEMPT 2  (corrected)
# ════════════════════════════════════════════════════════════════════════════
Y = 6.8
H = 2.85
box(0.3, Y, 6.15, H, BG_LLM)
tbar(0.45, Y + H - 0.38, 5.85, "\u2465  Re-Generation (Attempt 2)")

cy = Y + H - 0.58
t(0.6, cy, "Repair hint injected:", fontweight="bold", fs=7, color=C_P); cy -= L * 1.0
t(0.6, cy, '\u2022 Use findITerm(), not getITerm()', fs=6.8, color=C_P); cy -= L * 0.85
t(0.6, cy, '\u2022 Add None-check after findInst()', fs=6.8, color=C_P); cy -= L * 1.3

t(0.6, cy, "Corrected code:", fontweight="bold", color=C_G); cy -= L * 1.0
m(0.6, cy, "block = design.getBlock()"); cy -= L * 0.85
m(0.6, cy, 'inst = block.findInst("inv_1")'); cy -= L * 0.85
m(0.6, cy, 'if inst is None:', color=C_G, fontweight="bold"); cy -= L * 0.85
m(0.6, cy, '    print("Not found")', color=C_G); cy -= L * 0.85
m(0.6, cy, 'else:', color=C_G, fontweight="bold"); cy -= L * 0.85
m(0.6, cy, '    pin = inst.findITerm("A")', color=C_G, fontweight="bold"); cy -= L * 0.85
m(0.6, cy, '    if pin is not None:', color=C_G); cy -= L * 0.85
m(0.6, cy, '        net = pin.getNet()', color=C_G); cy -= L * 0.85
m(0.6, cy, '        print(net.getName())', color=C_G)

# ── RE-VERIFICATION (right of attempt 2)
box(7.05, Y, 6.15, H, BG_PASS, ec=C_G, lw=1.3)
tbar(7.2, Y + H - 0.38, 5.85, "\u2466  Re-Verification \u2014 PASS \u2713", bg=C_G)

vy = Y + H - 0.65
pass_layers = [
    "L1 Syntax           \u2713",
    "L2 Causal Flow      \u2713  all edges OK",
    "L2c Null Safety     \u2713  findInst, findITerm checked",
    "L3 API Diff          \u2713  findITerm matches RAG",
    "L5 Semantic (LLM)  \u2713  5/5 angles pass",
]
for pl in pass_layers:
    t(7.4, vy, pl, fs=7, fontweight="bold", color=C_G)
    vy -= L * 1.05

vy -= L * 0.6
t(7.4, vy, "Controller:", fontweight="bold", fs=7.5)
m(9.0, vy, "commit_best", fs=7.5, color=C_T, fontweight="bold")
vy -= L * 1.2
t(7.4, vy, "Final code committed.", fs=8, fontweight="bold", color=C_T)
t(7.4, vy - L * 1.0, "Budget used: 1 of 6.", fs=7.5, color="#888")

# ════════════════════════════════════════════════════════════════════════════
# FLOW SUMMARY BAR at bottom
# ════════════════════════════════════════════════════════════════════════════
Y = 6.1
box(0.3, Y, 12.9, 0.45, "#F8F9F9", ec="#DDD", lw=0.7)
steps = [
    ("\u2460 Extract", BG_LLM),
    ("\u2192", None),
    ("Validate", BG_FAIL),
    ("\u2192", None),
    ("\u2461 RAG", BG_RAG),
    ("\u2192", None),
    ("\u2462 Generate", BG_LLM),
    ("\u2192", None),
    ("\u2463 Verify", BG_FAIL),
    ("\u2192", None),
    ("\u2464 Controller", BG_FIX),
    ("\u2192", None),
    ("\u2465 Re-gen", BG_LLM),
    ("\u2192", None),
    ("\u2466 PASS", BG_PASS),
]
sx = 0.6
for label, bg in steps:
    if bg:
        box(sx - 0.05, Y + 0.06, len(label) * 0.16 + 0.25, 0.28, bg, ec=C_B, lw=0.4)
    t(sx, Y + 0.25, label, fs=6, fontweight="bold" if bg else "normal",
      color=C_T if bg else "#AAA", va="center")
    sx += len(label) * 0.16 + 0.4

# ── Legend ───────────────────────────────────────────────────────────────────
items = [(BG_LLM, "LLM"), (BG_RAG, "RAG"), (BG_FAIL, "Verify/Fail"),
         (BG_FIX, "Controller"), (BG_PASS, "Pass")]
lx = 1.5
for c, lab in items:
    box(lx, 5.55, 0.25, 0.2, c, ec=C_B, lw=0.4)
    t(lx + 0.32, 5.68, lab, fs=6, color=C_A, va="center")
    lx += 2.4

# ── Save ────────────────────────────────────────────────────────────────────
plt.tight_layout()
d = "causal_verifier_4_2"
for ext in ("pdf", "png"):
    plt.savefig(f"{d}/fig_demo_e2e.{ext}", bbox_inches="tight", dpi=300)
print(f"Saved: {d}/fig_demo_e2e.pdf  &  .png")
