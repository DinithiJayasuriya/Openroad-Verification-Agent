#!/usr/bin/env python3
"""fig_demo_compact.py — Compact demo: Chain Extraction → Hallucination Catch → Correction → Generation.

Output: fig_demo_compact.pdf / .png
Usage:  python3 causal_verifier_4_2/fig_demo_compact.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Colours ─────────────────────────────────────────────────────────────────
BG_TASK   = "#EBF5FB"
BG_LLM    = "#FEF9E7"
BG_GATE   = "#FDEDEC"
BG_FIX    = "#F4ECF7"
BG_RAG    = "#E8F8F5"
BG_GEN    = "#FEF9E7"
BG_CODE   = "#F8F9F9"
BG_PASS   = "#D5F5E3"
C_TITLE   = "#2C3E50"
C_BORDER  = "#AEB6BF"
C_RED     = "#C0392B"
C_GREEN   = "#27AE60"
C_ARROW   = "#5D6D7E"
C_PURPLE  = "#8E44AD"

# ── Canvas ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 16.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 16.5)
ax.axis("off")

# ── Helpers ─────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, bg, ec=C_BORDER, lw=1.0, zorder=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                        lw=lw, ec=ec, fc=bg, zorder=zorder)
    ax.add_patch(p)
    return p

def title_bar(x, y, w, text, bg=C_TITLE):
    rbox(x, y, w, 0.36, bg, ec="none", lw=0)
    ax.text(x + w/2, y + 0.18, text, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", family="sans-serif")

def arrow_down(x, y1, y2, color=C_ARROW, lw=1.3):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

def arrow_right(x1, y, x2, color=C_ARROW, lw=1.3):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

def txt(x, y, s, fs=7.5, **kw):
    defaults = dict(ha="left", va="top", family="sans-serif", fontsize=fs)
    defaults.update(kw)
    ax.text(x, y, s, **defaults)

def mono(x, y, s, fs=7.3, **kw):
    txt(x, y, s, fs=fs, family="monospace", **kw)

LN = 0.26   # line height

# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — TASK INPUT  (top)
# ════════════════════════════════════════════════════════════════════════════
rbox(0.3, 15.55, 13.4, 0.75, BG_TASK)
title_bar(0.45, 15.95, 13.1, "NLP Task Prompt")
txt(7.0, 15.82, '"Find all instances in the chip of this design"',
    fs=9, ha="center", va="center", style="italic")

arrow_down(7.0, 15.52, 15.15)

# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — CHAIN EXTRACTION (left) + OUTPUT (right)
# ════════════════════════════════════════════════════════════════════════════
# Left: prompt
rbox(0.3, 12.15, 6.3, 2.9, BG_LLM)
title_bar(0.45, 14.7, 6.0, "Stage 1: Causal Chain Extraction (LLM)")

y = 14.52
txt(0.6, y, "System:", fontweight="bold"); y -= LN
txt(0.6, y, "You are an OpenROAD Python API expert.", style="italic", fs=7); y -= LN
txt(0.6, y, "Identify the mandatory sequence of object", style="italic", fs=7); y -= LN
txt(0.6, y, "types to complete the task.", style="italic", fs=7); y -= LN * 0.6
txt(0.6, y, "Rules:", fontweight="bold", fs=7); y -= LN * 0.85
txt(0.6, y, "  \u2022 Entry point = openroad.Design", fs=7); y -= LN * 0.85
txt(0.6, y, "  \u2022 No dbChip (C++ only, not in Python API)", fs=7); y -= LN * 0.85
txt(0.6, y, "  \u2022 Output ONLY a JSON array", fs=7); y -= LN * 1.2
txt(0.6, y, "User:", fontweight="bold"); y -= LN
txt(0.6, y, "Task: Find all instances in the chip of", fs=7); y -= LN * 0.85
txt(0.6, y, "         this design", fs=7)

# Right: LLM output with hallucination
rbox(7.0, 12.15, 6.7, 2.9, BG_LLM)
title_bar(7.15, 14.7, 6.4, "LLM Output (Initial Chain)")

y = 14.48
txt(7.3, y, "Extracted chain:", fontweight="bold"); y -= LN * 1.3
mono(7.6, y, '["openroad.Design",', fs=7.5); y -= LN
mono(7.6, y, ' "odb.dbDatabase",', fs=7.5); y -= LN
mono(7.6, y, ' "odb.dbChip",', fs=7.5, color=C_RED, fontweight="bold"); y -= LN
mono(7.6, y, ' "odb.dbBlock",', fs=7.5); y -= LN
mono(7.6, y, ' "odb.dbInst"]', fs=7.5); y -= LN * 1.4

# Hallucination callout
rbox(7.4, y - 0.45, 5.9, 0.55, "#FADBD8", ec=C_RED, lw=1.2)
txt(7.6, y - 0.05, "\u26a0  odb.dbChip is a C++ internal \u2014 not exposed",
    fs=7, color=C_RED, fontweight="bold")
txt(7.6, y - 0.30, "   in the Python API. This would crash at runtime.",
    fs=7, color=C_RED)

arrow_down(7.0, 12.12, 11.78)

# ════════════════════════════════════════════════════════════════════════════
# ROW 3 — VALIDATION GATE (left) + CORRECTION (right)
# ════════════════════════════════════════════════════════════════════════════
rbox(0.3, 8.55, 6.3, 3.1, BG_GATE)
title_bar(0.45, 11.3, 6.0, "Structured Validation Gate")

y = 11.1
txt(0.6, y, "Check each type against API database:", fontweight="bold", fs=7); y -= LN * 1.1

checks = [
    ("openroad.Design", "VALID",        C_GREEN, "\u2713"),
    ("odb.dbDatabase",  "VALID",        C_GREEN, "\u2713"),
    ("odb.dbChip",      "HALLUCINATION", C_RED,  "\u2717"),
    ("odb.dbBlock",     "VALID",        C_GREEN, "\u2713"),
    ("odb.dbInst",      "VALID",        C_GREEN, "\u2713"),
]
for typ, status, col, sym in checks:
    fw = "bold" if "HALL" in status else "normal"
    mono(0.8, y, typ, fs=7, color=col if "HALL" in status else "black", fontweight=fw)
    txt(3.8, y, f"{sym} {status}", fs=7, color=col, fontweight="bold")
    y -= LN * 0.95

y -= LN * 0.4
txt(0.6, y, "Correction (edit distance):", fontweight="bold", fs=7.5); y -= LN * 1.1
mono(0.8, y, "dbChip", fs=7, color=C_RED)
txt(2.1, y, "vs candidates:", fs=7)
y -= LN * 0.9
mono(0.8, y, '  difflib.get_close_matches(', fs=6.5, color="#555")
y -= LN * 0.85
mono(0.8, y, '    "dbChip", known_types, cutoff=0.55)', fs=6.5, color="#555")
y -= LN * 0.9
txt(0.8, y, "\u2192 Best match:", fs=7)
mono(3.3, y, "odb.dbBlock", fs=7.5, color=C_GREEN, fontweight="bold")
txt(5.2, y, "(73%)", fs=6.5, color="#888")

# Right: feedback + re-extraction
rbox(7.0, 8.55, 6.7, 3.1, BG_FIX)
title_bar(7.15, 11.3, 6.4, "Corrective Re-Extraction (LLM)", bg=C_PURPLE)

y = 11.08
txt(7.3, y, "Feedback injected into prompt:", fontweight="bold", fs=7); y -= LN * 1.1
rbox(7.4, y - 1.0, 6.0, 1.15, "white", ec=C_BORDER, lw=0.7)
txt(7.6, y - 0.08, "CORRECTION REQUIRED:", fs=6.8, fontweight="bold", color=C_RED)
txt(7.6, y - 0.33, "'odb.dbChip' is INVALID. Closest real type:", fs=6.8, color="#444")
txt(7.6, y - 0.55, "'odb.dbBlock'. Rewrite the path to use", fs=6.8, color="#444")
txt(7.6, y - 0.77, "'odb.dbBlock' instead.", fs=6.8, color="#444")

y -= 1.35
txt(7.3, y, "Corrected chain:", fontweight="bold", color=C_GREEN); y -= LN * 1.2
mono(7.6, y, '["openroad.Design",', fs=7.5); y -= LN
mono(7.6, y, ' "odb.dbBlock",', fs=7.5, color=C_GREEN, fontweight="bold"); y -= LN
mono(7.6, y, ' "odb.dbInst"]', fs=7.5); y -= LN * 1.1
txt(7.6, y, "\u2713 All types validated against API database",
    fs=7, color=C_GREEN, fontweight="bold")

# Arrow connecting validation to correction
arrow_right(6.65, 10.1, 6.95, color=C_PURPLE)

arrow_down(7.0, 8.52, 8.15)

# ════════════════════════════════════════════════════════════════════════════
# ROW 4 — RAG RETRIEVAL + CODE GENERATION
# ════════════════════════════════════════════════════════════════════════════
rbox(0.3, 4.75, 6.3, 3.3, BG_RAG)
title_bar(0.45, 7.7, 6.0, "Stage 2: Per-Edge RAG Retrieval")

y = 7.48
txt(0.6, y, "Query API database for each edge:", fontweight="bold", fs=7); y -= LN * 1.3

edges = [
    ("Design \u2192 dbBlock", "design.getBlock()", "1.00"),
    ("dbBlock \u2192 dbInst", "block.getInsts()", "1.00"),
]
for edge, method, score in edges:
    txt(0.8, y, edge, fs=7, fontweight="bold"); y -= LN * 0.9
    mono(1.0, y, f"  \u2192 {method}", fs=7, color="#2C3E50"); y -= LN * 0.5
    txt(5.0, y + LN * 0.5, f"[{score}]", fs=6.5, color="#888"); y -= LN * 0.8

y -= LN * 0.3
txt(0.6, y, "Verified chain + APIs passed to generator:", fontweight="bold", fs=7, color=C_GREEN)
y -= LN * 1.2
mono(0.8, y, "Design \u2500\u2500getBlock()\u2500\u2500\u25b6 dbBlock", fs=7); y -= LN
mono(0.8, y, "dbBlock \u2500\u2500getInsts()\u2500\u2500\u25b6 [dbInst]", fs=7)

# Right: generated code
rbox(7.0, 4.75, 6.7, 3.3, BG_GEN)
title_bar(7.15, 7.7, 6.4, "Stage 3: Code Generation (LLM)")

y = 7.45
txt(7.3, y, "Prompt includes locked chain + RAG APIs.", fs=7); y -= LN * 0.5
txt(7.3, y, "LLM fills in task logic only.", fs=7, style="italic"); y -= LN * 1.3

txt(7.3, y, "Generated Code:", fontweight="bold", color=C_GREEN); y -= LN * 1.1
rbox(7.35, y - 2.25, 6.15, 2.35, BG_CODE, ec=C_BORDER, lw=0.6)
mono(7.55, y - 0.08, "# Acquisition (locked skeleton)", fs=6.8, color="#888")
mono(7.55, y - 0.36, "block = design.getBlock()", fs=7)
mono(7.55, y - 0.62, "insts = block.getInsts()", fs=7)
mono(7.55, y - 0.88, "", fs=7)
mono(7.55, y - 1.02, "# Task logic (LLM-generated)", fs=6.8, color="#888")
mono(7.55, y - 1.30, "print(f'Total instances: {len(insts)}')", fs=7)
mono(7.55, y - 1.56, "for inst in insts:", fs=7)
mono(7.55, y - 1.82, "    print(inst.getName())", fs=7)

arrow_right(6.65, 6.4, 6.95, color=C_GREEN)
arrow_down(7.0, 4.72, 4.35)

# ════════════════════════════════════════════════════════════════════════════
# ROW 5 — VERIFICATION PASS
# ════════════════════════════════════════════════════════════════════════════
rbox(0.3, 3.0, 13.4, 1.25, BG_PASS, ec=C_GREEN, lw=1.5)
title_bar(0.45, 3.9, 13.1, "Verification \u2014 All Layers PASS", bg=C_GREEN)

y = 3.72
layers = [
    "L1\u2013Syntax: \u2713",
    "L2\u2013Causal Flow: \u2713",
    "L2c\u2013Null Safety: \u2713",
    "L3\u2013API Diff: \u2713",
    "L5\u2013Semantic: \u2713 (5/5)",
]
x_pos = 0.8
for l in layers:
    txt(x_pos, y, l, fs=7.5, fontweight="bold", color=C_GREEN)
    x_pos += 2.6

txt(3.5, y - LN * 1.3, "Controller:  commit_best  \u2192  Final code committed.",
    fs=8, fontweight="bold", color=C_TITLE, ha="left")

# ════════════════════════════════════════════════════════════════════════════
# FLOW ANNOTATION — dashed path showing "what would have gone wrong"
# ════════════════════════════════════════════════════════════════════════════

# Counterfactual annotation
rbox(0.5, 1.55, 13.0, 1.2, "white", ec="#DDD", lw=0.8)
txt(7.0, 2.55, "Without validation gate, the hallucinated chain would produce:",
    fs=7.5, ha="center", fontweight="bold", color="#888")
mono(2.0, 2.22, 'chip = db.getChip()  # \u2717 AttributeError: dbDatabase has no getChip() in Python API',
     fs=7, color=C_RED)
txt(2.0, 1.92, "\u2192 Runtime crash in OpenROAD shell. The validation gate catches this statically before any code runs.",
    fs=7, color="#888")

# ════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════
items = [
    (BG_LLM, "LLM Call"), (BG_GATE, "Validation"), (BG_FIX, "Correction"),
    (BG_RAG, "RAG Retrieval"), (BG_PASS, "PASS"),
]
lx = 1.0
for color, label in items:
    rbox(lx, 0.85, 0.3, 0.22, color, ec=C_BORDER, lw=0.5)
    txt(lx + 0.38, 1.02, label, fs=6.5, color=C_ARROW, va="center")
    lx += 2.7

# ── Save ────────────────────────────────────────────────────────────────────
plt.tight_layout()
d = "causal_verifier_4_2"
plt.savefig(f"{d}/fig_demo_compact.pdf", bbox_inches="tight", dpi=300)
plt.savefig(f"{d}/fig_demo_compact.png", bbox_inches="tight", dpi=300)
print(f"Saved: {d}/fig_demo_compact.pdf  &  .png")
