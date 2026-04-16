"""make_causal_viz.py — Generate two publication figures from real causal-agent run data.

Image 1: causal_pipeline_viz.png
  Two prompts with 5-node chains, each showing 3 rows:
    Row 1 — causal chain (type nodes)
    Row 2 — RAG-retrieved APIs per edge (correct=teal, wrong=red ?)
    Row 3 — Verifier correction (corrected API in purple)

Image 2: causal_graphs_5cases.png
  Five causal graphs from real run, drawn as directed graphs.

Run:
  python causal_verifier/make_causal_viz.py
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Load real data
# ─────────────────────────────────────────────────────────────────────────────

_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "result", "causal_agent_run.csv")

rows = []
with open(_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

def parse_chain(s):
    return [c.strip() for c in s.split("->") if c.strip()]

def parse_apis(s):
    """'A->B: method | C->D: method' → list of method strings (one per edge)."""
    if not s:
        return []
    result = []
    for part in s.split("|"):
        part = part.strip()
        if ":" in part:
            result.append(part.split(":", 1)[1].strip())
    return result

def short(type_name):
    """'odb.dbTechLayerCutClassRule' → 'dbTechLayerCutClassRule'."""
    return type_name.split(".")[-1]

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

C_NODE_FACE  = "#FFFFFF"
C_NODE_EDGE  = "#2C3E50"
C_ARROW      = "#2980B9"
C_CORRECT    = "#148F77"   # teal — correct RAG API
C_WRONG      = "#C0392B"   # red  — wrong/missing RAG API
C_FIX        = "#8E44AD"   # purple — verifier correction
C_TITLE      = "#E74C3C"   # orange-red for prompt title
C_PASS       = "#1E8449"
C_FAIL       = "#C0392B"
C_ROW_LABEL  = "#555555"

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_node(ax, cx, cy, label, width=1.55, height=0.38,
              facecolor=C_NODE_FACE, edgecolor=C_NODE_EDGE,
              fontsize=7.5, bold=False):
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2), width, height,
        boxstyle="round,pad=0.04",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color="#1A1A1A", zorder=4, clip_on=False)

def draw_arrow(ax, x0, x1, y, color=C_ARROW):
    ax.annotate("",
        xy=(x1 - 0.83, y), xytext=(x0 + 0.83, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
        zorder=2,
    )

def draw_api_label(ax, cx, cy_node, label, color, wrong=False, fontsize=7):
    """Draw the API method label below the arrow midpoint."""
    box_y = cy_node - 0.50
    ax.text(cx, box_y, label,
            ha="center", va="top", fontsize=fontsize,
            color=color,
            fontweight="bold",
            zorder=5)
    if wrong:
        ax.text(cx + len(label) * 0.045 + 0.12, box_y + 0.02, "✗",
                ha="left", va="top", fontsize=8, color=C_WRONG, zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 1 — Two prompts, 3-row pipeline each
# ─────────────────────────────────────────────────────────────────────────────

# Case 4: 5-node chain, L3 FAIL — get_db_tech wrong, correct = getTech
# Case 7: 5-node chain, PASS after 1 re_generate

CASE4 = rows[3]   # index 3 = case 4
CASE7 = rows[6]   # index 6 = case 7

def _pipeline_data(row, corrections):
    """Build structured data for one prompt's 3-row visualisation.

    corrections : list of (edge_idx, correct_method) for mismatched edges.
    Returns dict with chain, apis, wrong_edges, corrections.
    """
    chain = parse_chain(row["chain"])
    apis  = parse_apis(row["node_apis"])
    # pad if shorter
    while len(apis) < len(chain) - 1:
        apis.append("?")
    return dict(chain=chain, apis=apis,
                wrong_edges={idx for idx, _ in corrections},
                fix_map={idx: m for idx, m in corrections})

# Case 4 corrections: edge index 1 (dbDatabase→dbTech) used get_db_tech → should be getTech
p4 = _pipeline_data(CASE4, corrections=[(1, "getTech")])
# Case 7: no corrections needed — all edges correct
p7 = _pipeline_data(CASE7, corrections=[])


def draw_pipeline(ax, data, prompt_text, result_tag,
                  x_start=0.5, y_top=0.0, gap_x=2.0):
    """Draw 3-row pipeline for one prompt onto ax.

    y_top = top y of row 1. Rows spaced 1.2 units apart.
    """
    chain  = data["chain"]
    apis   = data["apis"]
    n      = len(chain)
    xs     = [x_start + i * gap_x for i in range(n)]
    rows_y = [y_top, y_top - 1.30, y_top - 2.60]
    row_labels = ["Causal chain", "RAG-retrieved APIs", "Verifier correction"]

    # ── Prompt title ──────────────────────────────────────────────────────────
    tag_color = C_PASS if result_tag == "PASS" else C_FAIL
    ax.text(xs[0] - 0.7, y_top + 0.55,
            f"{prompt_text}",
            fontsize=9, fontweight="bold", color=C_TITLE, va="bottom")
    ax.text(xs[-1] + 0.7, y_top + 0.55,
            result_tag,
            fontsize=9, fontweight="bold", color=tag_color, va="bottom",
            ha="right")

    for row_idx, (row_y, row_label) in enumerate(zip(rows_y, row_labels)):
        # Row label on the left
        ax.text(xs[0] - 0.85, row_y, row_label,
                fontsize=7, color=C_ROW_LABEL, va="center",
                style="italic", ha="right")

        # Draw nodes
        for i, (x, node) in enumerate(zip(xs, chain)):
            draw_node(ax, x, row_y, short(node), width=1.55, height=0.38)

        # Draw arrows + API labels
        for i in range(n - 1):
            x0, x1 = xs[i], xs[i + 1]
            mid_x   = (x0 + x1) / 2
            wrong   = (i in data["wrong_edges"])
            fixed   = (i in data.get("fix_map", {}))

            if row_idx == 0:
                # Row 1: plain arrows, no API label
                draw_arrow(ax, x0, x1, row_y, color=C_ARROW)

            elif row_idx == 1:
                # Row 2: RAG-retrieved API (correct or wrong)
                method = apis[i] if i < len(apis) else "?"
                color  = C_WRONG if wrong else C_CORRECT
                draw_arrow(ax, x0, x1, row_y,
                           color=C_WRONG if wrong else C_ARROW)
                if wrong:
                    # Show wrong method with ✗ marker
                    ax.text(mid_x, row_y - 0.25,
                            method, ha="center", va="top",
                            fontsize=6.5, color=C_WRONG, fontweight="bold")
                    ax.text(mid_x, row_y - 0.44,
                            "RAG could not\nretrieve correct API",
                            ha="center", va="top",
                            fontsize=5.5, color=C_WRONG, style="italic")
                else:
                    ax.text(mid_x, row_y - 0.25,
                            method, ha="center", va="top",
                            fontsize=6.5, color=C_CORRECT, fontweight="bold")

            elif row_idx == 2:
                # Row 3: corrected API
                draw_arrow(ax, x0, x1, row_y,
                           color=C_FIX if fixed else C_ARROW)
                if fixed:
                    correct_method = data["fix_map"][i]
                    # Purple highlighted box for LLM-suggested correction
                    bw, bh = 1.05, 0.28
                    fix_box = FancyBboxPatch(
                        (mid_x - bw/2, row_y - 0.52),
                        bw, bh,
                        boxstyle="round,pad=0.04",
                        facecolor="#D2B4DE", edgecolor=C_FIX, linewidth=1.3,
                        zorder=5,
                    )
                    ax.add_patch(fix_box)
                    ax.text(mid_x, row_y - 0.38,
                            correct_method,
                            ha="center", va="center",
                            fontsize=6.5, color=C_FIX, fontweight="bold",
                            zorder=6)
                    ax.text(mid_x + bw/2 + 0.08, row_y - 0.38,
                            "Verifier catches\nthis and verifies",
                            ha="left", va="center",
                            fontsize=5.5, color=C_FIX, style="italic")
                else:
                    method = apis[i] if i < len(apis) else "?"
                    ax.text(mid_x, row_y - 0.25,
                            method, ha="center", va="top",
                            fontsize=6.5, color=C_CORRECT, fontweight="bold")


fig1, ax1 = plt.subplots(figsize=(14, 4.2))
ax1.set_xlim(-1.2, 10.5)
ax1.set_ylim(-3.2, 1.2)
ax1.axis("off")
ax1.set_aspect("equal")

# Case 4 only
draw_pipeline(
    ax1, p4,
    prompt_text="Get the cut layer class rule of layer 'metal8'",
    result_tag="FAIL → Correction",
    x_start=0.5, y_top=0.0, gap_x=2.1,
)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=C_NODE_FACE, edgecolor=C_NODE_EDGE, label="Type node"),
    mpatches.Patch(facecolor=C_CORRECT,   edgecolor=C_CORRECT,   label="Correct API (RAG)"),
    mpatches.Patch(facecolor=C_WRONG,     edgecolor=C_WRONG,     label="Wrong API (RAG miss)"),
    mpatches.Patch(facecolor="#D2B4DE",   edgecolor=C_FIX,       label="Verifier correction"),
]
fig1.legend(handles=legend_elements,
            loc="lower center", bbox_to_anchor=(0.5, -0.04),
            ncol=4, fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")

fig1.suptitle("Causal Pipeline Visualisation",
              fontsize=11, fontweight="bold", y=1.01)
fig1.subplots_adjust(bottom=0.12)

out1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "result", "causal_pipeline_viz.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out1}")
plt.close(fig1)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 2 — Five causal graphs
# ─────────────────────────────────────────────────────────────────────────────

# Pick 5 diverse cases by chain shape and outcome
PICKS = [
    # (row_index, label_override)
    (3,  None),   # 5-node: Design→dbDatabase→dbTech→dbTechLayer→dbTechLayerCutClassRule FAIL
    (6,  None),   # 5-node: Design→dbBlock→dbInst→dbITerm→dbNet PASS
    (12, None),   # 4-node: Design→dbBlock→dbInst→dbMaster PASS
    (14, None),   # 4-node: Design→dbBlock→dbBTerm→dbNet FAIL
    (18, None),   # 4-node: Design→dbBlock→dbInst→dbITerm PASS
]

fig2, axes = plt.subplots(1, 5, figsize=(18, 3.8))
fig2.suptitle("Causal Chain Graphs — Five Representative Prompts",
              fontsize=11, fontweight="bold", y=1.02)

NODE_W, NODE_H = 1.4, 0.42

for ax, (ridx, _) in zip(axes, PICKS):
    row    = rows[ridx]
    chain  = parse_chain(row["chain"])
    apis   = parse_apis(row["node_apis"])
    ora    = row["openroad_result"].strip()
    passed = (ora == "PASS")

    n      = len(chain)
    gap    = 2.0
    xs     = [i * gap for i in range(n)]
    y      = 0.5

    ax.set_xlim(-0.9, xs[-1] + 0.9)
    ax.set_ylim(-0.9, 1.6)
    ax.axis("off")
    ax.set_aspect("equal")

    # Prompt (truncated)
    prompt_short = row["prompt"][:42] + ("…" if len(row["prompt"]) > 42 else "")
    ax.set_title(prompt_short, fontsize=6.8, wrap=True,
                 color=C_TITLE, fontweight="bold", pad=4)

    # Result badge
    badge_color = C_PASS if passed else C_FAIL
    badge_label = "PASS" if passed else "FAIL"
    ax.text(xs[-1] + 0.85, 1.45, badge_label,
            ha="right", va="top", fontsize=7.5,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=badge_color, edgecolor=badge_color))

    # Nodes
    for i, (x, node) in enumerate(zip(xs, chain)):
        node_color = "#EBF5FB" if i == 0 else "#FDFEFE"
        edge_color = "#1A5276" if i == 0 else C_NODE_EDGE
        box = FancyBboxPatch(
            (x - NODE_W/2, y - NODE_H/2), NODE_W, NODE_H,
            boxstyle="round,pad=0.04",
            facecolor=node_color, edgecolor=edge_color, linewidth=1.3,
            zorder=3,
        )
        ax.add_patch(box)
        label = short(node)
        # Abbreviate very long names
        if len(label) > 18:
            label = label[:16] + "…"
        ax.text(x, y, label, ha="center", va="center",
                fontsize=6.0, fontweight="bold", color="#1A1A1A", zorder=4)

    # Arrows + API method labels
    for i in range(n - 1):
        x0, x1 = xs[i], xs[i + 1]
        mid    = (x0 + x1) / 2
        method = apis[i] if i < len(apis) else "?"

        # Check if this edge had a failure
        verdict = row["causal_verdict"]
        edge_failed = ("FAIL" in verdict and
                       short(chain[i]) in verdict.replace("→", "->") or
                       short(chain[i+1]) in verdict)
        a_color = C_WRONG if (not passed and edge_failed and i == n-2) else C_ARROW

        ax.annotate("",
            xy=(x1 - NODE_W/2 - 0.05, y),
            xytext=(x0 + NODE_W/2 + 0.05, y),
            arrowprops=dict(arrowstyle="-|>", color=a_color, lw=1.3),
            zorder=2,
        )
        m_color = C_WRONG if a_color == C_WRONG else C_CORRECT
        ax.text(mid, y - 0.34, f".{method}()",
                ha="center", va="top", fontsize=5.5,
                color=m_color, style="italic")

    # Verifier verdict note
    if not passed:
        short_issue = row["causal_verdict"].replace("FAIL(L3): ", "")[:55]
        ax.text((xs[0] + xs[-1]) / 2, -0.70,
                f"L3: {short_issue}",
                ha="center", va="bottom", fontsize=5.2,
                color=C_FAIL, style="italic", wrap=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "result", "causal_graphs_5cases.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out2}")
plt.close(fig2)

print("Done.")
