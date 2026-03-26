#!/usr/bin/env python3
"""Generate Figure 1: CETT pipeline schematic (three-panel matplotlib figure).

Panel A – SwiGLU FFN decomposition (architecture diagram)
Panel B – H-neuron layer distribution (peak zone layers 10–14)
Panel C – Graded activation scaling → sensitivity curve

Changes from previous version:
  - No dashed vertical separators between panels
  - No connecting arrows between panels
  - A/B/C labels positioned high above panel content
  - All text fits within its bounding box
  - α labels placed off the line (α=0 and α=3 only, with offsets)
  - x-axis label below tick labels (no overlap)
"""

import json, math, shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
REPO      = Path("/Users/sanjaybasu/waymark-local")
OUT_DIR   = REPO / "packaging/h_neuron_triage/output"
NB_FIG    = REPO / "notebooks/h_neuron_triage/figures"
PKG_FIG   = REPO / "packaging/h_neuron_triage/figures"
NB_FIG.mkdir(parents=True, exist_ok=True)
PKG_FIG.mkdir(parents=True, exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────────────
BLUE   = "#3274A1"
ORANGE = "#C4681A"
GREEN  = "#2CA02C"
GRAY   = "#555555"
LTBLUE = "#D6E4F0"
LTORANGE = "#FDEBD0"

# ── data ─────────────────────────────────────────────────────────────────────
h_data   = json.load(open(OUT_DIR / "h_neurons.json"))
abl_raw  = json.load(open(OUT_DIR / "ablation_results.json"))

# Panel B
n_layers = h_data["n_layers"]
layer_counts = Counter(h["layer"] for h in h_data["h_neurons"])
counts_arr   = np.array([layer_counts.get(l, 0) for l in range(n_layers)])
auc_mean = h_data["cv_auc_mean"]
auc_std  = h_data["cv_auc_std"]
auc_lo   = auc_mean - 1.96 * auc_std
auc_hi   = auc_mean + 1.96 * auc_std

# Panel C
def wilson(k, n, z=1.96):
    if not n: return 0.0, 0.0
    p = k / n; dm = 1 + z**2/n
    ct = (p + z**2/(2*n)) / dm
    w  = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / dm
    return max(0, ct - w), min(1, ct + w)

bkts = defaultdict(list)
for r in abl_raw:
    bkts[(r["dataset"], r["alpha"])].append(r)

phys_rows = []
for (ds, a), rows in sorted(bkts.items(), key=lambda x: x[0][1]):
    if ds != "physician": continue
    tp = sum(1 for r in rows if r["detection_truth"]==1 and r["predicted_detection"]==1)
    fn = sum(1 for r in rows if r["detection_truth"]==1 and r["predicted_detection"]==0)
    n  = tp + fn
    s  = tp / n if n else 0
    lo, hi = wilson(tp, n)
    phys_rows.append((a, s, lo, hi))

alphas_c = np.array([r[0] for r in phys_rows])
sens_c   = np.array([r[1] for r in phys_rows])
lo_c     = np.array([r[2] for r in phys_rows])
hi_c     = np.array([r[3] for r in phys_rows])

# ── figure layout ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                          gridspec_kw={"wspace": 0.38})

# ── Panel A label ─────────────────────────────────────────────────────────────
# (drawn before content so it sits at the very top)
for ax, letter, title in zip(axes,
                              ["A", "B", "C"],
                              ["Stage 1 — SwiGLU FFN Decomposition",
                               "Stage 2 — Two-Stage Sparse Probing",
                               "Stage 3 — Graded Activation Scaling"]):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=20)
    ax.text(-0.10, 1.10, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")


# ═════════════════════════════════════════════════════════════════════════════
# PANEL A — SwiGLU architecture diagram
# ═════════════════════════════════════════════════════════════════════════════
ax_a = axes[0]
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 10)
ax_a.axis("off")

def draw_box(ax, cx, cy, w, h, text, fc=LTBLUE, ec=BLUE, fs=8.5, bold=False):
    """Draw a rounded rectangle centred at (cx, cy) with wrapped text."""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.1",
                           facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            zorder=4, wrap=False,
            color="#1a1a1a" if fc != BLUE else "white")

def draw_arrow_down(ax, x, y_top, y_bot, color=BLUE):
    ax.annotate("", xy=(x, y_bot + 0.05), xytext=(x, y_top - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.2, mutation_scale=10), zorder=2)

# Boxes (centred at x=5, stacked top-to-bottom)
nodes = [
    (5, 9.0, 7.0, 0.75, "Hidden state  hₓ",                     LTBLUE,   BLUE,   False),
    (5, 7.7, 7.0, 0.75, "W_gate (gate)  ·  W_up (up)",          LTBLUE,   BLUE,   False),
    (5, 6.4, 7.0, 0.75, "SiLU(W_gate · x)  activation",         LTBLUE,   BLUE,   False),
    (5, 5.1, 7.0, 0.75, "× element-wise multiplication",         LTBLUE,   BLUE,   False),
    (5, 3.8, 7.0, 0.75, "W_down (down projection)",              LTBLUE,   BLUE,   False),
    (5, 2.3, 7.0, 1.00, "CETT Attribution Scores\n/ Neurons",   BLUE,     BLUE,   True),
]
for cx, cy, w, h, txt, fc, ec, bold in nodes:
    draw_box(ax_a, cx, cy, w, h, txt, fc=fc, ec=ec, bold=bold)

# Arrows between consecutive boxes
for i in range(len(nodes) - 1):
    _, y_top, _, h_top, *_ = nodes[i]
    _, y_bot, _, h_bot, *_ = nodes[i+1]
    draw_arrow_down(ax_a, 5, y_top - h_top/2, y_bot + h_bot/2)



# ═════════════════════════════════════════════════════════════════════════════
# PANEL B — h-neuron layer distribution
# ═════════════════════════════════════════════════════════════════════════════
ax_b = axes[1]
PEAK_LO, PEAK_HI = 10, 14
bar_colors = [ORANGE if PEAK_LO <= l <= PEAK_HI else BLUE
              for l in range(n_layers)]

ax_b.axvspan(PEAK_LO - 0.5, PEAK_HI + 0.5, color=ORANGE, alpha=0.10, zorder=0)
ax_b.bar(range(n_layers), counts_arr, color=bar_colors,
         edgecolor="white", linewidth=0.4, width=0.85, zorder=2)

# Axis formatting — no overlap
ax_b.set_xlabel("Transformer layer", fontsize=8.5)
ax_b.set_ylabel("H-neuron count per layer", fontsize=8.5)
ax_b.set_xlim(-0.5, n_layers - 0.5)
ax_b.set_ylim(0, counts_arr.max() * 1.30)
ax_b.set_xticks(range(0, n_layers, 4))
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.grid(axis="y", alpha=0.25, linewidth=0.6)

# AUC annotation — upper right
ax_b.text(0.97, 0.60,
          f"Probe AUC\n{auc_mean:.3f}\n({auc_lo:.3f}–{auc_hi:.3f})",
          transform=ax_b.transAxes, ha="right", va="top",
          fontsize=7.5, color=GRAY,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.9))

# Legend
blue_p   = mpatches.Patch(color=BLUE,   label="Other layers")
orange_p = mpatches.Patch(color=ORANGE, label="Peak zone (10–14)")
ax_b.legend(handles=[blue_p, orange_p], loc="upper left", fontsize=7.5,
            framealpha=0.9)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL C — Sensitivity vs scaling factor α
# ═════════════════════════════════════════════════════════════════════════════
ax_c = axes[2]

ax_c.fill_between(alphas_c, lo_c, hi_c, alpha=0.18, color=BLUE)
ax_c.plot(alphas_c, sens_c, marker="o", color=BLUE,
          linewidth=2.0, markersize=5, zorder=5)

# Baseline dashed line only (no extra vertical lines)
ax_c.axvline(1.0, color=GRAY, linestyle="--", linewidth=0.9, alpha=0.7)

# Label only α=0 and α=3, placed AWAY from the line
# α=0 is upper-left of its point → annotate above and left
ax_c.annotate("α=0\n(suppress)",
              xy=(0.0, sens_c[alphas_c == 0.0][0]),
              xytext=(0.35, sens_c[alphas_c == 0.0][0] + 0.055),
              arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8),
              fontsize=7.5, ha="center", color=GRAY,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9))

# α=1 baseline label — above the dashed line, no overlap
bl = float(sens_c[alphas_c == 1.0][0])
ax_c.text(1.12, bl + 0.045, "α=1\n(baseline)",
          fontsize=7.5, ha="left", va="bottom", color=GRAY,
          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9))

# α=3 label below the point
ax_c.annotate("α=3\n(amplify)",
              xy=(3.0, sens_c[alphas_c == 3.0][0]),
              xytext=(2.60, sens_c[alphas_c == 3.0][0] - 0.060),
              arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8),
              fontsize=7.5, ha="center", color=GRAY,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9))

ax_c.set_xlabel("Scaling factor  α", fontsize=8.5)
ax_c.set_ylabel("Hazard detection sensitivity", fontsize=8.5)
ax_c.set_xlim(-0.2, 3.2)
ax_c.set_ylim(0.28, 0.53)
ax_c.set_xticks([0, 1, 2, 3])
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.grid(True, alpha=0.20)

# ── save ─────────────────────────────────────────────────────────────────────
fig.tight_layout()
STEM = "fig1_cett_pipeline_schematic"
for ext in ("pdf", "png"):
    out_pkg = PKG_FIG / f"{STEM}.{ext}"
    out_nb  = NB_FIG  / f"{STEM}.{ext}"
    fig.savefig(out_pkg, dpi=300, bbox_inches="tight")
    shutil.copy2(out_pkg, out_nb)
    print(f"Saved {out_nb}  ({out_nb.stat().st_size:,} bytes)")
plt.close(fig)
print("Done.")
