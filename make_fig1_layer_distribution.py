#!/usr/bin/env python3
"""Generate Figure 2 (Extended): H-neuron layer distribution histogram.

Orange bars = peak zone (layers 10–14); blue bars = other layers.
Probe AUC annotation placed in the upper-right corner.
"Other layers" legend label placed without arrow overlap.
"""

import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path("/Users/sanjaybasu/waymark-local")
OUTPUT_DIR  = REPO_ROOT / "packaging/h_neuron_triage/output"
NB_FIGURES  = REPO_ROOT / "notebooks/h_neuron_triage/figures"
PKG_FIGURES = REPO_ROOT / "packaging/h_neuron_triage/figures"
NB_FIGURES.mkdir(parents=True, exist_ok=True)
PKG_FIGURES.mkdir(parents=True, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
with open(OUTPUT_DIR / "h_neurons.json") as f:
    data = json.load(f)

h_neurons  = data["h_neurons"]
n_layers   = data["n_layers"]          # 32
n_total    = data["n_h_neurons"]       # 213
pct_total  = data["pct_of_total"]      # 0.046 %
auc_mean   = data["cv_auc_mean"]       # 0.8916
auc_std    = data["cv_auc_std"]        # 0.01858

# 95 % CI: mean ± 1.96 × std  (std of cross-validation fold AUCs)
auc_lo = auc_mean - 1.96 * auc_std
auc_hi = auc_mean + 1.96 * auc_std

# Layer distribution
from collections import Counter
layer_counts = Counter(h["layer"] for h in h_neurons)
layers_arr   = np.arange(n_layers)
counts_arr   = np.array([layer_counts.get(l, 0) for l in layers_arr])

# Peak zone
PEAK_LO, PEAK_HI = 10, 14  # inclusive

# ── figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      300,
})

fig, ax = plt.subplots(figsize=(7.5, 4.0))

BLUE   = "#3274A1"
ORANGE = "#C4681A"

bar_colors = [ORANGE if PEAK_LO <= l <= PEAK_HI else BLUE for l in layers_arr]

# Shaded background for peak zone (drawn first so bars sit on top)
ax.axvspan(PEAK_LO - 0.5, PEAK_HI + 0.5, color=ORANGE, alpha=0.10, zorder=0)
ax.axvline(PEAK_LO - 0.5, color=ORANGE, linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
ax.axvline(PEAK_HI + 0.5, color=ORANGE, linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

# Bars
ax.bar(layers_arr, counts_arr, color=bar_colors, edgecolor="white",
       linewidth=0.5, width=0.85, zorder=2)

# ── Probe AUC annotation (upper-right, well away from bars) ──────────────────
auc_text = (
    f"Probe AUC = {auc_mean:.3f}\n"
    f"(95% CI {auc_lo:.3f}–{auc_hi:.3f})"
)
ax.text(0.98, 0.97, auc_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#AAAAAA", alpha=0.95))

# ── Legend (bottom right, inside axes but below the bars) ────────────────────
blue_patch   = mpatches.Patch(color=BLUE,   label="Other layers")
orange_patch = mpatches.Patch(color=ORANGE, label="Peak zone (layers 10–14)")
ax.legend(handles=[blue_patch, orange_patch],
          loc="upper left", fontsize=8.5, framealpha=0.90)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlabel("Transformer layer")
ax.set_ylabel("H-neuron count per layer")
ax.set_title(
    f"H-neuron layer distribution  (N = {n_total}, {pct_total:.3f}% of total neurons)",
    fontsize=10.5
)
ax.set_xlim(-0.5, n_layers - 0.5)
ax.set_ylim(0, counts_arr.max() * 1.22)
ax.set_xticks(range(0, n_layers, 4))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)

fig.tight_layout()

# ── Save ─────────────────────────────────────────────────────────────────────
STEM = "fig2_h_neuron_layer_distribution"
for ext in ("pdf", "png"):
    out_pkg = PKG_FIGURES / f"{STEM}.{ext}"
    out_nb  = NB_FIGURES  / f"{STEM}.{ext}"
    fig.savefig(out_pkg, dpi=300, bbox_inches="tight")
    shutil.copy2(out_pkg, out_nb)
    print(f"Saved {out_nb}  ({out_nb.stat().st_size:,} bytes)")

plt.close(fig)
print("Done.")
