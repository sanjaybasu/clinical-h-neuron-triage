"""
make_silent_failure_figure.py

Publication-quality "silent failure fingerprint" figure for the H-Neuron Triage manuscript.
Nature Medicine style, two-panel, figsize=(11, 4.5).

Outputs:
  fig4_silent_failure_paradox.pdf  (300 dpi)
  fig4_silent_failure_paradox.png  (300 dpi)

Saved to both:
  packaging/h_neuron_triage/figures/
  notebooks/h_neuron_triage/figures/
"""

import json
import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from scipy.stats import chi2 as chi2_dist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = "/Users/sanjaybasu/waymark-local"
PKG_FIGURES = os.path.join(REPO_ROOT, "packaging/h_neuron_triage/figures")
NB_FIGURES  = os.path.join(REPO_ROOT, "notebooks/h_neuron_triage/figures")
ABLATION_JSON = os.path.join(
    REPO_ROOT, "packaging/h_neuron_triage/output/ablation_results.json"
)
FT_JSON = os.path.join(
    REPO_ROOT,
    "packaging/h_neuron_triage/output/finetuned_baseline_physician.json",
)
STEM = "fig4_silent_failure_paradox"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BASE_COLOR = "#3274A1"    # blue
FT_COLOR   = "#C4681A"   # orange
GRAY       = "#888888"

# ---------------------------------------------------------------------------
# Wilson 95% CI helper
# ---------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    """Return (lower, upper) Wilson score interval."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

# ---------------------------------------------------------------------------
# McNemar (with continuity correction)
# ---------------------------------------------------------------------------
def mcnemar_p(b, c):
    """Two-sided McNemar p with continuity correction."""
    if b + c == 0:
        return 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2_dist.sf(chi2_stat, df=1))

# ---------------------------------------------------------------------------
# Load data and compute McNemar
# ---------------------------------------------------------------------------
with open(ABLATION_JSON) as f:
    ablation_raw = json.load(f)

base_lookup = {
    d["case_index"]: d
    for d in ablation_raw
    if d.get("alpha") == 1.0 and d.get("dataset") == "physician"
}

with open(FT_JSON) as f:
    ft_raw = json.load(f)

ft_lookup = {d["case_index"]: d for d in ft_raw}

common = set(base_lookup.keys()) & set(ft_lookup.keys())
benign = [ci for ci in common if base_lookup[ci]["detection_truth"] == 0]
hazard = [ci for ci in common if base_lookup[ci]["detection_truth"] == 1]

n_benign = len(benign)
n_hazard = len(hazard)

# Sensitivity (hazard cases)
base_sens_k  = sum(1 for ci in hazard if base_lookup[ci]["predicted_detection"] == 1)
ft_sens_k    = sum(1 for ci in hazard if ft_lookup[ci]["detection_pred"] == 1)

base_h_only  = sum(
    1 for ci in hazard
    if base_lookup[ci]["predicted_detection"] == 1 and ft_lookup[ci]["detection_pred"] != 1
)
ft_h_only    = sum(
    1 for ci in hazard
    if base_lookup[ci]["predicted_detection"] != 1 and ft_lookup[ci]["detection_pred"] == 1
)
p_sens = mcnemar_p(base_h_only, ft_h_only)

# Specificity (benign cases)
base_spec_k  = sum(1 for ci in benign if base_lookup[ci]["predicted_detection"] == 0)
ft_spec_k    = sum(1 for ci in benign if ft_lookup[ci]["detection_pred"] == 0)

base_b_only  = sum(
    1 for ci in benign
    if base_lookup[ci]["predicted_detection"] == 0 and ft_lookup[ci]["detection_pred"] != 0
)
ft_b_only    = sum(
    1 for ci in benign
    if base_lookup[ci]["predicted_detection"] != 0 and ft_lookup[ci]["detection_pred"] == 0
)
p_spec = mcnemar_p(base_b_only, ft_b_only)

# Point estimates and CI
base_sens  = base_sens_k / n_hazard
ft_sens    = ft_sens_k   / n_hazard
base_spec  = base_spec_k / n_benign
ft_spec    = ft_spec_k   / n_benign

base_sens_ci  = wilson_ci(base_sens_k,  n_hazard)
ft_sens_ci    = wilson_ci(ft_sens_k,    n_hazard)
base_spec_ci  = wilson_ci(base_spec_k,  n_benign)
ft_spec_ci    = wilson_ci(ft_spec_k,    n_benign)

print(f"Base sensitivity:  {base_sens:.3f}  CI={base_sens_ci}")
print(f"FT sensitivity:    {ft_sens:.3f}  CI={ft_sens_ci}")
print(f"Base specificity:  {base_spec:.3f}  CI={base_spec_ci}")
print(f"FT specificity:    {ft_spec:.3f}  CI={ft_spec_ci}")
print(f"McNemar sens p={p_sens:.4e},  spec p={p_spec:.4e}")

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
})

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    "Fine-tuning improves conventional performance while silently destroying safety",
    fontsize=11, fontweight="bold", y=1.01,
)

# ===========================================================================
# Panel a — Grouped bar chart
# ===========================================================================
x_positions = np.array([0.0, 1.0])   # "Sensitivity", "Specificity"
bar_width = 0.32
offset     = bar_width / 2 + 0.03

# Bar heights
heights = {
    "base_sens":  base_sens,
    "ft_sens":    ft_sens,
    "base_spec":  base_spec,
    "ft_spec":    ft_spec,
}

# Error bar arrays [lower_err, upper_err]
def err_arrays(val, ci):
    return [[val - ci[0]], [ci[1] - val]]

bars_base_sens = ax_a.bar(
    x_positions[0] - offset, base_sens, bar_width,
    color=BASE_COLOR, alpha=0.9, zorder=3,
    label="Base model"
)
bars_ft_sens = ax_a.bar(
    x_positions[0] + offset, ft_sens, bar_width,
    color=FT_COLOR, alpha=0.9, zorder=3,
    label="Fine-tuned model"
)
bars_base_spec = ax_a.bar(
    x_positions[1] - offset, base_spec, bar_width,
    color=BASE_COLOR, alpha=0.9, zorder=3,
)
bars_ft_spec = ax_a.bar(
    x_positions[1] + offset, ft_spec, bar_width,
    color=FT_COLOR, alpha=0.9, zorder=3,
)

# Error bars
ax_a.errorbar(
    x_positions[0] - offset, base_sens,
    yerr=err_arrays(base_sens, base_sens_ci),
    fmt="none", color="black", capsize=3, linewidth=1.2, zorder=4,
)
ax_a.errorbar(
    x_positions[0] + offset, ft_sens,
    yerr=err_arrays(ft_sens, ft_sens_ci),
    fmt="none", color="black", capsize=3, linewidth=1.2, zorder=4,
)
ax_a.errorbar(
    x_positions[1] - offset, base_spec,
    yerr=err_arrays(base_spec, base_spec_ci),
    fmt="none", color="black", capsize=3, linewidth=1.2, zorder=4,
)
ax_a.errorbar(
    x_positions[1] + offset, ft_spec,
    yerr=err_arrays(ft_spec, ft_spec_ci),
    fmt="none", color="black", capsize=3, linewidth=1.2, zorder=4,
)

# ---- Significance brackets ----
def draw_bracket(ax, x1, x2, y_top, label, lw=1.0):
    """Draw a simple significance bracket above two bars."""
    tick_h = 0.018
    ax.plot([x1, x1, x2, x2], [y_top - tick_h, y_top, y_top, y_top - tick_h],
            color="black", lw=lw, zorder=5)
    ax.text((x1 + x2) / 2, y_top + 0.008, label,
            ha="center", va="bottom", fontsize=7.5, zorder=5)

# Sensitivity bracket — p<0.0001
y_sens_bracket = max(base_sens_ci[1], ft_sens_ci[1]) + 0.10
draw_bracket(ax_a,
             x_positions[0] - offset,
             x_positions[0] + offset,
             y_sens_bracket,
             "p<0.0001 ***")

# Specificity bracket
y_spec_bracket = max(base_spec_ci[1], ft_spec_ci[1]) + 0.06
draw_bracket(ax_a,
             x_positions[1] - offset,
             x_positions[1] + offset,
             y_spec_bracket,
             "p<0.0001 ***")

# ---- Relative change annotations ----
# Place text ABOVE the error bar cap (+ small gap) so it never overlaps the whisker
rel_sens_decline = abs(base_sens - ft_sens) / base_sens * 100
ax_a.text(
    x_positions[0] + offset,
    ft_sens_ci[1] + 0.05,
    f"down {rel_sens_decline:.0f}% relative",
    ha="center", va="bottom", fontsize=7.5, color=FT_COLOR, fontweight="bold"
)

rel_spec_gain = abs(ft_spec - base_spec) / base_spec * 100
ax_a.text(
    x_positions[1] - offset,
    base_spec_ci[1] + 0.05,
    f"up {rel_spec_gain:.0f}% relative",
    ha="center", va="bottom", fontsize=7.5, color=BASE_COLOR, fontweight="bold"
)

# Axes formatting
ax_a.set_xlim(-0.6, 1.6)
ax_a.set_ylim(0, 1.1)
ax_a.set_xticks(x_positions)
ax_a.set_xticklabels(["Sensitivity", "Specificity"], fontsize=9)
ax_a.set_ylabel("Proportion (95% Wilson CI)", fontsize=9)
ax_a.set_title("Standard metrics obscure catastrophic safety failure",
               fontsize=10, fontweight="bold", pad=8)
ax_a.legend(loc="upper left", frameon=False, fontsize=7.5)

# Panel label
ax_a.text(-0.14, 1.06, "a", transform=ax_a.transAxes,
          fontsize=13, fontweight="bold", va="top", ha="left")

# ===========================================================================
# Panel b — Operating point scatter
# ===========================================================================

# Gray diamond systems
systems = [
    ("GPT-5.1 Unassisted",   0.985, 0.661),
    ("GPT-5.1 Safety Prompt", 0.956, 0.836),
    ("Guardrail Classifier",  0.500, 0.977),
    ("CQL Controller",        0.691, 0.955),
]

for name, sp, se in systems:
    ax_b.scatter(sp, se, marker="D", color=GRAY, s=55, zorder=4)
    # Adjust label position to avoid crowding
    if name == "GPT-5.1 Unassisted":
        ax_b.text(sp - 0.01, se - 0.055, name, fontsize=6.5, ha="right",
                  color="#555555")
    elif name == "GPT-5.1 Safety Prompt":
        ax_b.text(sp - 0.01, se + 0.025, name, fontsize=6.5, ha="right",
                  color="#555555")
    elif name == "Guardrail Classifier":
        ax_b.text(sp + 0.015, se - 0.045, name, fontsize=6.5, ha="left",
                  color="#555555")
    elif name == "CQL Controller":
        ax_b.text(sp + 0.015, se - 0.045, name, fontsize=6.5, ha="left",
                  color="#555555")

# Base and Fine-tuned Llama
ax_b.scatter(base_spec, base_sens, marker="o", color=BASE_COLOR, s=80, zorder=5,
             label="Base model (Llama-8B)")
ax_b.scatter(ft_spec, ft_sens, marker="s", color=FT_COLOR, s=80, zorder=5,
             label="Fine-tuned model (Llama-8B)")

# Labels for base and FT
ax_b.text(base_spec - 0.015, base_sens + 0.028, "Base model",
          fontsize=7, ha="right", color=BASE_COLOR, fontweight="bold")
ax_b.text(ft_spec + 0.015, ft_sens + 0.028, "Fine-tuned model",
          fontsize=7, ha="left", color=FT_COLOR, fontweight="bold")

# Arrow from base to fine-tuned
ax_b.annotate(
    "",
    xy=(ft_spec, ft_sens),
    xytext=(base_spec, base_sens),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#C4681A",
        lw=2.0,
        mutation_scale=14,
    ),
    zorder=6,
)

# Arrow label midpoint
mid_x = (base_spec + ft_spec) / 2 + 0.02
mid_y = (base_sens + ft_sens) / 2 + 0.04
ax_b.text(mid_x, mid_y, "QLoRA\nfine-tuning", fontsize=7.5, ha="left",
          color=FT_COLOR, fontweight="bold", linespacing=1.3)

# Ideal trajectory arrow (lower-left toward upper-right)
ideal_x_start, ideal_y_start = 0.50, 0.55
ideal_x_end,   ideal_y_end   = 0.78, 0.85
ax_b.annotate(
    "",
    xy=(ideal_x_end, ideal_y_end),
    xytext=(ideal_x_start, ideal_y_start),
    arrowprops=dict(
        arrowstyle="-|>",
        color=GRAY,
        lw=1.3,
        linestyle="dashed",
        mutation_scale=11,
    ),
    zorder=3,
)
ax_b.text(
    (ideal_x_start + ideal_x_end) / 2 - 0.05,
    (ideal_y_start + ideal_y_end) / 2 - 0.07,
    "Ideal fine-tuning\ntrajectory",
    fontsize=6.5, style="italic", color=GRAY, ha="center", linespacing=1.3,
)

ax_b.set_xlim(0.30, 1.05)
ax_b.set_ylim(0.00, 1.05)
ax_b.set_xlabel("Specificity (Benign Classification)", fontsize=9)
ax_b.set_ylabel("Sensitivity (Hazard Detection)", fontsize=9)
ax_b.set_title("Fine-tuning moves in the wrong direction",
               fontsize=10, fontweight="bold", pad=8)

# Gray diamond legend entry
gray_diamond = mpatches.Patch(facecolor=GRAY, label="Comparison systems")
handles, labels = ax_b.get_legend_handles_labels()
# lower-left: fine-tuned model is lower-right (spec~0.97, sens~0.05); arrow is there too
ax_b.legend(
    handles=[gray_diamond] + handles,
    loc="lower left",
    frameon=False,
    fontsize=7.0,
)

# Panel label
ax_b.text(-0.14, 1.04, "b", transform=ax_b.transAxes,
          fontsize=13, fontweight="bold", va="top", ha="left")

# ===========================================================================
# Save
# ===========================================================================
fig.tight_layout(rect=[0, 0, 1, 0.98], pad=1.4, w_pad=2.5)

os.makedirs(PKG_FIGURES, exist_ok=True)
os.makedirs(NB_FIGURES,  exist_ok=True)

for ext in ("pdf", "png"):
    fname = f"{STEM}.{ext}"
    out_pkg = os.path.join(PKG_FIGURES, fname)
    out_nb  = os.path.join(NB_FIGURES,  fname)
    fig.savefig(out_pkg, dpi=300, bbox_inches="tight")
    shutil.copy2(out_pkg, out_nb)
    size_pkg = os.path.getsize(out_pkg)
    size_nb  = os.path.getsize(out_nb)
    print(f"Saved {out_pkg}  ({size_pkg:,} bytes)")
    print(f"Saved {out_nb}  ({size_nb:,} bytes)")

plt.close(fig)
print("Done.")
