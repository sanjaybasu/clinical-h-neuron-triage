#!/usr/bin/env python3
"""Generate the graphical abstract for the h-neuron triage paper.

Three-panel landscape figure (Nature Medicine style):
  a – Circuit diagram: fine-tuning shifts h-neurons to new layers
  b – Sensitivity collapse bar chart (base vs fine-tuned)
  c – 2×2 factorial: Arm B vs Arm D across alphas (physician set)
"""

import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

OUTPUT_DIR  = Path("output")
NB_FIGURES  = Path("/Users/sanjaybasu/waymark-local/notebooks/h_neuron_triage/figures")
NB_FIGURES.mkdir(parents=True, exist_ok=True)

# ── colour palette ──
BLUE   = "#3274A1"  # Arm A/B (base)
ORANGE = "#C4681A"  # Arm C/D (fine-tuned)
GRAY   = "#666666"
GREEN  = "#2CA02C"

# ── known metrics ──
BASE_SENS    = 0.409
FT_SENS      = 0.053
COLLAPSE_PCT = 87

# Base h-neurons: layers 10-14, n=213
BASE_LAYER_LO, BASE_LAYER_HI, BASE_N = 10, 14, 213
# Medical h-neurons: layers 23-30, n=5
MED_LAYER_LO,  MED_LAYER_HI,  MED_N  = 23, 30, 5
N_LAYERS = 32


def draw_panel_a(ax):
    """Circuit diagram: base-model neurons vs fine-tuned neurons."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("a", loc="left", fontsize=11, fontweight="bold", pad=4)

    # ── subtitle ──
    ax.text(5, 9.6, "Fine-tuning installs new over-compliance circuit",
            ha="center", va="top", fontsize=9, fontweight="bold", color="black")

    # ──────────────────────────────────────────
    # Neuron layer bars  (mini-histograms)
    # ──────────────────────────────────────────
    bar_y    = 5.5   # centre y
    bar_h    = 1.8
    bar_w    = 0.18  # width per layer slot
    n_show   = N_LAYERS

    def draw_layer_bars(cx, highlight_lo, highlight_hi, color, n_label):
        """Draw a row of 32 layer bars, highlighted range coloured."""
        for li in range(n_show):
            x0 = cx - (n_show * bar_w) / 2 + li * bar_w
            h  = 0.4  # background bar height
            fill = color if highlight_lo <= li <= highlight_hi else "#E0E0E0"
            ax.bar(x0 + bar_w / 2, h, width=bar_w * 0.85, bottom=bar_y - h / 2,
                   color=fill, edgecolor="none", zorder=2)
        # bracket
        lo_x = cx - (n_show * bar_w) / 2 + highlight_lo * bar_w
        hi_x = cx - (n_show * bar_w) / 2 + (highlight_hi + 1) * bar_w
        mid_x = (lo_x + hi_x) / 2
        ax.annotate("", xy=(lo_x, bar_y - 0.85), xytext=(hi_x, bar_y - 0.85),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
        return mid_x

    cx_base = 2.6
    cx_ft   = 7.4

    mid_base = draw_layer_bars(cx_base, BASE_LAYER_LO, BASE_LAYER_HI, BLUE, BASE_N)
    mid_ft   = draw_layer_bars(cx_ft,   MED_LAYER_LO,  MED_LAYER_HI,  ORANGE, MED_N)

    # ── labels under bars ──
    ax.text(cx_base, bar_y - 1.4,
            f"Base model\nlayers {BASE_LAYER_LO}–{BASE_LAYER_HI},  n = {BASE_N}",
            ha="center", va="top", fontsize=7.5, color=BLUE)
    ax.text(cx_ft, bar_y - 1.4,
            f"Fine-tuned model\nlayers {MED_LAYER_LO}–{MED_LAYER_HI},  n = {MED_N}",
            ha="center", va="top", fontsize=7.5, color=ORANGE)

    # ── Jaccard label (centre, above arrow) ──
    ax.text(5.0, bar_y + 1.1, "Jaccard overlap  =  0.000",
            ha="center", va="bottom", fontsize=7.5, color=GRAY,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, alpha=0.8))

    # ── horizontal arrow: base → fine-tuned ──
    ax.annotate("", xy=(6.5, bar_y + 0.6), xytext=(3.5, bar_y + 0.6),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
    # QLoRA box on arrow
    ax.text(5.0, bar_y + 0.6,
            "QLoRA fine-tuning\n1,280 messages  ·  8.3% hazard",
            ha="center", va="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9E6", ec="#C8A000", lw=0.8))

    # ── sensitivity labels ──
    ax.text(cx_base, 2.0, f"Sensitivity: {BASE_SENS:.3f}",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=BLUE)
    ax.text(cx_ft, 2.0, f"Sensitivity: {FT_SENS:.3f}",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=ORANGE)
    ax.text(5.0, 1.15, f"↓ {COLLAPSE_PCT}% collapse",
            ha="center", va="center", fontsize=8, color=ORANGE,
            style="italic", fontweight="bold")

    # ── CETT box ──
    ax.text(5.0, 0.35,
            "CETT sparse probing identifies over-compliance neurons",
            ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREEN, lw=1.0))


def draw_panel_b(ax):
    """Bar chart: sensitivity collapse."""
    with open(OUTPUT_DIR / "finetuned_baseline_metrics.json") as f:
        m = json.load(f)
    ft_sens   = m.get("physician_sensitivity", FT_SENS)
    ft_lo     = m.get("physician_sens_lo",     max(0, ft_sens - 0.04))
    ft_hi     = m.get("physician_sens_hi",     min(1, ft_sens + 0.04))

    # Approximate CIs for base model (from triage_metrics_by_alpha.csv alpha=1.0)
    base_sens = BASE_SENS
    base_lo, base_hi = 0.329, 0.493  # Wilson 95% CI for 54/132

    bars = ax.bar([0, 1], [base_sens, ft_sens],
                  color=[BLUE, ORANGE], width=0.5, zorder=3)
    ax.errorbar([0, 1], [base_sens, ft_sens],
                yerr=[[base_sens-base_lo, ft_sens-ft_lo],
                       [base_hi-base_sens, ft_hi-ft_sens]],
                fmt="none", color="black", capsize=4, linewidth=1.2)

    # Significance bracket
    y_top = max(base_hi, ft_hi) + 0.04
    ax.plot([0, 0, 1, 1], [base_hi+0.015, y_top, y_top, ft_hi+0.015],
            color="black", linewidth=0.9)
    ax.text(0.5, y_top + 0.01, f"↓ {COLLAPSE_PCT}%", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=ORANGE)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Base\nmodel", "Fine-tuned\nmodel"], fontsize=9)
    ax.set_ylabel("Sensitivity (α=1.0, 95% Wilson CI)", fontsize=8)
    ax.set_ylim(0, min(1.0, y_top + 0.08))
    ax.set_title("b", loc="left", fontsize=11, fontweight="bold", pad=4)
    ax.text(5.0/10, 0.96, "Sensitivity collapse", ha="center", va="top",
            transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def draw_panel_c(ax):
    """2×2 factorial: Arm B (base+medical) vs Arm D (fine-tuned+medical), physician set."""
    summary_path = OUTPUT_DIR / "two_by_two_summary.json"
    if not summary_path.exists():
        ax.text(0.5, 0.5, "data pending", ha="center", va="center",
                transform=ax.transAxes); return

    with open(summary_path) as f:
        summary = json.load(f)
    df_data = {(r["arm"], r["dataset"], r["alpha"]): r for r in summary}

    alpha_order = [0.0, 1.0, 3.0]

    for arm, color, marker, ls, label in [
        ("B_base_medical",     BLUE,   "s", "-",  "Arm B  base + medical"),
        ("D_finetuned_medical", ORANGE, "D", "--", "Arm D  fine-tuned + medical"),
    ]:
        pts = [df_data.get((arm, "physician", a)) for a in alpha_order]
        alphas = [a for a, p in zip(alpha_order, pts) if p is not None]
        sens   = [p["sensitivity"] for p in pts if p is not None]
        lo     = [p["sensitivity_ci"][0] for p in pts if p is not None]
        hi     = [p["sensitivity_ci"][1] for p in pts if p is not None]

        if not alphas:
            continue
        yerr = [np.array(sens) - np.array(lo), np.array(hi) - np.array(sens)]
        ax.errorbar(alphas, sens, yerr=yerr, color=color,
                    marker=marker, linestyle=ls, linewidth=1.8,
                    markersize=6, capsize=3, label=label)

    ax.set_xticks(alpha_order)
    ax.set_xticklabels(["0.0\n(suppress)", "1.0\n(baseline)", "3.0\n(amplify)"], fontsize=8)
    ax.set_xlabel("Alpha", fontsize=8)
    ax.set_ylabel("Sensitivity  (95% Wilson CI)", fontsize=8)
    ax.set_ylim(0, 0.82)
    ax.set_xlim(-0.5, 3.5)
    ax.set_title("b", loc="left", fontsize=11, fontweight="bold", pad=4)  # relabeled below
    ax.text(5.0/10, 0.96, "2×2 factorial — key arms", ha="center", va="top",
            transform=ax.transAxes, fontsize=9, fontweight="bold")

    # Spearman annotation — top right, clear of data
    ax.text(0.97, 0.97, "ρ = −1.000\n(Arm B, base)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, alpha=0.9))
    ax.set_title("c", loc="left", fontsize=11, fontweight="bold", pad=4)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncol=1, fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


def make_graphical_abstract():
    # Nature Medicine graphical abstract: 170 mm × 85 mm ≈ 6.7 × 3.35 in
    # We use 14 × 5.5 in for better on-screen rendering
    fig = plt.figure(figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    # ── main title ──
    fig.text(0.5, 0.97,
             "Clinical fine-tuning installs over-compliance neurons that cause missed medical emergencies",
             ha="center", va="top", fontsize=10.5, fontweight="bold")

    # ── 3-column layout: a wide, b narrow, c narrow ──
    gs = fig.add_gridspec(1, 3, left=0.02, right=0.98,
                          bottom=0.14, top=0.89,
                          wspace=0.28,
                          width_ratios=[1.5, 1, 1])

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    out = NB_FIGURES / "graphical_abstract.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    import os
    os.chdir("/Users/sanjaybasu/waymark-local/packaging/h_neuron_triage")
    make_graphical_abstract()
