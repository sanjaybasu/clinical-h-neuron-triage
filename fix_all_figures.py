#!/usr/bin/env python3
"""Fix all figure issues:
  Fig 2  – right panel (realworld) was blank; compute from ablation_results.json
  Fig 3  – add magnified inset (axes 0–1 preserved, zoom box shows operating range)
  Fig 4  – add magnified inset for clustered operating points; cleaner alpha labels
  Fig 5  – add delta-from-baseline bottom row so differences are visible
  Fig 6  – auto-scale colormap instead of fixed ±0.01
  Fig 9  – move legend / annotation boxes outside data region
"""

import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import numpy as np
import pandas as pd

OUTPUT_DIR  = Path("output")
FIGURES_DIR = Path("figures")
NB_FIGURES  = Path("/Users/sanjaybasu/waymark-local/notebooks/h_neuron_triage/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       300,
})


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    width  = z * math.sqrt(p * (1-p)/n + z**2 / (4*n**2)) / denom
    return max(0.0, center - width), min(1.0, center + width)


def compute_metrics(rows):
    """rows is a list of dicts with detection_truth / predicted_detection."""
    tp = sum(1 for r in rows if r["detection_truth"]==1 and r["predicted_detection"]==1)
    fp = sum(1 for r in rows if r["detection_truth"]==0 and r["predicted_detection"]==1)
    tn = sum(1 for r in rows if r["detection_truth"]==0 and r["predicted_detection"]==0)
    fn = sum(1 for r in rows if r["detection_truth"]==1 and r["predicted_detection"]==0)
    n_haz = tp + fn; n_ben = tn + fp
    sens = tp / n_haz if n_haz else 0.0
    spec = tn / n_ben if n_ben else 0.0
    slo, shi = wilson_ci(tp, n_haz)
    xlo, xhi = wilson_ci(tn, n_ben)
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc   = (tp*tn - fp*fn) / denom if denom else 0.0
    crit_denom = fn + tp
    crit  = fn / crit_denom if crit_denom else 0.0
    return dict(sensitivity=sens, sens_lo=slo, sens_hi=shi,
                specificity=spec, spec_lo=xlo, spec_hi=xhi,
                mcc=mcc, critical_under_triage=crit,
                tp=tp, fp=fp, tn=tn, fn=fn)


def save_fig(fig, stem):
    for d in [FIGURES_DIR, NB_FIGURES]:
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2  – Sensitivity vs alpha (physician + realworld)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_sensitivity_vs_alpha():
    abl_path = OUTPUT_DIR / "ablation_results.json"
    if not abl_path.exists():
        print("Skipping fig2: ablation_results.json not found"); return

    with open(abl_path) as f:
        raw = json.load(f)

    # Build metrics per dataset × alpha
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in raw:
        buckets[(r["dataset"], r["alpha"])].append(r)

    records = []
    for (ds, alpha), rows in buckets.items():
        m = compute_metrics(rows)
        m["dataset"] = ds; m["alpha"] = alpha
        records.append(m)
    df = pd.DataFrame(records).sort_values(["dataset","alpha"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    colors  = {"physician": "#4C72B0", "realworld": "#DD8452"}
    markers = {"physician": "o",       "realworld": "s"}
    ylims   = {"physician": (0.25, 0.70), "realworld": (0.00, 0.30)}
    titles  = {"physician": "Physician-Created Test Set (n=200, 132 hazardous)",
               "realworld": "Real-World Test Set (n=2,000, 165 hazardous)"}

    for i, ds in enumerate(["physician", "realworld"]):
        ax    = axes[i]
        ds_df = df[df["dataset"] == ds].sort_values("alpha")
        if ds_df.empty:
            ax.set_title(titles[ds]); ax.text(0.5,0.5,"No data",ha="center",va="center",
                                               transform=ax.transAxes); continue

        alphas = ds_df["alpha"].values
        sens   = ds_df["sensitivity"].values
        lo     = ds_df["sens_lo"].values
        hi     = ds_df["sens_hi"].values

        ax.fill_between(alphas, lo, hi, alpha=0.18, color=colors[ds])
        ax.plot(alphas, sens, marker=markers[ds], color=colors[ds],
                linewidth=1.8, markersize=5, label="Sensitivity")
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8,
                   label="Baseline (α=1.0, no ablation)")

        # Annotate baseline — place text in lower-right, clear of the legend
        row = ds_df[ds_df["alpha"]==1.0]
        if not row.empty:
            bl   = row["sensitivity"].values[0]
            ylo, yhi = ylims[ds]
            # lower-right quadrant: x=2.5, y = 25% up the range
            text_y = ylo + 0.22 * (yhi - ylo)
            ax.annotate(f"Baseline: {bl:.3f}", xy=(1.0, bl),
                        xytext=(2.5, text_y),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                        fontsize=8, color="gray",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="#AAAAAA", alpha=0.9))

        ax.set_xlabel("H-Neuron Scaling Factor (α)")
        ax.set_ylabel("Hazard Detection Sensitivity")
        ax.set_title(titles[ds], fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(-0.15, 3.15)
        ax.set_ylim(*ylims[ds])   # use the per-dataset ylim (tighter range)
        ax.grid(True, alpha=0.25)

    plt.suptitle("Sensitivity Under H-Neuron Modulation", fontsize=12, y=1.01)
    plt.tight_layout()
    save_fig(fig, "fig3_sensitivity_vs_alpha")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3  – Safety trade-off with magnified inset
# ─────────────────────────────────────────────────────────────────────────────
def fig3_safety_tradeoff():
    metrics_path = OUTPUT_DIR / "triage_metrics_by_alpha.csv"
    abl_path     = OUTPUT_DIR / "ablation_results.json"
    if not abl_path.exists():
        print("Skipping fig3: ablation_results.json not found"); return

    with open(abl_path) as f:
        raw = json.load(f)
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in raw:
        buckets[(r["dataset"], r["alpha"])].append(r)
    records = []
    for (ds, alpha), rows in buckets.items():
        m = compute_metrics(rows)
        m["dataset"] = ds; m["alpha"] = alpha
        records.append(m)
    df = pd.DataFrame(records).sort_values(["dataset","alpha"])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"physician": "#4C72B0", "realworld": "#DD8452"}

    # JMIR baselines
    baselines = {
        "GPT-5.1\nUnassisted":    (1-0.661, 0.661),
        "GPT-5.1\nSafety Prompt": (1-0.836, 0.836),
        "Guardrail":              (1-0.977, 0.977),
        "CQL":                    (1-0.955, 0.955),
    }
    for name, (crit, sens) in baselines.items():
        ax.scatter(crit, sens, marker="D", s=55, color="#888888", zorder=5)
        ax.annotate(name, xy=(crit, sens), fontsize=6,
                    textcoords="offset points", xytext=(0, -14),
                    color="#666666", ha="center")

    for ds in ["physician", "realworld"]:
        ds_df = df[df["dataset"]==ds].sort_values("alpha")
        if ds_df.empty: continue
        sens   = ds_df["sensitivity"].values
        crit   = ds_df["critical_under_triage"].values
        alphas = ds_df["alpha"].values
        ax.plot(crit, sens, marker="o", color=colors[ds], linewidth=1.5, markersize=6,
                label=f"Llama-8B ({'Physician' if ds=='physician' else 'Real-world'})", zorder=10)
        # Label only endpoints (α=0 and α=3) with arrows so position is unambiguous
        a_min, a_max = alphas.min(), alphas.max()
        for a, s, c in zip(alphas, sens, crit):
            if a in {a_min, a_max}:
                label = f"α={int(a)}" if a == int(a) else f"α={a:.2f}"
                ox = 30 if a == a_min else -30
                oy = 18 if a == a_min else -18
                ax.annotate(label, xy=(c, s), fontsize=7,
                            textcoords="offset points", xytext=(ox, oy),
                            color=colors[ds],
                            arrowprops=dict(arrowstyle="->", color=colors[ds],
                                            lw=0.7, connectionstyle="arc3,rad=0"))

    ax.set_xlabel("Critical Under-Triage Rate (lower is safer)")
    ax.set_ylabel("Hazard Detection Sensitivity (higher is safer)")
    ax.set_title("Safety Trade-off: H-Neuron Modulation vs Baseline Systems")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1.05)

    # ── Magnified inset: physician cluster ──────────────────────────────────
    # Tight zoom: pad only 0.015 on each side of the actual data range
    phys_df = df[df["dataset"]=="physician"].sort_values("alpha")
    if not phys_df.empty:
        px    = phys_df["critical_under_triage"].values
        py    = phys_df["sensitivity"].values
        alphas_p = phys_df["alpha"].values
        PAD   = 0.015
        x_min = px.min() - PAD;  x_max = px.max() + PAD
        y_min = py.min() - PAD;  y_max = py.max() + PAD

        # Place inset upper-centre — clear of baselines (upper-left) and
        # real-world cluster (lower-right); no axis-label collision with main axes
        axins = ax.inset_axes([0.30, 0.55, 0.38, 0.38])
        axins.plot(px, py, marker="o", color=colors["physician"],
                   linewidth=1.5, markersize=5, zorder=10)

        # Label only α=0 and α=3 in the inset — arrows for clarity
        a_min, a_max = alphas_p.min(), alphas_p.max()
        for a, s, c in zip(alphas_p, py, px):
            if a in {a_min, a_max}:
                label = f"α={int(a)}" if a == int(a) else f"α={a:.2f}"
                ox = 22 if a == a_min else -22
                oy = 12 if a == a_min else -10
                axins.annotate(label, xy=(c, s), fontsize=6,
                               textcoords="offset points", xytext=(ox, oy),
                               color=colors["physician"],
                               arrowprops=dict(arrowstyle="->", color=colors["physician"],
                                               lw=0.6))

        axins.set_xlim(x_min, x_max)
        axins.set_ylim(y_min, y_max)
        # No axis labels — only tick values to avoid collision with main axes text
        axins.set_xlabel("")
        axins.set_ylabel("")
        axins.tick_params(labelsize=5)
        axins.grid(True, alpha=0.3)
        axins.set_title("Physician cluster (zoom)", fontsize=6, pad=2)
        try:
            mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=0.6)
        except Exception:
            pass

    plt.tight_layout()
    save_fig(fig, "efig1_safety_tradeoff")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4  – ROC / specificity-sensitivity scatter with inset
# ─────────────────────────────────────────────────────────────────────────────
def fig4_roc():
    abl_path = OUTPUT_DIR / "ablation_results.json"
    if not abl_path.exists():
        print("Skipping fig4: ablation_results.json not found"); return

    with open(abl_path) as f:
        raw = json.load(f)
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in raw:
        buckets[(r["dataset"], r["alpha"])].append(r)
    records = []
    for (ds, alpha), rows in buckets.items():
        m = compute_metrics(rows)
        m["dataset"] = ds; m["alpha"] = alpha
        records.append(m)
    df = pd.DataFrame(records).sort_values(["dataset","alpha"])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = {"physician": "#4C72B0", "realworld": "#DD8452"}

    for ds in ["physician", "realworld"]:
        ds_df = df[df["dataset"]==ds].sort_values("alpha")
        if ds_df.empty: continue
        spec   = ds_df["specificity"].values
        sens   = ds_df["sensitivity"].values
        alphas = ds_df["alpha"].values
        fpr    = 1 - spec

        ax.plot(fpr, sens, marker="o", color=colors[ds], linewidth=1.5,
                markersize=5, label=f"{'Physician' if ds=='physician' else 'Real-world'}")

        # Label only α=0 and α=3 in main plot with arrows
        a_min, a_max = alphas.min(), alphas.max()
        for a, s, fp in zip(alphas, sens, fpr):
            if a in {a_min, a_max}:
                label = f"α={int(a)}" if a == int(a) else f"α={a:.2f}"
                ox = 30 if a == a_min else -30
                oy = 14 if a == a_min else -14
                ax.annotate(label, xy=(fp, s), fontsize=7,
                            textcoords="offset points", xytext=(ox, oy),
                            color=colors[ds],
                            arrowprops=dict(arrowstyle="->", color=colors[ds],
                                            lw=0.7, connectionstyle="arc3,rad=0"))

    ax.plot([0,1],[0,1],"k--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("Detection Performance by H-Neuron Scaling Factor")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

    # ── Magnified inset: physician cluster only, tight zoom ──────────────────
    phys_df = df[df["dataset"]=="physician"].sort_values("alpha")
    if not phys_df.empty:
        spec_p   = phys_df["specificity"].values
        sens_p   = phys_df["sensitivity"].values
        alphas_p = phys_df["alpha"].values
        fpr_p    = 1 - spec_p

        PAD   = 0.012
        x_min = fpr_p.min()  - PAD;  x_max = fpr_p.max()  + PAD
        y_min = sens_p.min() - PAD;  y_max = sens_p.max() + PAD

        # Place inset in lower-right — physician cluster sits at centre of main axes
        axins = ax.inset_axes([0.52, 0.08, 0.45, 0.45])
        axins.plot(fpr_p, sens_p, marker="o", color=colors["physician"],
                   linewidth=1.5, markersize=5, zorder=10)

        # Label α=0, α=1 (baseline, peak sensitivity), and α=3 with arrows
        a_min_p, a_max_p = alphas_p.min(), alphas_p.max()
        for a, s, fp in zip(alphas_p, sens_p, fpr_p):
            if a == a_min_p:
                axins.annotate("α=0\n(suppress)", xy=(fp, s), fontsize=6,
                               textcoords="offset points", xytext=(24, 12),
                               color=colors["physician"],
                               arrowprops=dict(arrowstyle="->", color=colors["physician"],
                                               lw=0.6))
            elif a == 1.0:
                axins.annotate("α=1\n(baseline)", xy=(fp, s), fontsize=6,
                               textcoords="offset points", xytext=(-28, 10),
                               color="#555555",
                               arrowprops=dict(arrowstyle="->", color="#555555",
                                               lw=0.6))
            elif a == a_max_p:
                axins.annotate("α=3\n(amplify)", xy=(fp, s), fontsize=6,
                               textcoords="offset points", xytext=(-24, -16),
                               color=colors["physician"],
                               arrowprops=dict(arrowstyle="->", color=colors["physician"],
                                               lw=0.6))

        axins.set_xlim(x_min, x_max)
        axins.set_ylim(y_min, y_max)
        axins.tick_params(labelsize=5)
        axins.grid(True, alpha=0.3)
        # Title placed above the inset box via transAxes to avoid clipping
        axins.text(0.5, 1.04, "Physician cluster (zoom)",
                   transform=axins.transAxes, ha="center", va="bottom",
                   fontsize=6, color="#444444")
        try:
            mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5", lw=0.5)
        except Exception:
            pass

    plt.tight_layout()
    save_fig(fig, "efig2_roc_by_alpha")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5  – Logit lens: add delta-from-baseline row so differences are visible
# ─────────────────────────────────────────────────────────────────────────────
def fig5_logit_lens():
    path = OUTPUT_DIR / "logit_lens_results.json"
    if not path.exists():
        print("Skipping fig5: logit_lens_results.json not found"); return
    with open(path) as f:
        results = json.load(f)

    groups = ["emergency", "urgent", "routine", "reassure", "hedging"]
    colors = {"emergency": "#d62728", "urgent": "#ff7f0e", "routine": "#2ca02c",
              "reassure":  "#1f77b4", "hedging": "#9467bd"}

    alpha_keys   = ["0.5", "1.0", "2.0"]
    alpha_labels = ["Suppressed (α=0.5)", "Baseline (α=1.0)", "Amplified (α=2.0)"]

    # ── compute per-alpha per-layer group means for hazardous cases ──
    def get_means(ak):
        if ak not in results:
            return None, None
        ar = results[ak]
        haz = [r for r in ar if r["detection_truth"] == 1] or ar
        n_layers = max(int(k) for r in haz for k in r["per_layer"].keys()) + 1
        gm = {g: np.zeros(n_layers) for g in groups}
        cnt = np.zeros(n_layers)
        for r in haz:
            for ls, ld in r["per_layer"].items():
                li = int(ls)
                gp = ld["group_probs"]
                for g in groups:
                    gm[g][li] += gp.get(g, 0.0)
                cnt[li] += 1
        for g in groups:
            gm[g] /= np.maximum(cnt, 1)
        return gm, n_layers

    means = {ak: get_means(ak) for ak in alpha_keys}
    base_means, n_layers = means["1.0"]
    if base_means is None:
        print("Skipping fig5: no baseline data"); return

    # ── 2-row layout: top = absolute (layers 0-29), bottom = Δ from baseline ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [1, 0.7]},
                             sharey="row")
    fig.subplots_adjust(hspace=0.45, wspace=0.08)

    layers = np.arange(n_layers)

    for ai, (ak, alabel) in enumerate(zip(alpha_keys, alpha_labels)):
        gm, _ = means[ak]
        if gm is None:
            continue

        # ── row 0: absolute, zoom to layers 0–28 to exclude output spike ──
        ax0 = axes[0, ai]
        for g in groups:
            ax0.plot(layers[:-1], gm[g][:-1], label=g.capitalize(),
                     color=colors[g], linewidth=1.8)
        ax0.set_title(alabel, fontsize=10)
        ax0.set_xlim(0, n_layers - 2)
        ax0.set_xticks(range(0, n_layers - 1, 4))
        ax0.grid(True, alpha=0.2)
        if ai == 0:
            ax0.set_ylabel("Token Group Probability", fontsize=9)
        if ai == 2:
            ax0.legend(loc="lower right", fontsize=7, framealpha=0.9)

        # ── row 1: Δ from baseline ──
        ax1 = axes[1, ai]
        for g in groups:
            delta = gm[g][:-1] - base_means[g][:-1]
            ax1.plot(layers[:-1], delta, label=g.capitalize(),
                     color=colors[g], linewidth=1.6)
        ax1.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax1.set_xlim(0, n_layers - 2)
        ax1.set_xticks(range(0, n_layers - 1, 4))
        ax1.grid(True, alpha=0.2)
        ax1.set_xlabel("Transformer Layer", fontsize=9)
        if ai == 0:
            ax1.set_ylabel("Δ from Baseline", fontsize=9)
        if ak == "1.0":
            ax1.text(0.5, 0.88, "Baseline\n(Δ=0)", ha="center", va="top",
                     transform=ax1.transAxes, fontsize=9, color="gray", style="italic",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    fig.suptitle(
        "Logit Lens: Safety Token Probabilities Across Layers (Hazardous Cases)\n"
        "Top: absolute probability  ·  Bottom: change from α=1.0 baseline",
        fontsize=10, y=1.01)
    save_fig(fig, "efig3_logit_lens_group_probs")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6  – Reassurance heatmap with auto-scaled colormap
# ─────────────────────────────────────────────────────────────────────────────
def fig6_heatmap():
    path = OUTPUT_DIR / "logit_lens_divergence.json"
    if not path.exists():
        print("Skipping fig6: logit_lens_divergence.json not found"); return
    with open(path) as f:
        divergence = json.load(f)

    hazard_data = {k: v for k, v in divergence.items() if v["detection_truth"] == 1}
    benign_data = {k: v for k, v in divergence.items() if v["detection_truth"] == 0}
    if not hazard_data:
        print("Skipping fig6: no hazard cases"); return

    n_h = len(hazard_data); n_b = max(len(benign_data), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [n_h, n_b]})

    for ax, data, title in [(axes[0], hazard_data, "Hazardous Cases"),
                            (axes[1], benign_data, "Benign Cases")]:
        if not data:
            ax.text(0.5, 0.5, "No cases", ha="center", va="center",
                    transform=ax.transAxes); ax.set_title(title); continue

        matrix = np.array([v["reassure_shift_per_layer"] for v in data.values()])
        # Auto-scale: use 95th percentile of absolute values so colour is not washed out
        vmax = np.percentile(np.abs(matrix), 95)
        vmax = max(vmax, 1e-5)   # guard against all-zero
        vmin = -vmax

        im = ax.imshow(matrix.T, aspect="auto", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("Case Index")
        ax.set_ylabel("Transformer Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax,
                     label=f"Reassurance Prob. Shift\n(α=2.0 minus α=1.0)\nvmax={vmax:.4f}",
                     shrink=0.8)

    fig.suptitle("Over-Compliance Signal: Reassurance Shift Under H-Neuron Amplification",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save_fig(fig, "efig4_reassurance_shift_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9  – 2×2 factorial with legend/annotations outside data region
# ─────────────────────────────────────────────────────────────────────────────
def fig9_two_by_two():
    summary_path = OUTPUT_DIR / "two_by_two_summary.json"
    if not summary_path.exists():
        print("Skipping fig9: two_by_two_summary.json not found"); return
    with open(summary_path) as f:
        summary = json.load(f)

    df = pd.DataFrame(summary)

    arm_cfg = {
        "A_base_triviaqa":     dict(label="Arm A  base + TriviaQA",   color="#1f77b4",
                                    marker="o", ls="-"),
        "B_base_medical":      dict(label="Arm B  base + medical",    color="#2ca02c",
                                    marker="s", ls="-"),
        "C_finetuned_triviaqa":dict(label="Arm C  fine-tuned + TriviaQA", color="#ff7f0e",
                                    marker="^", ls="--"),
        "D_finetuned_medical": dict(label="Arm D  fine-tuned + medical",  color="#d62728",
                                    marker="D", ls="--"),
    }
    alpha_order = [0.0, 1.0, 3.0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ds_cfg = {
        "physician": dict(title="a  Physician test set  (n=200, 132 hazardous)",
                          ylim=(0.0, 0.82)),
        "realworld": dict(title="b  Real-world test set  (n=500, 165 hazardous)",
                          ylim=(0.0, 0.38)),
    }

    for ax, (ds, dcfg) in zip(axes, ds_cfg.items()):
        ds_df = df[df["dataset"] == ds]

        for arm, acfg in arm_cfg.items():
            arm_df = ds_df[ds_df["arm"] == arm].sort_values("alpha")
            arm_df = arm_df[arm_df["alpha"].isin(alpha_order)]
            if arm_df.empty:
                continue

            alphas   = arm_df["alpha"].values
            sens     = arm_df["sensitivity"].values
            lo       = np.array([r[0] for r in arm_df["sensitivity_ci"]])
            hi       = np.array([r[1] for r in arm_df["sensitivity_ci"]])
            yerr     = np.array([sens - lo, hi - sens])
            pending  = [a for a in alpha_order if a not in alphas]

            ax.errorbar(alphas, sens, yerr=yerr,
                        color=acfg["color"], marker=acfg["marker"],
                        linestyle=acfg["ls"], linewidth=1.6, markersize=6,
                        capsize=3, label=acfg["label"])

            for a in pending:
                ax.scatter([a], [0.05], marker=acfg["marker"], s=60,
                           facecolors="none", edgecolors=acfg["color"],
                           linewidth=1.2, zorder=5)
                ax.text(a, 0.04, "pend.", fontsize=6, ha="center",
                        color=acfg["color"])

        # Spearman rho annotation for arms A+B (base model), placed ABOVE the lines
        if ds == "physician":
            ax.text(0.98, 0.97,
                    "Monotone  ρ=−1.000\n(Arms A & B, base model)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="#2ca02c",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#2ca02c", alpha=0.85))
            # Arm D rescue annotation – place BELOW the fine-tuned lines, away from base
            ax.text(0.03, 0.48,
                    "Null rescue 3.2%\np=0.752",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5, color="#d62728",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#d62728", alpha=0.85))
        elif ds == "realworld":
            ax.text(0.98, 0.97,
                    "Arm D (fine-tuned + medical)\nall alphas pending",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="gray",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="gray", alpha=0.85))

        ax.set_xticks(alpha_order)
        ax.set_xticklabels(["0.0\n(suppress)", "1.0\n(baseline)", "3.0\n(amplify)"])
        ax.set_xlabel("Alpha  (0 = suppress · 1 = baseline · 3 = amplify)", fontsize=9)
        ax.set_ylabel("Sensitivity  (95% Wilson CI)", fontsize=9)
        ax.set_title(dcfg["title"], fontsize=10, fontweight="bold", loc="left")
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(*dcfg["ylim"])
        ax.grid(True, alpha=0.25)

        # No per-axis legend — collected once as fig.legend below

    # Single legend centred below both panels, with enough bottom margin
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, fontsize=8, framealpha=0.9)

    fig.suptitle("2×2 factorial: sensitivity under h-neuron modulation",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
             "Alpha = scaling factor applied to selected neuron activations.  "
             "Arms A–B: base Meta-Llama-3.1-8B-Instruct.  "
             "Arms C–D: QLoRA fine-tuned model.  Error bars = 95% Wilson score CI.",
             ha="center", fontsize=7.5, color="gray", wrap=True)
    plt.tight_layout(rect=[0, 0.11, 1, 1])
    save_fig(fig, "fig5_two_by_two_factorial")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import os
    os.chdir("/Users/sanjaybasu/waymark-local/packaging/h_neuron_triage")
    fig2_sensitivity_vs_alpha()
    fig3_safety_tradeoff()
    fig4_roc()
    fig5_logit_lens()
    fig6_heatmap()
    fig9_two_by_two()
    print("\nAll figures regenerated.")


if __name__ == "__main__":
    main()
