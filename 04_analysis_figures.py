#!/usr/bin/env python3
"""Step 4: Generate analysis figures and comparison tables.

Figures:
  1. H-neuron layer distribution histogram
  2. Sensitivity vs alpha curve (physician + realworld)
  3. Sensitivity vs critical under-triage trade-off
  4. Comparison with JMIR baselines (GPT-5.1, guardrails, RL)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output")
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def fig1_layer_distribution():
    """H-neuron layer distribution histogram."""
    h_path = OUTPUT_DIR / "h_neurons.json"
    if not h_path.exists():
        print("Skipping fig1: h_neurons.json not found")
        return
    with open(h_path) as f:
        data = json.load(f)

    layers = [hn["layer"] for hn in data["h_neurons"]]
    n_layers = data["n_layers"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts, bins, patches = ax.hist(layers, bins=np.arange(-0.5, n_layers + 0.5, 1),
                                     edgecolor="black", linewidth=0.5, color="#4C72B0")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Number of H-Neurons")
    ax.set_title(f"H-Neuron Distribution Across Layers (N={len(layers)})")
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xticks(range(0, n_layers, 4))
    ax.text(0.98, 0.95,
            f"{len(layers)} neurons\n{data['pct_of_total']:.3f}% of total\nCV AUC={data['cv_auc_mean']:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    path = FIGURES_DIR / "fig1_h_neuron_layer_distribution.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def fig2_sensitivity_vs_alpha():
    """Sensitivity vs alpha for physician and realworld datasets."""
    metrics_path = OUTPUT_DIR / "triage_metrics_by_alpha.csv"
    if not metrics_path.exists():
        print("Skipping fig2: triage_metrics_by_alpha.csv not found")
        return
    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = {"physician": "#4C72B0", "realworld": "#DD8452"}
    markers = {"physician": "o", "realworld": "s"}

    for i, ds in enumerate(["physician", "realworld"]):
        ax = axes[i]
        ds_df = df[df["dataset"] == ds].sort_values("alpha")
        if ds_df.empty:
            continue

        alphas = ds_df["alpha"].values
        sens = ds_df["sensitivity"].values
        lo = ds_df["sensitivity_ci_lo"].values
        hi = ds_df["sensitivity_ci_hi"].values

        ax.fill_between(alphas, lo, hi, alpha=0.2, color=colors[ds])
        ax.plot(alphas, sens, marker=markers[ds], color=colors[ds],
                linewidth=1.5, markersize=5, label=f"Sensitivity")
        ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Baseline (no ablation)")
        ax.set_xlabel("H-Neuron Scaling Factor (alpha)")
        if i == 0:
            ax.set_ylabel("Hazard Detection Sensitivity")
        ax.set_title(f"{'Physician-Created' if ds == 'physician' else 'Real-World'} Test Set")
        ax.legend(loc="lower right")
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Annotate baseline and best suppression
        baseline_row = ds_df[ds_df["alpha"] == 1.0]
        if not baseline_row.empty:
            bl_sens = baseline_row["sensitivity"].values[0]
            ax.annotate(f"Baseline: {bl_sens:.3f}",
                        xy=(1.0, bl_sens), xytext=(1.5, bl_sens - 0.1),
                        arrowprops=dict(arrowstyle="->", color="gray"),
                        fontsize=8, color="gray")

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_sensitivity_vs_alpha.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def fig3_safety_tradeoff():
    """Sensitivity vs critical under-triage at each alpha, with JMIR baselines."""
    metrics_path = OUTPUT_DIR / "triage_metrics_by_alpha.csv"
    if not metrics_path.exists():
        print("Skipping fig3: triage_metrics_by_alpha.csv not found")
        return
    df = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"physician": "#4C72B0", "realworld": "#DD8452"}

    # JMIR baselines
    baselines = {
        "GPT-5.1\nUnassisted": (1 - 0.661, 0.661),
        "GPT-5.1\nSafety Prompt": (1 - 0.836, 0.836),
        "Guardrail": (1 - 0.977, 0.977),
        "CQL": (1 - 0.955, 0.955),
        "AWR": (1 - 0.955, 0.955),
    }
    for name, (crit, sens) in baselines.items():
        ax.scatter(crit, sens, marker="D", s=60, color="#888888", zorder=5)
        ax.annotate(name, xy=(crit, sens), fontsize=6, textcoords="offset points",
                    xytext=(-5, -12), color="#666666", ha="center")

    for ds in ["physician", "realworld"]:
        ds_df = df[df["dataset"] == ds].sort_values("alpha")
        if ds_df.empty:
            continue

        sens = ds_df["sensitivity"].values
        crit = ds_df["critical_under_triage"].values
        alphas = ds_df["alpha"].values

        ax.plot(crit, sens, marker="o", color=colors[ds], linewidth=1.5, markersize=6,
                label=f"Llama-3.1-8B ({'Physician' if ds == 'physician' else 'Real-world'})",
                zorder=10)

        # Label alpha values
        for a, s, c in zip(alphas, sens, crit):
            label = f"α={a:.1f}" if a != int(a) else f"α={int(a)}"
            ax.annotate(label, xy=(c, s), fontsize=6, textcoords="offset points",
                        xytext=(5, 4), color=colors[ds])

    # Add arrow showing direction of amplification
    ax.annotate("", xy=(0.40, 0.35), xytext=(0.25, 0.42),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.text(0.42, 0.33, "H-neuron\namplification", fontsize=7, color="red", style="italic")

    ax.set_xlabel("Critical Under-Triage Rate (lower is safer)")
    ax.set_ylabel("Hazard Detection Sensitivity (higher is safer)")
    ax.set_title("Safety Trade-off: H-Neuron Modulation vs Baseline Systems")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = FIGURES_DIR / "fig3_safety_tradeoff.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def fig4_comparison_table():
    """Comparison table with JMIR baselines."""
    metrics_path = OUTPUT_DIR / "triage_metrics_by_alpha.csv"
    if not metrics_path.exists():
        print("Skipping fig4: triage_metrics_by_alpha.csv not found")
        return
    df = pd.read_csv(metrics_path)

    # JMIR baselines (from the published study)
    baselines = [
        {"System": "GPT-5.1 (Unassisted)", "Dataset": "physician",
         "Sensitivity": 0.661, "Specificity": 0.985, "MCC": 0.697,
         "Critical_Under_Triage": 0.371},
        {"System": "GPT-5.1 (Safety Prompt)", "Dataset": "physician",
         "Sensitivity": 0.836, "Specificity": 0.956, "MCC": 0.802,
         "Critical_Under_Triage": 0.210},
        {"System": "Guardrail Classifier", "Dataset": "physician",
         "Sensitivity": 0.977, "Specificity": 0.500, "MCC": 0.562,
         "Critical_Under_Triage": 1.000},
        {"System": "CQL Controller", "Dataset": "physician",
         "Sensitivity": 0.955, "Specificity": 0.691, "MCC": 0.692,
         "Critical_Under_Triage": 0.274},
        {"System": "AWR Controller", "Dataset": "physician",
         "Sensitivity": 0.955, "Specificity": 0.706, "MCC": 0.700,
         "Critical_Under_Triage": 0.258},
    ]

    # Add h-neuron ablation results at key alpha levels
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        for ds in ["physician"]:
            row = df[(df["dataset"] == ds) & (df["alpha"] == alpha)]
            if row.empty:
                continue
            row = row.iloc[0]
            baselines.append({
                "System": f"Llama-8B H-Ablation (a={alpha})",
                "Dataset": ds,
                "Sensitivity": row["sensitivity"],
                "Specificity": row["specificity"],
                "MCC": row["mcc"],
                "Critical_Under_Triage": row["critical_under_triage"],
            })

    comp_df = pd.DataFrame(baselines)
    path = TABLES_DIR / "comparison_with_baselines.csv"
    comp_df.to_csv(path, index=False)
    print(f"Saved {path}")
    print(comp_df.to_string(index=False))


def fig5_specificity_sensitivity_scatter():
    """ROC-like scatter of specificity vs sensitivity at each alpha."""
    metrics_path = OUTPUT_DIR / "triage_metrics_by_alpha.csv"
    if not metrics_path.exists():
        print("Skipping fig5: triage_metrics_by_alpha.csv not found")
        return
    df = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"physician": "#4C72B0", "realworld": "#DD8452"}

    for ds in ["physician", "realworld"]:
        ds_df = df[df["dataset"] == ds].sort_values("alpha")
        if ds_df.empty:
            continue
        spec = ds_df["specificity"].values
        sens = ds_df["sensitivity"].values
        alphas = ds_df["alpha"].values

        ax.plot(1 - spec, sens, marker="o", color=colors[ds], linewidth=1.5,
                markersize=5, label=f"{'Physician' if ds == 'physician' else 'Real-world'}")
        for a, s, fpr in zip(alphas, sens, 1 - spec):
            label = f"{a:.1f}" if a != int(a) else f"{int(a)}"
            ax.annotate(label, xy=(fpr, s), fontsize=6,
                        textcoords="offset points", xytext=(4, 4), color=colors[ds])

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Detection Performance by H-Neuron Scaling Factor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    path = FIGURES_DIR / "fig5_roc_by_alpha.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    fig1_layer_distribution()
    fig2_sensitivity_vs_alpha()
    fig3_safety_tradeoff()
    fig4_comparison_table()
    fig5_specificity_sensitivity_scatter()
    print("\nAll figures and tables generated.")


if __name__ == "__main__":
    main()
