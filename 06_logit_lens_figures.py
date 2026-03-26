#!/usr/bin/env python3
"""Step 6: Generate Logit Lens figures for the manuscript.

Figures:
  - Fig 5: Per-layer safety token group probabilities at baseline vs amplified
  - Fig 6: Reassurance shift heatmap (layers x cases) under h-neuron amplification
  - Fig 7: Decision locking layer distribution (hazardous vs benign cases)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("output")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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


def fig5_group_probs_by_layer():
    """Per-layer safety token group probabilities at different alpha levels."""
    path = OUTPUT_DIR / "logit_lens_results.json"
    if not path.exists():
        print("Skipping fig5_logit: logit_lens_results.json not found")
        return
    with open(path) as f:
        results = json.load(f)

    groups = ["emergency", "urgent", "routine", "reassure", "hedging"]
    colors = {"emergency": "#d62728", "urgent": "#ff7f0e", "routine": "#2ca02c",
              "reassure": "#1f77b4", "hedging": "#9467bd"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    alpha_keys = ["0.5", "1.0", "2.0"]
    alpha_labels = ["Suppressed (α=0.5)", "Baseline (α=1.0)", "Amplified (α=2.0)"]

    for ai, (alpha_key, alpha_label) in enumerate(zip(alpha_keys, alpha_labels)):
        ax = axes[ai]
        if alpha_key not in results:
            continue

        alpha_results = results[alpha_key]
        # Filter to hazardous cases only for clearer signal
        hazard_results = [r for r in alpha_results if r["detection_truth"] == 1]
        if not hazard_results:
            hazard_results = alpha_results

        # Aggregate per-layer group probs
        n_layers = max(int(k) for r in hazard_results for k in r["per_layer"].keys()) + 1
        group_means = {g: np.zeros(n_layers) for g in groups}
        counts = np.zeros(n_layers)

        for r in hazard_results:
            for layer_str, layer_data in r["per_layer"].items():
                li = int(layer_str)
                gp = layer_data["group_probs"]
                for g in groups:
                    group_means[g][li] += gp.get(g, 0.0)
                counts[li] += 1

        for g in groups:
            group_means[g] /= np.maximum(counts, 1)

        layers = np.arange(n_layers)
        for g in groups:
            ax.plot(layers, group_means[g], label=g.capitalize(),
                    color=colors[g], linewidth=1.2)

        ax.set_xlabel("Transformer Layer")
        if ai == 0:
            ax.set_ylabel("Token Group Probability")
        ax.set_title(alpha_label)
        ax.set_xlim(0, n_layers - 1)
        ax.set_xticks(range(0, n_layers, 4))
        ax.grid(True, alpha=0.2)
        if ai == 2:
            ax.legend(loc="upper right", fontsize=7)

    plt.suptitle("Logit Lens: Safety Token Probabilities Across Layers (Hazardous Cases)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    p = FIGURES_DIR / "fig5_logit_lens_group_probs.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def fig6_reassurance_shift_heatmap():
    """Heatmap of reassurance probability shift under amplification, by layer and case."""
    path = OUTPUT_DIR / "logit_lens_divergence.json"
    if not path.exists():
        print("Skipping fig6: logit_lens_divergence.json not found")
        return
    with open(path) as f:
        divergence = json.load(f)

    # Separate hazardous and benign cases
    hazard_data = {k: v for k, v in divergence.items() if v["detection_truth"] == 1}
    benign_data = {k: v for k, v in divergence.items() if v["detection_truth"] == 0}

    if not hazard_data:
        print("Skipping fig6: no hazard cases in divergence data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [len(hazard_data), max(len(benign_data), 1)]})

    for ax, data, title in [(axes[0], hazard_data, "Hazardous Cases"),
                            (axes[1], benign_data, "Benign Cases")]:
        if not data:
            ax.text(0.5, 0.5, "No cases", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        matrix = np.array([v["reassure_shift_per_layer"] for v in data.values()])
        # matrix shape: (n_cases, n_layers)
        im = ax.imshow(matrix.T, aspect="auto", cmap="RdBu_r",
                       vmin=-0.01, vmax=0.01, interpolation="nearest")
        ax.set_xlabel("Case Index")
        ax.set_ylabel("Transformer Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Reassurance Prob. Shift\n(α=2.0 minus α=1.0)",
                     shrink=0.8)

    plt.suptitle("Over-Compliance Signal: Reassurance Shift Under H-Neuron Amplification",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    p = FIGURES_DIR / "fig6_reassurance_shift_heatmap.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def fig7_divergence_layer_distribution():
    """Distribution of max-divergence layers for hazardous vs benign cases."""
    path = OUTPUT_DIR / "logit_lens_divergence.json"
    if not path.exists():
        print("Skipping fig7: logit_lens_divergence.json not found")
        return
    with open(path) as f:
        divergence = json.load(f)

    hazard_layers = [v["max_divergence_layer"] for v in divergence.values() if v["detection_truth"] == 1]
    benign_layers = [v["max_divergence_layer"] for v in divergence.values() if v["detection_truth"] == 0]

    if not hazard_layers:
        print("Skipping fig7: no hazard cases")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    n_layers = 32
    bins = np.arange(-0.5, n_layers + 0.5, 1)

    ax.hist(hazard_layers, bins=bins, alpha=0.6, color="#d62728", label="Hazardous", density=True)
    if benign_layers:
        ax.hist(benign_layers, bins=bins, alpha=0.6, color="#1f77b4", label="Benign", density=True)

    # Add h-neuron layer distribution for comparison
    h_path = OUTPUT_DIR / "h_neurons.json"
    if h_path.exists():
        with open(h_path) as f:
            h_data = json.load(f)
        h_layers = [hn["layer"] for hn in h_data["h_neurons"]]
        h_counts, _ = np.histogram(h_layers, bins=bins)
        h_density = h_counts / h_counts.sum()
        ax.plot(range(n_layers), h_density, "k--", linewidth=1.0, alpha=0.5, label="H-neuron density")

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Density")
    ax.set_title("Layer of Maximum Decision Divergence Under H-Neuron Amplification")
    ax.legend()
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_xticks(range(0, n_layers, 4))
    ax.grid(True, alpha=0.2)

    if hazard_layers:
        ax.text(0.98, 0.95,
                f"Hazard mean: {np.mean(hazard_layers):.1f}\n"
                f"Benign mean: {np.mean(benign_layers):.1f}" if benign_layers else
                f"Hazard mean: {np.mean(hazard_layers):.1f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    p = FIGURES_DIR / "fig7_divergence_layer_distribution.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def fig8_over_compliance_dose_response():
    """Dose-response: aggregated reassurance vs urgency probability at each alpha."""
    path = OUTPUT_DIR / "logit_lens_results.json"
    if not path.exists():
        print("Skipping fig8: logit_lens_results.json not found")
        return
    with open(path) as f:
        results = json.load(f)

    alpha_keys = sorted(results.keys(), key=float)
    alphas = [float(k) for k in alpha_keys]

    # For each alpha, compute mean reassure+routine vs emergency+urgent
    # at a specific layer range (10-14, the h-neuron hotspot)
    reassure_means = []
    urgent_means = []

    for ak in alpha_keys:
        hazard_results = [r for r in results[ak] if r["detection_truth"] == 1]
        if not hazard_results:
            hazard_results = results[ak]

        reassure_vals = []
        urgent_vals = []
        for r in hazard_results:
            for layer_str in r["per_layer"]:
                li = int(layer_str)
                if 10 <= li <= 14:  # h-neuron hotspot layers
                    gp = r["per_layer"][layer_str]["group_probs"]
                    reassure_vals.append(gp.get("reassure", 0) + gp.get("routine", 0))
                    urgent_vals.append(gp.get("emergency", 0) + gp.get("urgent", 0))

        reassure_means.append(np.mean(reassure_vals) if reassure_vals else 0)
        urgent_means.append(np.mean(urgent_vals) if urgent_vals else 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, reassure_means, "o-", color="#1f77b4", label="Reassurance tokens", linewidth=1.5)
    ax.plot(alphas, urgent_means, "s-", color="#d62728", label="Urgency tokens", linewidth=1.5)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("H-Neuron Scaling Factor (α)")
    ax.set_ylabel("Mean Token Group Probability\n(Layers 10-14, Hazardous Cases)")
    ax.set_title("Over-Compliance Dose-Response")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    p = FIGURES_DIR / "fig8_over_compliance_dose_response.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def main():
    fig5_group_probs_by_layer()
    fig6_reassurance_shift_heatmap()
    fig7_divergence_layer_distribution()
    fig8_over_compliance_dose_response()
    print("\nAll logit lens figures generated.")


if __name__ == "__main__":
    main()
