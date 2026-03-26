#!/usr/bin/env python3
"""Generate Extended Data Figure 5: permutation control ρ-distribution histogram.

Reads 46 random neuron set results from Modal volume and plots the distribution
of within-set Spearman ρ (sensitivity vs. α across 4 levels: 0.0, 1.0, 2.0, 3.0),
with vertical lines marking the h-neuron thresholds.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import spearmanr

# ── output paths ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/h_neuron_triage/figures")
PKG_DIR    = Path("/Users/sanjaybasu/waymark-local/packaging/h_neuron_triage/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PKG_DIR.mkdir(parents=True, exist_ok=True)

# ── precomputed ρ values (from 46 random sets in Modal volume) ───────────────
# Computed by make_fig_permutation_histogram.py using Modal volume h-neuron-results-v3
RHO_VALUES = [
    1.0, 0.4, -0.7746, 1.0, -0.7746, 0.0, 0.8, -0.3162, -0.7746, -0.3162,
    0.8, -0.6, -0.4, -0.7746, -1.0, -1.0, 0.8, -0.9487, -0.2, -0.7746,
    0.0, 1.0, -0.3162, 0.2, -0.4, -0.9487, -0.4472, 0.6, -1.0, -0.9487,
    0.6325, -0.4, 0.8, -0.9487, -0.2582, -0.7746, -0.6, -0.6, 0.0, 0.6,
    -0.2, -0.4472, -0.6325, -0.2108, -1.0, -0.4
]

# Fetch from Modal if available; otherwise use precomputed values
def fetch_rho_from_modal():
    try:
        import modal
        vol = modal.Volume.from_name('h-neuron-results-v3')
        files = list(vol.listdir('output/', recursive=False))
        set_files = sorted(
            [f.path for f in files if f.path.startswith('output/random_set_') and f.path.endswith('.json')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        rho_values = []
        for fp in set_files:
            data = b''.join(vol.read_file(fp))
            records = json.loads(data)
            by_alpha = defaultdict(list)
            for r in records:
                by_alpha[r['alpha']].append(r)
            alphas_present = sorted(by_alpha.keys())
            sens_list = []
            for a in alphas_present:
                recs = by_alpha[a]
                hazard = [r for r in recs if r['detection_truth'] == 1]
                tp = sum(1 for r in hazard if r['detection_pred'] == 1)
                fn = sum(1 for r in hazard if r['detection_pred'] == 0)
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                sens_list.append((a, sens))
            if len(sens_list) >= 2:
                alphas_arr = [s[0] for s in sens_list]
                sens_arr = [s[1] for s in sens_list]
                if len(set(sens_arr)) > 1:
                    rho, _ = spearmanr(alphas_arr, sens_arr)
                else:
                    rho = 0.0
                rho_values.append(rho)
        print(f"Fetched {len(rho_values)} ρ values from Modal volume")
        return rho_values
    except Exception as e:
        print(f"Modal fetch failed ({e}), using precomputed values")
        return RHO_VALUES


def make_figure(rho_values):
    n_sets = len(rho_values)
    rho_arr = np.array(rho_values)

    # Key statistics
    mean_rho = float(np.mean(rho_arr))
    sd_rho = float(np.std(rho_arr, ddof=1))
    threshold_full = -0.800          # h-neuron full-range ρ (4 alpha levels)
    threshold_amp  = -0.9487         # h-neuron amplification-range ρ (α=1.0→3.0)
    n_below_full = int((rho_arr <= threshold_full).sum())
    p_one_sided = n_below_full / n_sets

    print(f"n_sets={n_sets}, mean ρ={mean_rho:.3f}, SD={sd_rho:.3f}")
    print(f"≤ {threshold_full}: {n_below_full}/{n_sets}  →  p={p_one_sided:.3f}")

    # ── figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.patch.set_facecolor("white")

    # Histogram bins from -1.0 to +1.0
    bins = np.linspace(-1.05, 1.05, 22)
    colors = []
    counts, bin_edges = np.histogram(rho_arr, bins=bins)

    # Two-tone bars: ≤ threshold_full in orange, others in steel-blue
    for count, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
        mid = (left + right) / 2
        color = "#C4681A" if mid <= threshold_full else "#3274A1"
        ax.bar(mid, count, width=(right - left) * 0.9,
               color=color, edgecolor="white", linewidth=0.6, zorder=2)

    # Reference lines
    ax.axvline(threshold_full, color="#C4681A", linestyle="--", linewidth=1.6,
               zorder=3, label=f"h-neuron full-range ρ = {threshold_full:.3f}")
    ax.axvline(threshold_amp,  color="#555555", linestyle=":",  linewidth=1.4,
               zorder=3, label=f"h-neuron amplification ρ = {threshold_amp:.3f}")
    ax.axvline(mean_rho, color="#3274A1", linestyle="-", linewidth=1.0, alpha=0.7,
               zorder=3, label=f"Mean random ρ = {mean_rho:.3f} (SD {sd_rho:.3f})")

    # Annotation: p-value
    ax.text(threshold_full - 0.04, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 8,
            f"{n_below_full}/{n_sets} sets\none-sided p = {p_one_sided:.3f}",
            ha="right", va="top", fontsize=7.5, color="#C4681A",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C4681A", alpha=0.9))

    ax.set_xlabel("Spearman ρ  (sensitivity vs. α, full range α ∈ {0.0, 1.0, 2.0, 3.0})",
                  fontsize=9)
    ax.set_ylabel("Number of random neuron sets", fontsize=9)
    ax.set_title(
        f"Permutation control: null distribution of Spearman ρ  (n = {n_sets} sets)",
        fontsize=9.5, fontweight="bold"
    )
    ax.set_xlim(-1.15, 1.15)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    legend = ax.legend(fontsize=7.5, framealpha=0.95, loc="upper left",
                       borderpad=0.5, handlelength=1.5)

    # Footer note
    fig.text(0.5, -0.04,
             f"Each set: 213 layer-matched random neurons  ·  {n_sets} of 100 planned sets completed",
             ha="center", fontsize=7.5, color="#555555", style="italic")

    fig.tight_layout()

    for out_dir in [OUTPUT_DIR, PKG_DIR]:
        path = out_dir / "fig8_permutation_rho_distribution.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")

    plt.close(fig)
    return n_sets, mean_rho, sd_rho, n_below_full, p_one_sided


if __name__ == "__main__":
    rho_values = fetch_rho_from_modal()
    make_figure(rho_values)
