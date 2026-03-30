#!/usr/bin/env python3
"""Fetch 70B h-neuron results from Modal volume and print key metrics.

Usage (run after run_70b_phase1 completes):
    python fetch_70b_results.py

This pulls 70b_h_neurons.json from the h-neuron-results-v3 volume,
prints key metrics, and saves a local copy.
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    target = output_dir / "70b_h_neurons.json"

    print("Pulling 70b_h_neurons.json from h-neuron-results-v3 volume...")
    result = subprocess.run(
        ["modal", "volume", "get", "h-neuron-results-v3", "output/70b_h_neurons.json",
         str(target)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        print("Phase 1 may not be complete yet. Check:")
        print("  modal app logs ap-0RtNk2EpO4IecDq8Ikbx6x")
        sys.exit(1)

    with open(target) as f:
        data = json.load(f)

    print("\n" + "=" * 60)
    print("70B H-NEURON RESULTS")
    print("=" * 60)
    print(f"Model:          {data['model']}")
    print(f"Layers:         {data['n_layers']}")
    print(f"Intermediate:   {data['intermediate_size']}")
    print(f"Samples used:   {data['n_samples']}")
    print(f"CV AUC mean:    {data['cv_auc_mean']:.4f}")
    print(f"CV AUC std:     {data['cv_auc_std']:.4f}")
    print(f"H-neurons:      {data['n_h_neurons']}")
    pct = data['pct_of_total']
    total = data['n_layers'] * data['intermediate_size']
    print(f"% of FFN:       {pct:.4f}%  ({data['n_h_neurons']}/{total:,})")

    # Layer distribution
    from collections import Counter
    layer_counts = Counter(hn["layer"] for hn in data["h_neurons"])
    sorted_layers = sorted(layer_counts.items())
    print(f"\nLayer distribution (top 10 layers by count):")
    for layer, count in sorted(sorted_layers, key=lambda x: -x[1])[:10]:
        print(f"  Layer {layer:3d}: {count} neurons")

    print("\n" + "=" * 60)
    print("INSERT INTO MANUSCRIPT:")
    print("=" * 60)
    print(f"  Abstract / Results line: probe AUC {data['cv_auc_mean']:.3f} "
          f"(SD {data['cv_auc_std']:.3f})")
    print(f"  H-neuron count: {data['n_h_neurons']} ({pct:.3f}% of FFN neurons)")
    print(f"  Replace [VERIFY: output file not found...] with these values.")
    print("=" * 60)


if __name__ == "__main__":
    main()
