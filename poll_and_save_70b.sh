#!/usr/bin/env bash
# Poll Modal volume every 3 minutes until 70b_h_neurons.json appears.
# When found: download it locally and print a summary.

set -e
OUTPUT_DIR="/Users/sanjaybasu/waymark-local/packaging/h_neuron_triage/output"
TARGET="$OUTPUT_DIR/70b_h_neurons.json"

mkdir -p "$OUTPUT_DIR"

echo "Polling for 70b_h_neurons.json on h-neuron-results-v3 ..."

while true; do
    if modal volume get h-neuron-results-v3 output/70b_h_neurons.json "$TARGET" 2>/dev/null; then
        echo "SUCCESS: 70b_h_neurons.json downloaded to $TARGET"
        python3 /Users/sanjaybasu/waymark-local/packaging/h_neuron_triage/fetch_70b_results.py
        exit 0
    else
        echo "$(date '+%H:%M:%S')  Not ready yet, retrying in 3 minutes..."
        sleep 180
    fi
done
