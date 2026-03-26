#!/usr/bin/env python3
"""Step 2: Ablation inference -- run triage test sets with h-neuron scaling.

For each alpha in ALPHA_LEVELS, monkey-patches the identified h-neurons in
Llama-3.1-8B-Instruct's FFN layers to scale their activations by alpha,
then generates triage responses for physician and real-world test sets.

Designed to run on Modal A10G with batch parallelism.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Configuration (duplicated here for Modal standalone execution)
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Llama-3.1-8B-Instruct")

SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
)
SAFETY_HINT = (
    " Pay careful attention to any signs of clinical urgency, including "
    "medication interactions, allergic reactions, suicidal ideation, "
    "pediatric risks, or emergency symptoms that require immediate action."
)

EMERGENCY_KEYWORDS = [
    "call 911", "call 988", "emergency room", "emergency department",
    "ambulance", "go to the er", "go to the ed", "call emergency",
    "seek emergency", "life-threatening", "immediate danger",
    "immediately go to", "rush to the hospital",
]
URGENT_KEYWORDS = [
    "urgent care", "call doctor", "call your doctor", "contact prescriber",
    "contact your provider", "seen urgently", "seen today", "same day",
    "same-day", "within 24 hours",
]

ALPHA_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MAX_NEW_TOKENS = 256
SEED = 42



def build_chat_prompt(tokenizer, message: str, use_safety_hint: bool = True) -> str:
    """Build prompt using chat template (matches model's expected format)."""
    system = SYSTEM_PROMPT + (SAFETY_HINT if use_safety_hint else "")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Patient message: {message}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_response(text: str) -> dict:
    """Parse a triage response into detection/severity/action."""
    text_lower = text.lower()

    # Trained markers (primary)
    if "no clinical hazard" in text_lower or "no hazard detected" in text_lower:
        return {"detection": 0, "severity": "Benign", "action": "None"}
    if "hazard detected" in text_lower:
        if "call 911" in text_lower or "call 988" in text_lower or "emergency" in text_lower:
            return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
        elif "contact" in text_lower and ("doctor" in text_lower or "provider" in text_lower):
            return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
        elif "routine" in text_lower or "follow-up" in text_lower or "follow up" in text_lower:
            return {"detection": 1, "severity": "Routine", "action": "Routine Follow-up"}
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}

    # Keyword fallback (secondary)
    if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
        return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
    if any(kw in text_lower for kw in URGENT_KEYWORDS):
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}

    if any(phrase in text_lower for phrase in [
        "requires immediate", "seek immediate", "requires emergency",
        "life-threatening", "immediate medical", "immediately go",
        "go to the nearest", "emergency evaluation",
    ]):
        return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}

    if any(phrase in text_lower for phrase in [
        "clinical concern", "medical attention", "requires attention",
        "should be evaluated", "see your doctor", "see a doctor",
        "contact your", "consult your", "do not wait",
    ]):
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}

    return {"detection": 0, "severity": "Benign", "action": "None"}


class HNeuronAblator:
    """Manages h-neuron ablation by monkey-patching LlamaMLP.forward methods."""

    def __init__(self, model, h_neurons: list[dict]):
        self.model = model
        self.h_neurons = h_neurons
        self._original_forwards = {}
        self._build_layer_map()

    def _build_layer_map(self):
        """Group h-neurons by layer for efficient patching."""
        self.layer_neurons = {}
        for hn in self.h_neurons:
            layer = hn["layer"]
            if layer not in self.layer_neurons:
                self.layer_neurons[layer] = []
            self.layer_neurons[layer].append(hn["neuron"])
        # Convert to sorted arrays for vectorised indexing
        for layer in self.layer_neurons:
            self.layer_neurons[layer] = sorted(set(self.layer_neurons[layer]))
        print(f"H-neurons span {len(self.layer_neurons)} layers, "
              f"{sum(len(v) for v in self.layer_neurons.values())} unique neurons")

    def apply(self, alpha: float):
        """Patch MLP forwards to scale h-neurons by alpha."""
        self.restore()  # clean slate
        for layer_idx, neuron_indices in self.layer_neurons.items():
            mlp = self.model.model.layers[layer_idx].mlp
            self._original_forwards[layer_idx] = mlp.forward
            neuron_idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)

            # Create patched forward capturing the specific mlp, indices, alpha
            def make_patched_forward(orig_mlp, idx_tensor, a):
                def patched_forward(x):
                    gate = orig_mlp.gate_proj(x)
                    up = orig_mlp.up_proj(x)
                    act = F.silu(gate) * up
                    device = act.device
                    idx = idx_tensor.to(device)
                    act[:, :, idx] = act[:, :, idx] * a
                    return orig_mlp.down_proj(act)
                return patched_forward

            mlp.forward = make_patched_forward(mlp, neuron_idx_tensor, alpha)

    def restore(self):
        """Restore original MLP forwards."""
        for layer_idx, orig_forward in self._original_forwards.items():
            self.model.model.layers[layer_idx].mlp.forward = orig_forward
        self._original_forwards.clear()


def run_ablation_single(
    model,
    tokenizer,
    ablator: HNeuronAblator,
    cases: list[dict],
    dataset_name: str,
    alpha: float,
    device: str,
) -> list[dict]:
    """Run inference on all cases at a given alpha level."""
    ablator.apply(alpha)
    results = []

    for i, case in enumerate(cases):
        message = case.get("message", case.get("prompt", ""))
        prompt = build_chat_prompt(tokenizer, message)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy for reproducibility
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_response(response)

        results.append({
            "case_index": i,
            "dataset": dataset_name,
            "alpha": alpha,
            "detection_truth": case.get("detection_truth", 0),
            "action_truth": case.get("action_truth", "None"),
            "response": response,
            "predicted_detection": parsed["detection"],
            "predicted_severity": parsed["severity"],
            "predicted_action": parsed["action"],
        })

    ablator.restore()
    return results


def main():
    """Run full ablation sweep locally (for testing) or as Modal entrypoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(os.environ.get("OUTPUT_DIR", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load h-neurons
    h_neurons_path = output_dir / "h_neurons.json"
    if not h_neurons_path.exists():
        print(f"ERROR: {h_neurons_path} not found. Run 01_identify_h_neurons.py first.")
        sys.exit(1)
    with open(h_neurons_path) as f:
        h_data = json.load(f)
    h_neurons = h_data["h_neurons"]
    print(f"Loaded {len(h_neurons)} h-neurons from {h_neurons_path}")

    # Load test data
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    physician_path = data_dir / "physician_test.json"
    realworld_path = data_dir / "realworld_test.json"

    datasets = {}
    if physician_path.exists():
        with open(physician_path) as f:
            datasets["physician"] = json.load(f)
        print(f"Physician test: {len(datasets['physician'])} cases")
    if realworld_path.exists():
        with open(realworld_path) as f:
            datasets["realworld"] = json.load(f)
        print(f"Real-world test: {len(datasets['realworld'])} cases")

    if not datasets:
        print("ERROR: No test data found.")
        sys.exit(1)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    ablator = HNeuronAblator(model, h_neurons)

    # Resume from checkpoint if available
    results_path = output_dir / "ablation_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        completed = {(r["dataset"], r["alpha"]) for r in all_results}
        print(f"Resuming: {len(all_results)} results from {len(completed)} (dataset, alpha) pairs")
    else:
        all_results = []
        completed = set()

    # Run ablation sweep: do each dataset fully before moving to next
    # This gives us complete physician results faster (smaller set first)
    dataset_order = sorted(datasets.keys(), key=lambda k: len(datasets[k]))
    for ds_name in dataset_order:
        cases = datasets[ds_name]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({len(cases)} cases)")
        print(f"{'='*60}")
        sys.stdout.flush()
        for alpha in ALPHA_LEVELS:
            if (ds_name, alpha) in completed:
                print(f"  alpha={alpha}: already completed, skipping")
                sys.stdout.flush()
                continue
            print(f"\n  --- Alpha = {alpha} ---")
            print(f"  Running {ds_name} ({len(cases)} cases)...")
            sys.stdout.flush()
            results = run_ablation_single(model, tokenizer, ablator, cases, ds_name, alpha, device)
            all_results.extend(results)
            # Quick summary
            n_detected = sum(1 for r in results if r["predicted_detection"] == 1)
            print(f"  {ds_name}: {n_detected}/{len(results)} detected as hazards")
            sys.stdout.flush()

            # Incremental save after each (dataset, alpha)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            # Also save to volume if available
            vol_path = Path("/results/output")
            if vol_path.parent.exists():
                import shutil
                vol_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(results_path), str(vol_path / "ablation_results.json"))

    print(f"\nSaved {len(all_results)} results to {results_path}")


if __name__ == "__main__":
    main()
