#!/usr/bin/env python3
"""Step 13: Logit lens comparison between base and fine-tuned models.

Tracks how medical h-neuron suppression changes internal triage trajectory
in the fine-tuned model vs base model. Compares the "decision point" layer
where suppression shifts output from reassurance toward hazard detection.

Uses the same LogitLensExtractor and token groups as step 05, but runs on
both the base and fine-tuned models with both TriviaQA and medical h-neurons.

Output:
    output/finetuned_logit_lens.json       -- per-layer group probs (both models)
    output/finetuned_logit_lens_summary.json -- decision point comparison
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Llama-3.1-8B-Instruct")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
FINETUNED_MODEL_DIR = OUTPUT_DIR / "finetuned_model"

ANALYSIS_ALPHAS = [0.5, 1.0, 2.0]
MAX_CASES = 40
SEED = 42

sys.path.insert(0, str(Path(__file__).parent))
from config import SYSTEM_PROMPT, SAFETY_HINT

SAFETY_TOKEN_GROUPS = {
    "emergency": ["emergency", "911", "ambulance", "ER", "immediately", "life"],
    "urgent": ["urgent", "doctor", "prescriber", "provider", "soon", "today"],
    "routine": ["routine", "follow", "monitor", "check", "schedule", "appointment"],
    "reassure": ["reassure", "unlikely", "normal", "benign", "fine", "okay", "safe"],
    "hedging": ["however", "although", "but", "might", "could", "possibly", "uncertain"],
}




def build_chat_prompt(tokenizer, message: str) -> str:
    """Build prompt using chat template (matches fine-tuning format)."""
    system = SYSTEM_PROMPT + SAFETY_HINT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Patient message: {message}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_token_group_ids(tokenizer, groups: dict) -> dict:
    group_ids = {}
    for group_name, keywords in groups.items():
        ids = set()
        for kw in keywords:
            for variant in [kw, kw.lower(), kw.upper(), kw.capitalize(), f" {kw}", f" {kw.lower()}"]:
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                ids.update(tokens)
        group_ids[group_name] = sorted(ids)
    return group_ids


class HNeuronAblator:
    def __init__(self, model, h_neurons: list[dict]):
        self.model = model
        self.h_neurons = h_neurons
        self._original_forwards = {}
        self.layer_neurons = {}
        for hn in h_neurons:
            layer = hn["layer"]
            if layer not in self.layer_neurons:
                self.layer_neurons[layer] = []
            self.layer_neurons[layer].append(hn["neuron"])
        for layer in self.layer_neurons:
            self.layer_neurons[layer] = sorted(set(self.layer_neurons[layer]))

    def apply(self, alpha: float):
        self.restore()
        for layer_idx, neuron_indices in self.layer_neurons.items():
            mlp = self.model.model.layers[layer_idx].mlp
            self._original_forwards[layer_idx] = mlp.forward
            neuron_idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)

            def make_patched_forward(orig_mlp, idx_tensor, a):
                def patched_forward(x):
                    gate = orig_mlp.gate_proj(x)
                    up = orig_mlp.up_proj(x)
                    act = F.silu(gate) * up
                    idx = idx_tensor.to(act.device)
                    act[:, :, idx] = act[:, :, idx] * a
                    return orig_mlp.down_proj(act)
                return patched_forward
            mlp.forward = make_patched_forward(mlp, neuron_idx_tensor, alpha)

    def restore(self):
        for layer_idx, orig_forward in self._original_forwards.items():
            self.model.model.layers[layer_idx].mlp.forward = orig_forward
        self._original_forwards.clear()


def extract_layer_probs(model, tokenizer, case, group_ids, ablator, alpha, device):
    """Extract per-layer token group probabilities for a single case."""
    message = case.get("message", case.get("prompt", ""))
    prompt = build_chat_prompt(tokenizer, message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    ablator.apply(alpha)

    # Register hooks
    hidden_states = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                hidden_states[idx] = out[0].detach() if isinstance(out, tuple) else out.detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(inputs["input_ids"])

    for h in hooks:
        h.remove()

    norm = model.model.norm
    lm_head = model.lm_head

    per_layer = {}
    for layer_idx, hidden in hidden_states.items():
        h = hidden[:, -1, :]
        logits = lm_head(norm(h)).squeeze(0).float().cpu()
        probs = F.softmax(logits, dim=-1)
        gp = {}
        for gname, ids in group_ids.items():
            gp[gname] = float(probs[ids].sum().item()) if ids else 0.0
        per_layer[layer_idx] = gp

    ablator.restore()
    return per_layer


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "finetuned_logit_lens.json"

    # Load test cases (physician only, smaller)
    physician_path = DATA_DIR / "physician_test.json"
    with open(physician_path) as f:
        all_cases = json.load(f)

    # Select cases: prioritize FN from 2x2 results if available
    two_by_two_path = OUTPUT_DIR / "two_by_two_results.json"
    selected_indices = []
    if two_by_two_path.exists():
        with open(two_by_two_path) as f:
            results_2x2 = json.load(f)
        # Get FNs from Arm D baseline
        arm_d_baseline = [r for r in results_2x2
                          if r["arm"] == "D_finetuned_medical"
                          and r["dataset"] == "physician" and r["alpha"] == 1.0]
        fn_idx = [r["case_index"] for r in arm_d_baseline
                  if r["detection_truth"] == 1 and r["detection_pred"] == 0]
        tp_idx = [r["case_index"] for r in arm_d_baseline
                  if r["detection_truth"] == 1 and r["detection_pred"] == 1]
        tn_idx = [r["case_index"] for r in arm_d_baseline
                  if r["detection_truth"] == 0 and r["detection_pred"] == 0]
        rng = np.random.RandomState(SEED)
        selected_indices.extend(fn_idx[:15])
        if tp_idx:
            selected_indices.extend(rng.choice(tp_idx, min(15, len(tp_idx)), replace=False).tolist())
        if tn_idx:
            n_tn = MAX_CASES - len(selected_indices)
            selected_indices.extend(rng.choice(tn_idx, min(n_tn, len(tn_idx)), replace=False).tolist())
    if not selected_indices:
        selected_indices = list(range(min(MAX_CASES, len(all_cases))))

    cases = [all_cases[i] for i in selected_indices]
    print(f"Selected {len(cases)} physician cases for logit lens")

    # Load h-neuron sets
    with open(OUTPUT_DIR / "h_neurons.json") as f:
        triviaqa_neurons = json.load(f)["h_neurons"]
    with open(OUTPUT_DIR / "medical_h_neurons.json") as f:
        medical_neurons = json.load(f)["medical_h_neurons"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configurations to run: (model_name, model_path, neuron_name, neurons)
    configs = [
        ("base_triviaqa", BASE_MODEL_ID, "triviaqa", triviaqa_neurons),
        ("base_medical", BASE_MODEL_ID, "medical", medical_neurons),
        ("finetuned_triviaqa", str(FINETUNED_MODEL_DIR), "triviaqa", triviaqa_neurons),
        ("finetuned_medical", str(FINETUNED_MODEL_DIR), "medical", medical_neurons),
    ]

    all_results = {}

    # Group by model to avoid redundant loading
    model_configs = {}
    for config_name, model_path, neuron_name, neurons in configs:
        if model_path not in model_configs:
            model_configs[model_path] = []
        model_configs[model_path].append((config_name, neuron_name, neurons))

    for model_path, config_list in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Loading: {model_path}")
        print(f"{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        group_ids = get_token_group_ids(tokenizer, SAFETY_TOKEN_GROUPS)

        for config_name, neuron_name, neurons in config_list:
            print(f"\n  Config: {config_name} ({len(neurons)} {neuron_name} neurons)")
            ablator = HNeuronAblator(model, neurons)

            config_results = {}
            for alpha in ANALYSIS_ALPHAS:
                print(f"    alpha={alpha}...")
                alpha_results = []
                for ci, (orig_idx, case) in enumerate(zip(selected_indices, cases)):
                    if ci % 10 == 0:
                        print(f"      Case {ci}/{len(cases)}...")
                        sys.stdout.flush()
                    per_layer = extract_layer_probs(
                        model, tokenizer, case, group_ids, ablator, alpha, device)
                    alpha_results.append({
                        "case_index": orig_idx,
                        "detection_truth": case.get("detection_truth", 0),
                        "per_layer": {str(k): v for k, v in per_layer.items()},
                    })
                config_results[str(alpha)] = alpha_results

            all_results[config_name] = config_results

        del model
        torch.cuda.empty_cache()

    # Save raw results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Compute summary: decision point layer for each config
    summary = {}
    n_layers = 32
    for config_name in all_results:
        baseline = all_results[config_name].get("1.0", [])
        suppressed = all_results[config_name].get("0.5", [])
        if not baseline or not suppressed:
            continue

        # Mean emergency prob shift per layer (suppression vs baseline)
        emergency_shifts = []
        reassure_shifts = []
        for bl, sup in zip(baseline, suppressed):
            if bl["detection_truth"] != 1:
                continue
            e_shift = []
            r_shift = []
            for l in range(n_layers):
                bl_probs = bl["per_layer"].get(str(l), {})
                sup_probs = sup["per_layer"].get(str(l), {})
                e_shift.append(
                    sup_probs.get("emergency", 0) - bl_probs.get("emergency", 0))
                r_shift.append(
                    sup_probs.get("reassure", 0) - bl_probs.get("reassure", 0))
            emergency_shifts.append(e_shift)
            reassure_shifts.append(r_shift)

        if emergency_shifts:
            mean_e = np.mean(emergency_shifts, axis=0)
            mean_r = np.mean(reassure_shifts, axis=0)
            decision_layer = int(np.argmax(np.abs(mean_e)))
            summary[config_name] = {
                "n_hazard_cases": len(emergency_shifts),
                "decision_layer": decision_layer,
                "peak_emergency_shift": round(float(mean_e[decision_layer]), 6),
                "peak_reassure_shift": round(float(mean_r[decision_layer]), 6),
                "mean_emergency_shift_by_layer": [round(float(x), 6) for x in mean_e],
                "mean_reassure_shift_by_layer": [round(float(x), 6) for x in mean_r],
            }
            print(f"\n{config_name}:")
            print(f"  Decision layer: {decision_layer}")
            print(f"  Peak emergency shift: {mean_e[decision_layer]:.6f}")
            print(f"  Peak reassure shift: {mean_r[decision_layer]:.6f}")

    summary_path = OUTPUT_DIR / "finetuned_logit_lens_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
