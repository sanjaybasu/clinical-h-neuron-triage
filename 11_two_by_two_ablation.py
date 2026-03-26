#!/usr/bin/env python3
"""Step 11: 2x2 factorial ablation sweep.

Four arms:
  A: Base model + TriviaQA h-neurons
  B: Base model + medical h-neurons
  C: Fine-tuned model + TriviaQA h-neurons
  D: Fine-tuned model + medical h-neurons  (PRIMARY)

Each arm: 8 alpha levels x physician (200) + real-world (2000) test sets.

Key metric: FN->TP conversion rate under suppression (alpha < 1.0).
Statistical test: interaction test via ordinal regression.

Output:
    output/two_by_two_results.json    -- all per-case results (4 arms x 8 alphas)
    output/two_by_two_summary.json    -- aggregate metrics per arm/alpha
    output/two_by_two_interaction.json -- interaction test results
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

ALPHA_LEVELS = [0.0, 1.0, 3.0]  # Focused: suppression, baseline, amplification
MAX_NEW_TOKENS = 256
SEED = 42

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SYSTEM_PROMPT, SAFETY_HINT, EMERGENCY_KEYWORDS, URGENT_KEYWORDS,
)




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


def parse_response(text: str) -> dict:
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
        self.layer_neurons = {}
        for hn in self.h_neurons:
            layer = hn["layer"]
            if layer not in self.layer_neurons:
                self.layer_neurons[layer] = []
            self.layer_neurons[layer].append(hn["neuron"])
        for layer in self.layer_neurons:
            self.layer_neurons[layer] = sorted(set(self.layer_neurons[layer]))
        print(f"  Ablator: {len(self.layer_neurons)} layers, "
              f"{sum(len(v) for v in self.layer_neurons.values())} neurons")

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
                    # Cast to float32 before scaling to avoid bfloat16 overflow
                    # (bfloat16 max ~65504; alpha=3.0 amplification can overflow)
                    orig_dtype = act.dtype
                    act_f32 = act.to(torch.float32)
                    act_f32[:, :, idx] = act_f32[:, :, idx] * a
                    act = act_f32.to(orig_dtype)
                    return orig_mlp.down_proj(act)
                return patched_forward

            mlp.forward = make_patched_forward(mlp, neuron_idx_tensor, alpha)

    def restore(self):
        for layer_idx, orig_forward in self._original_forwards.items():
            self.model.model.layers[layer_idx].mlp.forward = orig_forward
        self._original_forwards.clear()


def run_arm(model, tokenizer, ablator, cases, arm_name, dataset_name, alpha, device,
            skip_indices: set = None):
    """Run inference for one arm/alpha/dataset combination.

    Args:
        skip_indices: set of case_index values already completed (for intra-arm resume).
    """
    ablator.apply(alpha)
    if skip_indices is None:
        skip_indices = set()
    results = []
    n_errors = 0
    n_skipped = 0
    for i, case in enumerate(cases):
        if i in skip_indices:
            n_skipped += 1
            continue
        message = case.get("message", case.get("prompt", ""))
        prompt = build_chat_prompt(tokenizer, message)
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            parsed = parse_response(response)
        except Exception as exc:
            n_errors += 1
            print(f"  [WARN] Case {i} inference error (alpha={alpha}): {exc}. Fallback: benign.")
            sys.stdout.flush()
            torch.cuda.empty_cache()
            response = f"[INFERENCE_ERROR: {str(exc)[:80]}]"
            parsed = {"detection": 0, "severity": "Benign", "action": "None"}

        results.append({
            "arm": arm_name,
            "dataset": dataset_name,
            "alpha": alpha,
            "case_index": i,
            "detection_truth": case.get("detection_truth", 0),
            "detection_pred": parsed["detection"],
            "action_truth": case.get("action_truth", "None"),
            "action_pred": parsed["action"],
            "severity_pred": parsed["severity"],
            "response": response[:300],
        })
    if n_errors > 0:
        print(f"  [WARN] {n_errors}/{len(cases) - n_skipped} cases used benign fallback "
              f"due to inference errors")
    if n_skipped > 0:
        print(f"  [INFO] {n_skipped} cases skipped (intra-arm resume)")
    ablator.restore()
    return results


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    rad = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, (center - rad) / denom), min(1.0, (center + rad) / denom))


def compute_metrics(results: list) -> dict:
    y_true = np.array([r["detection_truth"] for r in results])
    y_pred = np.array([r["detection_pred"] for r in results])
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / max(denom, 1e-10)
    return {
        "n": len(results), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "sensitivity_ci": [round(x, 4) for x in wilson_ci(tp, tp + fn)],
        "specificity_ci": [round(x, 4) for x in wilson_ci(tn, tn + fp)],
        "mcc": round(float(mcc), 4),
    }


def compute_fn_to_tp_conversions(baseline_results, ablated_results):
    """Count cases that were FN at baseline but TP after ablation."""
    baseline_map = {}
    for r in baseline_results:
        key = (r["dataset"], r["case_index"])
        baseline_map[key] = r

    conversions = []
    for r in ablated_results:
        key = (r["dataset"], r["case_index"])
        base = baseline_map.get(key)
        if base is None:
            continue
        if (base["detection_truth"] == 1 and base["detection_pred"] == 0
                and r["detection_pred"] == 1):
            conversions.append({
                "case_index": r["case_index"],
                "dataset": r["dataset"],
                "alpha": r["alpha"],
                "action_truth": r["action_truth"],
                "action_pred": r["action_pred"],
            })
    return conversions


def interaction_test(summary: dict) -> dict:
    """Test 2x2 interaction: does h-neuron type x model type predict rescue rate?

    Uses the FN->TP conversion count at alpha=0.5 across 4 arms.
    Ordinal regression would require statsmodels; here we compute the
    interaction contrast and a Fisher's exact test on the 2x2 table.
    """
    from scipy.stats import fisher_exact

    arms = {}
    for entry in summary:
        if entry["alpha"] == 0.5 and entry["dataset"] == "physician":
            arms[entry["arm"]] = entry

    if len(arms) < 4:
        return {"error": "Not all 4 arms available at alpha=0.5"}

    # 2x2 table: rescued FNs vs unrescued FNs
    def get_fn_tp(arm_name):
        m = arms[arm_name]
        baseline_fn = m.get("baseline_fn", m["fn"] + m["tp"])  # total true hazards
        rescued = m.get("fn_to_tp", 0)
        return rescued, baseline_fn - rescued

    a_r, a_u = get_fn_tp("A_base_triviaqa")
    b_r, b_u = get_fn_tp("B_base_medical")
    c_r, c_u = get_fn_tp("C_finetuned_triviaqa")
    d_r, d_u = get_fn_tp("D_finetuned_medical")

    # Interaction contrast: (D - C) - (B - A) in rescue rates
    n_a = max(a_r + a_u, 1)
    n_b = max(b_r + b_u, 1)
    n_c = max(c_r + c_u, 1)
    n_d = max(d_r + d_u, 1)
    interaction = (d_r / n_d - c_r / n_c) - (b_r / n_b - a_r / n_a)

    # Fisher's exact on Arm D vs pooled others
    others_r = a_r + b_r + c_r
    others_u = a_u + b_u + c_u
    table = [[d_r, d_u], [others_r, others_u]]
    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    return {
        "interaction_contrast": round(interaction, 4),
        "arm_d_rescue_rate": round(d_r / n_d, 4) if n_d > 0 else 0,
        "arm_c_rescue_rate": round(c_r / n_c, 4) if n_c > 0 else 0,
        "arm_b_rescue_rate": round(b_r / n_b, 4) if n_b > 0 else 0,
        "arm_a_rescue_rate": round(a_r / n_a, 4) if n_a > 0 else 0,
        "fisher_exact_or": round(odds_ratio, 4),
        "fisher_exact_p": round(p_value, 6),
        "table": table,
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "two_by_two_results.json"

    # Resume from partial results if available
    all_results = []
    completed = set()
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        for r in all_results:
            completed.add((r["arm"], r["dataset"], r["alpha"]))
        print(f"Resuming: {len(completed)} arm/dataset/alpha combinations already done")

    # Load test data
    physician_path = DATA_DIR / "physician_test.json"
    realworld_path = DATA_DIR / "realworld_test.json"
    with open(physician_path) as f:
        physician_cases = json.load(f)
    with open(realworld_path) as f:
        realworld_all = json.load(f)

    # Stratified 500-case subset: all hazardous + random benign sample
    rng = np.random.RandomState(SEED)
    hazardous_idx = [i for i, c in enumerate(realworld_all) if c.get("detection_truth") == 1]
    benign_idx = [i for i, c in enumerate(realworld_all) if c.get("detection_truth") == 0]
    n_benign_sample = min(500 - len(hazardous_idx), len(benign_idx))
    sampled_benign = rng.choice(benign_idx, n_benign_sample, replace=False).tolist()
    stratified_idx = sorted(hazardous_idx + sampled_benign)
    realworld_cases = [realworld_all[i] for i in stratified_idx]
    print(f"Real-world stratified subset: {len(realworld_cases)} cases "
          f"({len(hazardous_idx)} hazardous + {n_benign_sample} benign)")

    test_sets = {"physician": physician_cases, "realworld": realworld_cases}

    # Load h-neuron sets
    triviaqa_path = OUTPUT_DIR / "h_neurons.json"
    medical_path = OUTPUT_DIR / "medical_h_neurons.json"

    with open(triviaqa_path) as f:
        triviaqa_neurons = json.load(f)["h_neurons"]
    with open(medical_path) as f:
        medical_neurons = json.load(f)["medical_h_neurons"]
    print(f"TriviaQA h-neurons: {len(triviaqa_neurons)}")
    print(f"Medical h-neurons: {len(medical_neurons)}")

    # Define arms
    arms = [
        ("A_base_triviaqa", BASE_MODEL_ID, triviaqa_neurons),
        ("B_base_medical", BASE_MODEL_ID, medical_neurons),
        ("C_finetuned_triviaqa", str(FINETUNED_MODEL_DIR), triviaqa_neurons),
        ("D_finetuned_medical", str(FINETUNED_MODEL_DIR), medical_neurons),
    ]

    # Group arms by model to avoid reloading
    model_arms = {}
    for arm_name, model_path, neurons in arms:
        if model_path not in model_arms:
            model_arms[model_path] = []
        model_arms[model_path].append((arm_name, neurons))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_path, arm_list in model_arms.items():
        # Check if all combos for this model are done
        all_done = True
        for arm_name, _ in arm_list:
            for ds_name in test_sets:
                for alpha in ALPHA_LEVELS:
                    if (arm_name, ds_name, alpha) not in completed:
                        all_done = False
                        break
        if all_done:
            print(f"\nAll arms for {model_path} already complete, skipping model load")
            continue

        print(f"\n{'='*60}")
        print(f"Loading model: {model_path}")
        print(f"{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()

        for arm_name, neurons in arm_list:
            print(f"\n--- Arm {arm_name} ---")
            ablator = HNeuronAblator(model, neurons)

            # Physician first (smaller), then real-world
            for ds_name in ["physician", "realworld"]:
                cases = test_sets[ds_name]
                for alpha in ALPHA_LEVELS:
                    if (arm_name, ds_name, alpha) in completed:
                        print(f"  {arm_name}/{ds_name}/alpha={alpha}: done, skipping")
                        continue
                    print(f"  {arm_name}/{ds_name}/alpha={alpha} ({len(cases)} cases)...")
                    sys.stdout.flush()
                    results = run_arm(model, tokenizer, ablator, cases,
                                      arm_name, ds_name, alpha, device)
                    all_results.extend(results)
                    completed.add((arm_name, ds_name, alpha))

                    # Quick summary
                    n_det = sum(1 for r in results if r["detection_pred"] == 1)
                    print(f"    -> {n_det}/{len(results)} detected")

                    # Incremental save
                    with open(results_path, "w") as f:
                        json.dump(all_results, f)
                    vol_path = Path("/results/output")
                    if vol_path.parent.exists():
                        vol_path.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy(str(results_path), str(vol_path / "two_by_two_results.json"))

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # Compute summary metrics
    print(f"\n{'='*60}")
    print("COMPUTING SUMMARY METRICS")
    print(f"{'='*60}")

    summary = []
    for arm_name in ["A_base_triviaqa", "B_base_medical",
                     "C_finetuned_triviaqa", "D_finetuned_medical"]:
        for ds_name in ["physician", "realworld"]:
            # Get baseline results (alpha=1.0)
            baseline = [r for r in all_results
                        if r["arm"] == arm_name and r["dataset"] == ds_name
                        and r["alpha"] == 1.0]
            for alpha in ALPHA_LEVELS:
                arm_ds_alpha = [r for r in all_results
                                if r["arm"] == arm_name and r["dataset"] == ds_name
                                and r["alpha"] == alpha]
                if not arm_ds_alpha:
                    continue
                metrics = compute_metrics(arm_ds_alpha)
                conversions = compute_fn_to_tp_conversions(baseline, arm_ds_alpha)
                entry = {
                    "arm": arm_name, "dataset": ds_name, "alpha": alpha,
                    **metrics,
                    "fn_to_tp": len(conversions),
                    "baseline_fn": metrics["fn"] + len(conversions),
                }
                summary.append(entry)
                if ds_name == "physician":
                    print(f"  {arm_name} | {ds_name} | a={alpha:.2f} | "
                          f"sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f} "
                          f"mcc={metrics['mcc']:.3f} FN->TP={len(conversions)}")

    with open(OUTPUT_DIR / "two_by_two_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Interaction test
    print(f"\n{'='*60}")
    print("INTERACTION TEST")
    print(f"{'='*60}")
    try:
        interaction = interaction_test(summary)
        print(f"  Interaction contrast: {interaction['interaction_contrast']}")
        print(f"  Arm D rescue rate: {interaction['arm_d_rescue_rate']}")
        print(f"  Fisher's exact p: {interaction['fisher_exact_p']}")
        with open(OUTPUT_DIR / "two_by_two_interaction.json", "w") as f:
            json.dump(interaction, f, indent=2)
    except Exception as e:
        print(f"  Interaction test failed: {e}")
        interaction = {"error": str(e)}
        with open(OUTPUT_DIR / "two_by_two_interaction.json", "w") as f:
            json.dump(interaction, f, indent=2)

    print(f"\nDone. Results: {results_path}")


if __name__ == "__main__":
    main()
