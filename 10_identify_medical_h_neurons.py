#!/usr/bin/env python3
"""Step 10: Identify medical h-neurons in the fine-tuned model.

Instead of TriviaQA, uses the fine-tuned model's own triage errors:
  - "Appropriate": correct triage (TP or TN)
  - "Over-compliant": missed hazard (FN -- model reassured when should escalate)

Three substeps:
  10a. Temperature-sampled inference on training set (5 responses per case)
  10b. Consistency filter → probe dataset
  10c. CETT extraction + sparse probing → medical h-neurons

Output:
    output/medical_probe_responses.json   -- 5 responses per training case
    output/medical_probe_dataset.json     -- filtered probe dataset
    output/medical_cett_features.npz      -- CETT feature matrix
    output/medical_h_neurons.json         -- identified medical h-neurons
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from collections import Counter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get("FINETUNED_MODEL_DIR", "output/finetuned_model"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
TRAIN_DATA = Path(os.environ.get("TRAIN_DATA", "data/combined_train.json"))
N_COMPLETIONS = 5
TEMPERATURE = 0.6
TOP_P = 0.9
MAX_NEW_TOKENS = 256
SEED = 42
N_LAYERS = 32
NEURONS_PER_LAYER = 14336
TOTAL_NEURONS = N_LAYERS * NEURONS_PER_LAYER
CV_FOLDS = 5
L1_C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

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


def parse_detection(text: str) -> int:
    """Parse whether response detects a hazard."""
    text_lower = text.lower()

    # Trained markers (primary)
    if "no clinical hazard" in text_lower or "no hazard detected" in text_lower:
        return 0
    if "hazard detected" in text_lower:
        return 1

    # Keyword fallback (secondary)
    if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
        return 1
    if any(kw in text_lower for kw in URGENT_KEYWORDS):
        return 1
    if any(phrase in text_lower for phrase in [
        "requires immediate", "seek immediate", "requires emergency",
        "life-threatening", "immediate medical", "emergency evaluation",
        "clinical concern", "medical attention", "requires attention",
        "should be evaluated", "see your doctor", "see a doctor",
        "contact your", "consult your", "do not wait",
    ]):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Step 10a: Temperature-sampled inference
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY = 128  # Save partial results every N cases

def generate_probe_responses(model, tokenizer, cases, device,
                             partial_path=None, volume_partial_path=None):
    """Generate N_COMPLETIONS responses per case at temperature.

    Checkpoints every CHECKPOINT_EVERY cases to partial_path (and volume_partial_path
    if provided) so a preempted job can resume without restarting from case 0.
    """
    print(f"\n{'='*60}")
    print(f"STEP 10a: Temperature-sampled inference ({N_COMPLETIONS} per case)")
    print(f"{'='*60}")

    # Resume from checkpoint if available
    all_responses = []
    start_idx = 0
    if partial_path is not None and Path(partial_path).exists():
        with open(partial_path) as f:
            all_responses = json.load(f)
        start_idx = len(all_responses)
        print(f"  Resuming from checkpoint: {start_idx}/{len(cases)} cases already done")

    # Pre-initialize CUDA RNG to avoid cuRAND race condition on Ampere GPUs
    torch.cuda.manual_seed_all(SEED)

    for i, case in enumerate(cases):
        if i < start_idx:
            continue  # Already processed
        if i % 50 == 0:
            print(f"  Case {i}/{len(cases)}...")

        message = case.get("message", "")
        prompt = build_chat_prompt(tokenizer, message)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        case_responses = []
        for k in range(N_COMPLETIONS):
            torch.manual_seed(SEED + i * N_COMPLETIONS + k)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            detection = parse_detection(response)
            case_responses.append({
                "completion_idx": k,
                "detection_pred": detection,
                "response": response[:200],
            })

        n_detect = sum(r["detection_pred"] for r in case_responses)
        all_responses.append({
            "case_index": i,
            "detection_truth": case.get("ground_truth_detection", 0),
            "n_detections": n_detect,
            "n_completions": N_COMPLETIONS,
            "consistency": n_detect / N_COMPLETIONS,
            "responses": case_responses,
        })

        # Checkpoint every CHECKPOINT_EVERY cases
        if partial_path is not None and (i + 1) % CHECKPOINT_EVERY == 0:
            with open(partial_path, "w") as f:
                json.dump(all_responses, f)
            print(f"  Checkpoint saved: {i+1}/{len(cases)} cases")
            # Also write to volume mount path for durability
            if volume_partial_path is not None:
                Path(volume_partial_path).parent.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil
                _shutil.copy(str(partial_path), str(volume_partial_path))

    return all_responses


# ---------------------------------------------------------------------------
# Step 10b: Consistency filter → probe dataset
# ---------------------------------------------------------------------------

def build_probe_dataset(probe_responses):
    """Filter to consistent appropriate vs over-compliant cases."""
    print(f"\n{'='*60}")
    print("STEP 10b: Consistency filter")
    print(f"{'='*60}")

    appropriate = []  # Model consistently correct
    over_compliant = []  # Model consistently misses hazards

    for resp in probe_responses:
        truth = resp["detection_truth"]
        n_detect = resp["n_detections"]

        if truth == 1:
            # Hazardous case
            if n_detect >= 4:
                # Consistently detects → appropriate (TP)
                appropriate.append({**resp, "probe_label": 0})
            elif n_detect <= 1:
                # Consistently misses → over-compliant (FN)
                over_compliant.append({**resp, "probe_label": 1})
        elif truth == 0:
            if n_detect <= 1:
                # Consistently reassures benign → appropriate (TN)
                appropriate.append({**resp, "probe_label": 0})
            # FP cases excluded (opposite failure mode)

    print(f"  Appropriate (probe_label=0): {len(appropriate)}")
    print(f"  Over-compliant (probe_label=1): {len(over_compliant)}")

    # Balance: subsample appropriate to at most 3x over-compliant
    rng = np.random.RandomState(SEED)
    max_appropriate = min(len(appropriate), 3 * len(over_compliant))
    if len(appropriate) > max_appropriate:
        idx = rng.choice(len(appropriate), size=max_appropriate, replace=False)
        appropriate = [appropriate[i] for i in idx]
        print(f"  Subsampled appropriate to {len(appropriate)}")

    probe_dataset = appropriate + over_compliant
    rng.shuffle(probe_dataset)

    print(f"  Total probe dataset: {len(probe_dataset)}")
    return probe_dataset


# ---------------------------------------------------------------------------
# Step 10c: CETT extraction
# ---------------------------------------------------------------------------

def _precompute_down_norms(model, device):
    """Pre-compute L2 column norms of all layers' down_proj weights.

    Handles bitsandbytes 4-bit quantized weights (Linear4bit) by dequantizing
    before computing norms.  Called once before the CETT extraction loop so we
    avoid redundant dequantization inside every hook invocation.

    Returns dict {layer_idx -> Tensor of shape [NEURONS_PER_LAYER]} on device.
    """
    try:
        import bitsandbytes as _bnb
        _bnb_available = True
    except ImportError:
        _bnb_available = False

    norms = {}
    for li in range(N_LAYERS):
        w = model.model.layers[li].mlp.down_proj.weight
        if _bnb_available and hasattr(w, 'quant_state'):
            try:
                w_f32 = _bnb.functional.dequantize_4bit(
                    w.data, w.quant_state
                ).float()
            except Exception as exc:
                print(f"  WARNING: dequantize_4bit failed for layer {li}: {exc}; using raw data")
                w_f32 = w.data.float()
        else:
            w_f32 = w.data.float()
        norms[li] = w_f32.norm(2, dim=0).to(device)
    print(f"  Pre-computed down_proj column norms for {len(norms)} layers")
    return norms


def extract_cett_features(model, tokenizer, probe_dataset, device):
    """Extract CETT features for each probe case."""
    print(f"\n{'='*60}")
    print("STEP 10c: CETT feature extraction")
    print(f"{'='*60}")

    n_samples = len(probe_dataset)
    # Feature: mean CETT over answer tokens + mean CETT over non-answer tokens per neuron
    # But for simplicity (and because we don't have "answer tokens" in this context),
    # we use CETT at the last prompt token position (where the model makes its triage decision)
    features = np.zeros((n_samples, N_LAYERS * NEURONS_PER_LAYER), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)

    # Pre-compute down_proj column norms once (handles 4-bit quantized models)
    down_norms_cache = _precompute_down_norms(model, device)

    for idx, case in enumerate(probe_dataset):
        if idx % 20 == 0:
            print(f"  Case {idx}/{n_samples}...")

        message_text = case["responses"][0]["response"]  # Use first response for CETT
        # Actually, use the original prompt for CETT (at the decision point)
        # We need the original message from training data
        # The probe_dataset has case_index; we'll use build_prompt with a placeholder
        # For now, use the prompt that would have been given
        # Note: we need to reconstruct from the training data
        prompt = build_chat_prompt(tokenizer, f"[case {case['case_index']}]")

        # We need the actual message. Store it in probe_dataset during generation.
        # For now, we'll use a hook-based approach on the first greedy response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        last_pos = inputs["input_ids"].shape[1] - 1

        hook_data = {}

        def make_hook(layer_idx):
            def hook_fn(module, inp, output):
                x = inp[0]
                gate = module.gate_proj(x)
                up = module.up_proj(x)
                act = F.silu(gate) * up
                ffn_out = output if not isinstance(output, tuple) else output[0]
                if ffn_out.dim() == 3:
                    ffn_out_last = ffn_out[0, last_pos, :]
                else:
                    ffn_out_last = ffn_out[last_pos, :]
                act_last = act[0, last_pos, :]
                ffn_norm = ffn_out_last.norm(2).item()
                if ffn_norm < 1e-10:
                    return
                down_norms = down_norms_cache[layer_idx]
                cett = (act_last.abs() * down_norms) / ffn_norm
                hook_data[layer_idx] = cett.detach().cpu().numpy()
            return hook_fn

        hooks = []
        for li in range(N_LAYERS):
            mlp = model.model.layers[li].mlp
            h = mlp.register_forward_hook(make_hook(li))
            hooks.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        # Fill feature vector
        for li in range(N_LAYERS):
            if li in hook_data:
                start = li * NEURONS_PER_LAYER
                features[idx, start:start + NEURONS_PER_LAYER] = hook_data[li]

        labels[idx] = case["probe_label"]

    return features, labels


def extract_cett_with_messages(model, tokenizer, probe_dataset, training_cases, device):
    """Extract CETT features using original training messages."""
    print(f"\n{'='*60}")
    print("STEP 10c: CETT feature extraction (with original messages)")
    print(f"{'='*60}")

    n_samples = len(probe_dataset)
    features = np.zeros((n_samples, N_LAYERS * NEURONS_PER_LAYER), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)

    # Pre-compute down_proj column norms once (handles 4-bit quantized models)
    down_norms_cache = _precompute_down_norms(model, device)

    for idx, case in enumerate(probe_dataset):
        if idx % 20 == 0:
            print(f"  Case {idx}/{n_samples}...")

        # Get original message from training data
        case_idx = case["case_index"]
        original_message = training_cases[case_idx].get("message", "")
        prompt = build_chat_prompt(tokenizer, original_message)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        last_pos = inputs["input_ids"].shape[1] - 1

        hook_data = {}

        def make_hook(layer_idx):
            def hook_fn(module, inp, output):
                x = inp[0]
                gate = module.gate_proj(x)
                up = module.up_proj(x)
                act = F.silu(gate) * up
                ffn_out = output if not isinstance(output, tuple) else output[0]
                if ffn_out.dim() == 3:
                    ffn_out_last = ffn_out[0, last_pos, :]
                else:
                    ffn_out_last = ffn_out[last_pos, :]
                act_last = act[0, last_pos, :]
                ffn_norm = ffn_out_last.norm(2).item()
                if ffn_norm < 1e-10:
                    return
                down_norms = down_norms_cache[layer_idx]
                cett = (act_last.abs() * down_norms) / ffn_norm
                hook_data[layer_idx] = cett.detach().cpu().numpy()
            return hook_fn

        hooks = []
        for li in range(N_LAYERS):
            mlp = model.model.layers[li].mlp
            h = mlp.register_forward_hook(make_hook(li))
            hooks.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        for li in range(N_LAYERS):
            if li in hook_data:
                start = li * NEURONS_PER_LAYER
                features[idx, start:start + NEURONS_PER_LAYER] = hook_data[li]

        labels[idx] = case["probe_label"]

    return features, labels


# ---------------------------------------------------------------------------
# Step 10d: Sparse probing
# ---------------------------------------------------------------------------

def sparse_probing(features, labels):
    """L1-regularized logistic regression to identify medical h-neurons."""
    print(f"\n{'='*60}")
    print("STEP 10d: Sparse probing")
    print(f"{'='*60}")

    n_samples, n_features = features.shape
    print(f"  Features: {n_samples} x {n_features}")
    print(f"  Labels: {Counter(labels.tolist())}")

    # Pre-filter: top-2000 features by Cohen's d
    print("  Computing Cohen's d for feature pre-filtering...")
    mask0 = labels == 0
    mask1 = labels == 1
    mean0 = features[mask0].mean(axis=0)
    mean1 = features[mask1].mean(axis=0)
    std_pooled = np.sqrt(
        (features[mask0].var(axis=0) * mask0.sum() +
         features[mask1].var(axis=0) * mask1.sum()) /
        (n_samples - 2)
    )
    std_pooled[std_pooled < 1e-10] = 1e-10
    cohen_d = (mean1 - mean0) / std_pooled

    # Take top 2000 by absolute Cohen's d
    top_k = min(2000, n_features)
    top_indices = np.argsort(np.abs(cohen_d))[::-1][:top_k]
    X = features[:, top_indices]
    print(f"  Pre-filtered to top {top_k} features by |Cohen's d|")
    print(f"  Max |d| = {np.abs(cohen_d[top_indices[0]]):.4f}")

    # Cross-validate C parameter
    best_c = None
    best_score = -1
    best_score_sd = 0.0
    for c in L1_C_VALUES:
        clf = LogisticRegression(
            penalty="l1", C=c, solver="saga", max_iter=5000,
            random_state=SEED, class_weight="balanced",
        )
        scores = cross_val_score(clf, X, labels, cv=CV_FOLDS, scoring="roc_auc")
        mean_auc = scores.mean()
        print(f"  C={c:.4f}: AUC={mean_auc:.4f} (+/- {scores.std():.4f})")
        if mean_auc > best_score:
            best_score = mean_auc
            best_score_sd = scores.std()
            best_c = c

    print(f"\n  Best C={best_c}, AUC={best_score:.4f} (SD={best_score_sd:.4f})")

    # Fit final model
    clf = LogisticRegression(
        penalty="l1", C=best_c, solver="saga", max_iter=10000,
        random_state=SEED, class_weight="balanced",
    )
    clf.fit(X, labels)

    # Extract h-neurons (positive coefficients → over-compliance)
    coefficients = clf.coef_[0]
    h_neuron_mask = coefficients > 0
    h_neuron_indices = np.where(h_neuron_mask)[0]

    medical_h_neurons = []
    for idx in h_neuron_indices:
        original_feature_idx = top_indices[idx]
        layer = original_feature_idx // NEURONS_PER_LAYER
        neuron = original_feature_idx % NEURONS_PER_LAYER
        medical_h_neurons.append({
            "layer": int(layer),
            "neuron": int(neuron),
            "coefficient": float(coefficients[idx]),
            "cohen_d": float(cohen_d[original_feature_idx]),
        })

    # Sort by coefficient magnitude
    medical_h_neurons.sort(key=lambda x: -x["coefficient"])

    print(f"\n  Medical h-neurons: {len(medical_h_neurons)}")
    print(f"  Percentage: {len(medical_h_neurons) / TOTAL_NEURONS * 100:.4f}%")

    # Layer distribution
    layer_counts = Counter(h["layer"] for h in medical_h_neurons)
    print(f"  Layer distribution: {dict(sorted(layer_counts.items()))}")

    return medical_h_neurons, best_c, best_score, best_score_sd, clf


# ---------------------------------------------------------------------------
# Comparison with TriviaQA h-neurons
# ---------------------------------------------------------------------------

def compare_with_triviaqa(medical_h_neurons):
    """Compute Jaccard overlap with TriviaQA h-neurons."""
    triviaqa_path = OUTPUT_DIR / "h_neurons.json"
    if not triviaqa_path.exists():
        print("  TriviaQA h_neurons.json not found — skipping comparison")
        return None

    with open(triviaqa_path) as f:
        triviaqa_data = json.load(f)
    triviaqa_neurons = triviaqa_data["h_neurons"]

    # Convert to sets of (layer, neuron) tuples
    med_set = set((h["layer"], h["neuron"]) for h in medical_h_neurons)
    tqa_set = set((h["layer"], h["neuron"]) for h in triviaqa_neurons)

    intersection = med_set & tqa_set
    union = med_set | tqa_set
    jaccard = len(intersection) / max(len(union), 1)

    print(f"\n  TriviaQA h-neurons: {len(tqa_set)}")
    print(f"  Medical h-neurons: {len(med_set)}")
    print(f"  Overlap: {len(intersection)}")
    print(f"  Jaccard index: {jaccard:.4f}")

    return {
        "n_triviaqa": len(tqa_set),
        "n_medical": len(med_set),
        "n_overlap": len(intersection),
        "jaccard": jaccard,
        "overlap_neurons": [{"layer": l, "neuron": n} for l, n in intersection],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import transformers

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already done
    h_neurons_path = OUTPUT_DIR / "medical_h_neurons.json"
    if h_neurons_path.exists():
        print(f"Medical h-neurons already exist at {h_neurons_path}. Skipping.")
        return

    # Load training data
    # Try local data dir first, then full path
    if TRAIN_DATA.exists():
        train_path = TRAIN_DATA
    else:
        train_path = Path("/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/"
                         "data_final_outcome_splits/combined_train.json")
    print(f"Loading training data from {train_path}...")
    with open(train_path) as f:
        training_cases = json.load(f)
    print(f"  {len(training_cases)} cases loaded")

    # Load fine-tuned model
    print(f"Loading fine-tuned model from {MODEL_DIR}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print("Model loaded.")

    # Volume mount path for direct saves (resilience against preemption)
    RESULTS_DIR = Path("/results") / OUTPUT_DIR

    def save_to_volume(local_path):
        """Copy a local output file directly to the Modal volume mount if available."""
        vol_path = RESULTS_DIR / local_path.name
        if RESULTS_DIR.parent.exists():
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            import shutil as _shutil
            _shutil.copy(str(local_path), str(vol_path))
            print(f"  Saved to volume: {vol_path}")

    # Step 10a: Generate probe responses
    probe_path = OUTPUT_DIR / "medical_probe_responses.json"
    partial_path = OUTPUT_DIR / "medical_probe_responses_partial.json"
    vol_partial_path = RESULTS_DIR / "medical_probe_responses_partial.json" if RESULTS_DIR.parent.exists() else None

    # Restore partial checkpoint from volume if local copy missing
    if not partial_path.exists() and vol_partial_path and Path(vol_partial_path).exists():
        import shutil as _shutil
        _shutil.copy(str(vol_partial_path), str(partial_path))
        with open(partial_path) as f:
            n_done = len(json.load(f))
        print(f"Restored partial checkpoint from volume: {n_done} cases")

    if probe_path.exists():
        print(f"Loading existing probe responses from {probe_path}")
        with open(probe_path) as f:
            probe_responses = json.load(f)
    else:
        probe_responses = generate_probe_responses(
            model, tokenizer, training_cases, device,
            partial_path=str(partial_path),
            volume_partial_path=str(vol_partial_path) if vol_partial_path else None,
        )
        with open(probe_path, "w") as f:
            json.dump(probe_responses, f, indent=2)
        print(f"Saved {len(probe_responses)} probe responses to {probe_path}")
        save_to_volume(probe_path)
        # Clean up partial checkpoint
        if partial_path.exists():
            partial_path.unlink()

    # Step 10b: Build probe dataset
    probe_dataset_path = OUTPUT_DIR / "medical_probe_dataset.json"
    if probe_dataset_path.exists():
        print(f"Loading existing probe dataset from {probe_dataset_path}")
        with open(probe_dataset_path) as f:
            probe_dataset = json.load(f)
    else:
        probe_dataset = build_probe_dataset(probe_responses)
        with open(probe_dataset_path, "w") as f:
            json.dump(probe_dataset, f, indent=2)
        save_to_volume(probe_dataset_path)

    n_appropriate = sum(1 for p in probe_dataset if p["probe_label"] == 0)
    n_overcompliant = sum(1 for p in probe_dataset if p["probe_label"] == 1)
    print(f"\nProbe dataset: {n_appropriate} appropriate + {n_overcompliant} over-compliant")

    if n_overcompliant < 10:
        print("WARNING: Very few over-compliant samples. Results may be unreliable.")
        print("Consider: model may be too good at triage, or temperature sampling insufficient.")

    # Step 10c: CETT extraction
    cett_path = OUTPUT_DIR / "medical_cett_features.npz"
    if cett_path.exists():
        print(f"Loading existing CETT features from {cett_path}")
        data = np.load(cett_path)
        features, labels = data["features"], data["labels"]
    else:
        features, labels = extract_cett_with_messages(
            model, tokenizer, probe_dataset, training_cases, device
        )
        np.savez_compressed(cett_path, features=features, labels=labels)
        print(f"Saved CETT features to {cett_path}")
        save_to_volume(cett_path)

    # Step 10d: Sparse probing
    medical_h_neurons, best_c, best_auc, best_auc_sd, clf = sparse_probing(features, labels)

    # Compare with TriviaQA h-neurons
    comparison = compare_with_triviaqa(medical_h_neurons)

    # Save results
    result = {
        "medical_h_neurons": medical_h_neurons,
        "n_h_neurons": len(medical_h_neurons),
        "pct_total": len(medical_h_neurons) / TOTAL_NEURONS * 100,
        "probe_auc": best_auc,
        "probe_auc_sd": best_auc_sd,
        "probe_c": best_c,
        "n_probe_samples": len(probe_dataset),
        "n_appropriate": n_appropriate,
        "n_overcompliant": n_overcompliant,
        "triviaqa_comparison": comparison,
        "layer_distribution": dict(Counter(h["layer"] for h in medical_h_neurons)),
    }
    with open(h_neurons_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved medical h-neurons to {h_neurons_path}")


if __name__ == "__main__":
    main()
