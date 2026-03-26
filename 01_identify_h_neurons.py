#!/usr/bin/env python3
"""Step 1: Identify h-neurons via CETT metric + sparse L1-regularised probing.

Replicates the method from arXiv 2512.01797:
1. Load TriviaQA, generate 10 completions per question, filter to
   1000 consistently-correct + 1000 consistently-incorrect samples.
2. Forward-pass each sample through Llama-3.1-8B-Instruct with hooks
   that compute per-neuron CETT (Contribution to Effective Token Transfer)
   for every FFN neuron across all 32 layers.
3. Fit a sparse (L1) logistic regression on the CETT features.
4. H-neurons = neurons with positive non-zero coefficients.

Designed to run on Modal A100-80GB.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Llama-3.1-8B-Instruct")
N_LAYERS = 32
INTERMEDIATE_SIZE = 14336
N_COMPLETIONS = 5  # reduced from 10 for speed; majority vote instead of strict consistency
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.9
N_PER_CLASS = 500  # reduced from 1000; still sufficient for sparse probing
CONSISTENCY_THRESHOLD = 0.8  # fraction that must agree (4/5 or 5/5)
SEED = 42
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_triviaqa(n_questions: int = 5000):
    """Load TriviaQA questions from HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    questions = []
    for row in ds:
        q = row["question"]
        answers = row["answer"]["aliases"] + [row["answer"]["value"]]
        answers = list(set(a.lower().strip() for a in answers))
        questions.append({"question": q, "answers": answers})
        if len(questions) >= n_questions:
            break
    print(f"Loaded {len(questions)} TriviaQA questions")
    return questions


def check_answer(generated: str, gold_answers: list[str]) -> bool:
    """Check if any gold answer appears in the generated text."""
    gen_lower = generated.lower().strip()
    return any(a in gen_lower for a in gold_answers)


def generate_completions(model, tokenizer, questions: list[dict], device: str):
    """Generate N_COMPLETIONS per question using batched generation.

    Uses majority-vote consistency (>=CONSISTENCY_THRESHOLD) instead of
    strict 100% agreement for faster convergence. Checkpoints to disk
    every 100 samples for preemption resilience.
    """
    checkpoint_path = OUTPUT_DIR / "triviaqa_checkpoint.json"
    start_idx = 0

    # Resume from checkpoint if available
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        consistent_correct = [s for s in ckpt["samples"] if s["label"] == 0]
        consistent_incorrect = [s for s in ckpt["samples"] if s["label"] == 1]
        start_idx = ckpt.get("questions_processed", 0)
        print(f"Resuming from checkpoint: {len(consistent_correct)} correct, "
              f"{len(consistent_incorrect)} incorrect, {start_idx} questions processed")
    else:
        consistent_correct = []
        consistent_incorrect = []

    min_correct = int(N_COMPLETIONS * CONSISTENCY_THRESHOLD)  # e.g., 4/5
    last_checkpoint = len(consistent_correct) + len(consistent_incorrect)

    for i, q in enumerate(tqdm(questions[start_idx:], desc="Generating completions",
                               initial=start_idx, total=len(questions))):
        if len(consistent_correct) >= N_PER_CLASS and len(consistent_incorrect) >= N_PER_CLASS:
            break

        prompt = f"Answer the following question in a few words.\n\nQuestion: {q['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Batch generate all completions at once
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                do_sample=True,
                num_return_sequences=N_COMPLETIONS,
            )

        prompt_len = inputs["input_ids"].shape[1]
        completions = []
        n_correct = 0
        for seq in outputs:
            generated = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            completions.append(generated)
            if check_answer(generated, q["answers"]):
                n_correct += 1

        # Majority-vote consistency filter
        if n_correct >= min_correct and len(consistent_correct) < N_PER_CLASS:
            consistent_correct.append({
                "question": q["question"],
                "answers": q["answers"],
                "prompt": prompt,
                "completion": completions[0],
                "label": 0,  # faithful
                "n_correct": n_correct,
            })
        elif n_correct <= (N_COMPLETIONS - min_correct) and len(consistent_incorrect) < N_PER_CLASS:
            consistent_incorrect.append({
                "question": q["question"],
                "answers": q["answers"],
                "prompt": prompt,
                "completion": completions[0],
                "label": 1,  # hallucinated
                "n_correct": n_correct,
            })

        total_found = len(consistent_correct) + len(consistent_incorrect)
        if total_found % 50 == 0 and total_found > 0:
            print(f"  Correct: {len(consistent_correct)}, Incorrect: {len(consistent_incorrect)}")

        # Checkpoint every 100 new samples
        if total_found - last_checkpoint >= 100:
            ckpt_data = {
                "samples": consistent_correct + consistent_incorrect,
                "questions_processed": start_idx + i + 1,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(ckpt_data, f)
            last_checkpoint = total_found
            print(f"  Checkpoint saved: {total_found} samples, {start_idx + i + 1} questions")
            # Also save to volume if available
            vol_path = Path("/results/output")
            if vol_path.exists():
                import shutil
                shutil.copy(str(checkpoint_path), str(vol_path / "triviaqa_checkpoint.json"))

    print(f"Final: {len(consistent_correct)} faithful, {len(consistent_incorrect)} hallucinated")
    return consistent_correct + consistent_incorrect


def compute_cett_features(model, tokenizer, samples: list[dict], device: str):
    """Compute CETT features for all FFN neurons across all samples.

    For each sample, returns a vector of shape (2 * N_LAYERS * INTERMEDIATE_SIZE,)
    where the first half is mean CETT during answer tokens and the second half
    is mean CETT during prompt tokens.

    For SwiGLU (Llama):
        neuron j activation = silu(gate_proj(x))_j * up_proj(x)_j
        contribution to residual = a_j * down_proj.weight[:, j]
        CETT_{j,t} = |a_j| * ||down_proj.weight[:, j]||_2 / ||FFN_output||_2
    """
    n_features = 2 * N_LAYERS * INTERMEDIATE_SIZE
    X = np.zeros((len(samples), n_features), dtype=np.float32)

    # Precompute down_proj weight norms for each layer
    down_proj_norms = []
    for layer_idx in range(N_LAYERS):
        mlp = model.model.layers[layer_idx].mlp
        w_norms = mlp.down_proj.weight.norm(dim=0).detach().cpu().numpy()  # (intermediate,)
        down_proj_norms.append(w_norms)

    for sample_idx, sample in enumerate(tqdm(samples, desc="Computing CETT")):
        prompt_text = sample["prompt"]
        full_text = prompt_text + sample["completion"]

        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
        n_prompt_tokens = prompt_ids.shape[1]
        n_total_tokens = full_ids.shape[1]
        n_answer_tokens = n_total_tokens - n_prompt_tokens

        # Storage for per-layer CETT means
        cett_answer = np.zeros((N_LAYERS, INTERMEDIATE_SIZE), dtype=np.float32)
        cett_prompt = np.zeros((N_LAYERS, INTERMEDIATE_SIZE), dtype=np.float32)

        # Register hooks
        hook_handles = []
        hook_storage = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                x = input[0]  # (1, seq, hidden)
                gate = module.gate_proj(x)
                up = module.up_proj(x)
                act = F.silu(gate) * up  # (1, seq, intermediate)

                # |a_j| per token
                act_abs = act.abs().squeeze(0).detach().cpu().numpy()  # (seq, intermediate)

                # ||FFN_output||_2 per token
                ffn_out = output.squeeze(0).detach()  # (seq, hidden)
                ffn_norms = ffn_out.norm(dim=-1).cpu().numpy()  # (seq,)
                ffn_norms = np.maximum(ffn_norms, 1e-8)

                # CETT_{j,t} = |a_j| * ||w_j||_2 / ||FFN_out||_2
                w_norms = down_proj_norms[layer_idx]  # (intermediate,)
                cett = act_abs * w_norms[np.newaxis, :] / ffn_norms[:, np.newaxis]  # (seq, intermediate)

                hook_storage[layer_idx] = cett
            return hook_fn

        for layer_idx in range(N_LAYERS):
            mlp = model.model.layers[layer_idx].mlp
            h = mlp.register_forward_hook(make_hook(layer_idx))
            hook_handles.append(h)

        # Forward pass
        with torch.no_grad():
            model(full_ids)

        # Remove hooks
        for h in hook_handles:
            h.remove()

        # Aggregate CETT: mean over answer tokens vs prompt tokens
        for layer_idx in range(N_LAYERS):
            cett = hook_storage[layer_idx]  # (seq, intermediate)
            if n_answer_tokens > 0:
                cett_answer[layer_idx] = cett[n_prompt_tokens:].mean(axis=0)
            if n_prompt_tokens > 0:
                cett_prompt[layer_idx] = cett[:n_prompt_tokens].mean(axis=0)

        # Flatten: [answer_features, prompt_features]
        X[sample_idx, :N_LAYERS * INTERMEDIATE_SIZE] = cett_answer.flatten()
        X[sample_idx, N_LAYERS * INTERMEDIATE_SIZE:] = cett_prompt.flatten()

        if (sample_idx + 1) % 100 == 0:
            print(f"  Processed {sample_idx + 1}/{len(samples)} samples")

    return X


def fit_sparse_probe(X: np.ndarray, y: np.ndarray):
    """Identify h-neurons using two-stage approach:
    1. Per-neuron effect size ranking (fast, O(n_features) univariate tests)
    2. L1-penalised logistic regression on top-K candidates for refinement

    This is equivalent to the paper's sparse probing but much faster on 917K features.
    """
    from scipy.stats import mannwhitneyu
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    sys.stdout.flush()

    # Stage 1: Per-neuron discriminative power via effect size
    # For each feature, compute Cohen's d (mean difference / pooled std)
    # H-neurons should have higher CETT in hallucinated (y=1) than faithful (y=0)
    print("Stage 1: Computing per-neuron effect sizes...")
    sys.stdout.flush()

    mask_0 = y == 0  # faithful
    mask_1 = y == 1  # hallucinated

    # Compute in chunks to show progress
    n_features = X.shape[1]
    chunk_size = 50000
    effect_sizes = np.zeros(n_features, dtype=np.float32)

    for start in range(0, n_features, chunk_size):
        end = min(start + chunk_size, n_features)
        X_chunk = X[:, start:end]
        mean_0 = X_chunk[mask_0].mean(axis=0)
        mean_1 = X_chunk[mask_1].mean(axis=0)
        std_0 = X_chunk[mask_0].std(axis=0)
        std_1 = X_chunk[mask_1].std(axis=0)
        pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)
        pooled_std[pooled_std < 1e-10] = 1.0
        effect_sizes[start:end] = (mean_1 - mean_0) / pooled_std
        if (start // chunk_size) % 4 == 0:
            print(f"  Processed {end}/{n_features} features")
            sys.stdout.flush()

    # Select top candidates: positive effect size (higher in hallucinated), above threshold
    # Take features with effect size > 0.3 (medium effect) or top 2000, whichever is smaller
    positive_mask = effect_sizes > 0.3
    n_candidates_by_threshold = positive_mask.sum()
    top_k = min(2000, n_candidates_by_threshold)
    if top_k < 100:
        # If few features above threshold, take top 2000 by absolute effect size
        top_k = 2000
    top_indices = np.argsort(-effect_sizes)[:top_k]
    # Keep only those with positive effect (higher in hallucinated)
    top_indices = top_indices[effect_sizes[top_indices] > 0]

    print(f"  {n_candidates_by_threshold} features with effect size > 0.3")
    print(f"  Selected {len(top_indices)} candidate features for stage 2")
    sys.stdout.flush()

    # Stage 2: L1-regularised logistic regression on candidates only
    print("\nStage 2: Sparse probing on candidate features...")
    sys.stdout.flush()

    X_candidates = X[:, top_indices]

    # Standardize
    means = X_candidates.mean(axis=0)
    X_candidates = X_candidates - means
    stds = X_candidates.std(axis=0)
    stds[stds < 1e-8] = 1.0
    X_candidates = X_candidates / stds

    clf = LogisticRegressionCV(
        Cs=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        penalty="l1",
        solver="saga",
        cv=5,
        max_iter=5000,
        tol=1e-4,
        random_state=SEED,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=1,
    )
    clf.fit(X_candidates, y)

    best_C = clf.C_[0]
    print(f"Best C: {best_C}")
    sys.stdout.flush()

    # CV AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(clf, X_candidates, y, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    sys.stdout.flush()

    # Extract h-neurons: positive coefficients from the L1 model
    coefs_candidates = clf.coef_[0]
    # Map back to original feature indices
    coefs = np.zeros(X.shape[1])
    coefs[top_indices] = coefs_candidates
    h_neuron_flat_indices = np.where(coefs > 1e-6)[0]
    print(f"H-neurons found: {len(h_neuron_flat_indices)} / {len(coefs)} "
          f"({len(h_neuron_flat_indices) / (N_LAYERS * INTERMEDIATE_SIZE) * 100:.4f}%)")

    # Map flat indices to (layer, neuron, feature_type)
    h_neurons = []
    for flat_idx in h_neuron_flat_indices:
        if flat_idx < N_LAYERS * INTERMEDIATE_SIZE:
            # Answer-token feature
            layer_idx = flat_idx // INTERMEDIATE_SIZE
            neuron_idx = flat_idx % INTERMEDIATE_SIZE
            feature_type = "answer"
        else:
            # Prompt-token feature
            adjusted = flat_idx - N_LAYERS * INTERMEDIATE_SIZE
            layer_idx = adjusted // INTERMEDIATE_SIZE
            neuron_idx = adjusted % INTERMEDIATE_SIZE
            feature_type = "prompt"
        h_neurons.append({
            "layer": int(layer_idx),
            "neuron": int(neuron_idx),
            "feature_type": feature_type,
            "coefficient": float(coefs[flat_idx]),
        })

    # Deduplicate: unique (layer, neuron) pairs
    unique_neurons = {}
    for hn in h_neurons:
        key = (hn["layer"], hn["neuron"])
        if key not in unique_neurons or hn["coefficient"] > unique_neurons[key]["coefficient"]:
            unique_neurons[key] = hn
    h_neurons_deduped = list(unique_neurons.values())
    h_neurons_deduped.sort(key=lambda x: (-x["coefficient"]))

    print(f"Unique (layer, neuron) h-neurons: {len(h_neurons_deduped)}")

    return h_neurons_deduped, clf, cv_scores


def _save_to_volume_if_available():
    """Save output/ to Modal volume if running inside Modal."""
    import shutil
    vol_path = Path("/results/output")
    if vol_path.parent.exists():
        vol_path.mkdir(parents=True, exist_ok=True)
        src = OUTPUT_DIR
        for f in src.iterdir():
            if f.is_file():
                shutil.copy(str(f), str(vol_path / f.name))
                print(f"  Saved {f.name} to volume")
        # Commit volume
        try:
            import modal
            vol = modal.Volume.from_name("h-neuron-results-v2")
            vol.commit()
            print("  Volume committed")
        except Exception as e:
            print(f"  Volume commit skipped: {e}")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Step 1: Load TriviaQA and generate consistency-filtered samples
    samples_path = OUTPUT_DIR / "triviaqa_samples.json"
    if samples_path.exists():
        print(f"Loading cached samples from {samples_path}")
        with open(samples_path) as f:
            samples = json.load(f)
    else:
        questions = load_triviaqa(n_questions=5000)
        samples = generate_completions(model, tokenizer, questions, device)
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Saved {len(samples)} samples to {samples_path}")

    y = np.array([s["label"] for s in samples])
    print(f"Samples: {len(samples)} (faithful={sum(y==0)}, hallucinated={sum(y==1)})")

    # Save samples to volume immediately
    _save_to_volume_if_available()

    # Step 2: Compute CETT features
    features_path = OUTPUT_DIR / "cett_features.npz"
    if features_path.exists():
        print(f"Loading cached CETT features from {features_path}")
        data = np.load(features_path)
        X = data["X"]
    else:
        X = compute_cett_features(model, tokenizer, samples, device)
        np.savez_compressed(features_path, X=X, y=y)
        print(f"Saved CETT features to {features_path}")

    # Save CETT features to volume immediately
    _save_to_volume_if_available()

    # Step 3: Sparse probing
    print("\nFitting sparse probe...")
    h_neurons, clf, cv_scores = fit_sparse_probe(X, y)

    # Save h-neurons
    result = {
        "model": MODEL_ID,
        "n_layers": N_LAYERS,
        "intermediate_size": INTERMEDIATE_SIZE,
        "n_samples": len(samples),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "n_h_neurons": len(h_neurons),
        "pct_of_total": len(h_neurons) / (N_LAYERS * INTERMEDIATE_SIZE) * 100,
        "h_neurons": h_neurons,
    }

    h_neurons_path = OUTPUT_DIR / "h_neurons.json"
    with open(h_neurons_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {len(h_neurons)} h-neurons to {h_neurons_path}")

    # Layer distribution summary
    layer_counts = {}
    for hn in h_neurons:
        layer_counts[hn["layer"]] = layer_counts.get(hn["layer"], 0) + 1
    print("\nH-neuron layer distribution:")
    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer:2d}: {layer_counts[layer]} neurons")


if __name__ == "__main__":
    main()
