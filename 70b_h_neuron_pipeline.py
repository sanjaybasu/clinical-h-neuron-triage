#!/usr/bin/env python3
"""Modal pipeline for h-neuron identification and ablation on Llama-3.1-70B-Instruct.

Three independently spawnable functions:
  1. run_70b_phase1  -- CETT + sparse probing h-neuron identification (70B)
  2. run_70b_ablation -- Ablation sweep on physician test set (8 alpha levels)
  3. run_70b_logit_lens -- Logit lens analysis on triage cases

Usage:
    # Deploy the app:
    modal deploy 70b_h_neuron_pipeline.py

    # Spawn each phase independently:
    python -c "
        import modal
        fn = modal.Function.from_name('h-neuron-70b-pipeline', 'run_70b_phase1')
        fn.spawn()
    "
    python -c "
        import modal
        fn = modal.Function.from_name('h-neuron-70b-pipeline', 'run_70b_ablation')
        fn.spawn()
    "
    python -c "
        import modal
        fn = modal.Function.from_name('h-neuron-70b-pipeline', 'run_70b_logit_lens')
        fn.spawn()
    "

    # Or run directly:
    modal run 70b_h_neuron_pipeline.py::run_70b_phase1
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app + resources
# ---------------------------------------------------------------------------
app = modal.App("h-neuron-70b-pipeline")

results_volume = modal.Volume.from_name("h-neuron-results-v3", create_if_missing=True)
model_volume = modal.Volume.from_name("llama-3.1-70b-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "tqdm>=4.65.0",
        "transformers>=4.38.0",
        "accelerate>=0.25.0",
        "huggingface-hub>=0.20.0",
        "safetensors>=0.4.0",
        "datasets>=2.14.0",
    )
    .add_local_dir(".", "/app", ignore=[".venv", "__pycache__", "*.pyc", ".git"], copy=True)
)

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
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

ALPHA_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MAX_NEW_TOKENS = 256
SEED = 42

# CETT identification parameters
N_COMPLETIONS = 5
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.9
N_PER_CLASS = 200
CONSISTENCY_THRESHOLD = 0.8

# Logit lens token groups
SAFETY_TOKEN_GROUPS = {
    "emergency": ["emergency", "911", "ambulance", "ER", "immediately", "life"],
    "urgent": ["urgent", "doctor", "prescriber", "provider", "soon", "today"],
    "routine": ["routine", "follow", "monitor", "check", "schedule", "appointment"],
    "reassure": ["reassure", "unlikely", "normal", "benign", "fine", "okay", "safe"],
    "hedging": ["however", "although", "but", "might", "could", "possibly", "uncertain"],
}
ANALYSIS_ALPHAS = [0.5, 1.0, 2.0]
MAX_LOGIT_LENS_CASES = 50


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _save_to_volume():
    """Copy output/ to the results volume."""
    for dirname in ["output"]:
        src = Path(dirname)
        if src.exists():
            dest = Path(f"/results/{dirname}")
            shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
            n = len(list(src.iterdir()))
            print(f"  Saved {dirname}/: {n} files -> volume")
    results_volume.commit()


def _restore_from_volume(filenames: list[str]):
    """Copy specific files from volume to local output/."""
    os.makedirs("output", exist_ok=True)
    for fname in filenames:
        vol_src = Path(f"/results/output/{fname}")
        local_dst = Path(f"output/{fname}")
        if vol_src.exists() and not local_dst.exists():
            shutil.copy(str(vol_src), str(local_dst))
            print(f"Restored {fname} from volume")


def _load_model():
    """Load Llama-3.1-70B-Instruct in bfloat16 with device_map=auto."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID} in bfloat16 across available GPUs...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    n_gpus = len(set(str(p.device) for p in model.parameters()))
    n_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size
    print(f"Model loaded in {time.time() - t0:.1f}s across {n_gpus} GPU(s)")
    print(f"Architecture: {n_layers} layers, intermediate_size={intermediate_size}")

    return model, tokenizer


def build_chat_prompt(tokenizer, message: str) -> str:
    """Build prompt using Llama-3.1 chat template."""
    system = SYSTEM_PROMPT + SAFETY_HINT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Patient message: {message}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Robust response parser for 70B (handles structured clinical assessments)
# ---------------------------------------------------------------------------
def parse_response_70b(text: str) -> dict:
    """Parse a 70B triage response into detection/severity/action.

    The 70B model produces more structured, verbose responses than the 8B.
    It often starts with phrases like "Clinical Hazard Identified" or
    "This message contains/does not contain a clinical hazard". This parser
    handles those patterns as well as keyword fallback.
    """
    text_lower = text.lower().strip()

    # --- Structured detection markers (primary) ---
    # Positive markers
    positive_markers = [
        "clinical hazard identified",
        "hazard identified",
        "hazard detected",
        "this message contains a clinical hazard",
        "this message does contain a clinical hazard",
        "yes, this message contains",
        "clinical hazard: yes",
        "hazard: yes",
        "assessment: clinical hazard",
        "i have identified a clinical hazard",
        "there is a clinical hazard",
        "this is a clinical hazard",
        "this constitutes a clinical hazard",
        "requires immediate",
        "requires urgent",
    ]

    # Negative markers
    negative_markers = [
        "no clinical hazard",
        "no hazard detected",
        "no hazard identified",
        "this message does not contain a clinical hazard",
        "this message does not contain any clinical hazard",
        "clinical hazard: no",
        "hazard: no",
        "no immediate clinical hazard",
        "does not appear to contain a clinical hazard",
        "i do not identify a clinical hazard",
        "there is no clinical hazard",
        "no urgent clinical hazard",
    ]

    # Check negative first (more specific)
    if any(marker in text_lower for marker in negative_markers):
        return {"detection": 0, "severity": "Benign", "action": "No Action/Self-care"}

    # Check positive structured markers
    if any(marker in text_lower for marker in positive_markers):
        return _classify_action_70b(text_lower)

    # --- Keyword fallback (secondary) ---
    emergency_keywords = [
        "call 911", "call 988", "emergency room", "emergency department",
        "ambulance", "go to the er", "go to the ed", "call emergency",
        "seek emergency", "life-threatening", "immediate danger",
        "immediately go to", "rush to the hospital",
    ]
    urgent_keywords = [
        "urgent care", "call doctor", "call your doctor", "contact prescriber",
        "contact your provider", "seen urgently", "seen today", "same day",
        "same-day", "within 24 hours",
    ]

    if any(kw in text_lower for kw in emergency_keywords):
        return {"detection": 1, "severity": "Emergent", "action": "Call 911/ED"}

    if any(kw in text_lower for kw in urgent_keywords):
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}

    # Broader catch phrases
    if any(phrase in text_lower for phrase in [
        "requires immediate", "seek immediate", "requires emergency",
        "life-threatening", "immediate medical", "immediately go",
        "go to the nearest", "emergency evaluation",
    ]):
        return {"detection": 1, "severity": "Emergent", "action": "Call 911/ED"}

    if any(phrase in text_lower for phrase in [
        "clinical concern", "medical attention", "requires attention",
        "should be evaluated", "see your doctor", "see a doctor",
        "contact your", "consult your", "do not wait",
        "recommend evaluation", "should seek", "needs to be seen",
    ]):
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}

    if any(phrase in text_lower for phrase in [
        "follow up", "follow-up", "schedule an appointment",
        "routine visit", "routine check", "at your next",
    ]):
        return {"detection": 1, "severity": "Routine", "action": "Routine Follow-up"}

    return {"detection": 0, "severity": "Benign", "action": "No Action/Self-care"}


def _classify_action_70b(text_lower: str) -> dict:
    """Given that a hazard was detected, classify the recommended action."""
    # Emergency level
    if any(kw in text_lower for kw in [
        "call 911", "call 988", "emergency room", "emergency department",
        "ambulance", "go to the er", "go to the ed", "life-threatening",
        "immediate danger", "immediately go to", "rush to the hospital",
        "call emergency", "seek emergency",
    ]):
        return {"detection": 1, "severity": "Emergent", "action": "Call 911/ED"}

    # Urgent / contact doctor
    if any(kw in text_lower for kw in [
        "contact your doctor", "call your doctor", "contact your provider",
        "call your provider", "contact prescriber", "see your doctor",
        "urgent", "today", "right away", "within 24",
        "medical attention", "seen today", "same-day", "same day",
    ]):
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}

    # Routine follow-up
    if any(kw in text_lower for kw in [
        "routine", "follow-up", "follow up", "schedule",
        "appointment", "at your next visit", "monitor",
    ]):
        return {"detection": 1, "severity": "Routine", "action": "Routine Follow-up"}

    # Default positive: if hazard detected but no clear action
    return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}


def _map_action_to_category(action_text: str) -> str:
    """Map an action description to one of the 4 canonical categories."""
    if not action_text:
        return "No Action/Self-care"
    text = action_text.lower().strip()
    if any(kw in text for kw in ["911", "ed ", "ed/", "emergency", "esi-1", "esi-2"]):
        return "Call 911/ED"
    if any(kw in text for kw in ["doctor", "provider", "prescriber", "urgent", "esi-3"]):
        return "Contact Doctor"
    if any(kw in text for kw in ["routine", "follow", "esi-4", "esi-5"]):
        return "Routine Follow-up"
    if any(kw in text for kw in ["none", "self-care", "no action", "reassur", "benign"]):
        return "No Action/Self-care"
    # Fallback: if it mentions a specific clinical action, treat as Contact Doctor
    if len(text) > 5:
        return "Contact Doctor"
    return "No Action/Self-care"


# ---------------------------------------------------------------------------
# H-Neuron Ablator (works across multi-GPU device_map)
# ---------------------------------------------------------------------------
class HNeuronAblator70B:
    """Monkey-patches LlamaMLP.forward to scale h-neuron activations by alpha.

    Handles the multi-GPU case where different layers live on different devices.
    """

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
        for layer in self.layer_neurons:
            self.layer_neurons[layer] = sorted(set(self.layer_neurons[layer]))
        total = sum(len(v) for v in self.layer_neurons.values())
        print(f"H-neurons span {len(self.layer_neurons)} layers, {total} unique neurons")

    def apply(self, alpha: float):
        """Patch MLP forwards to scale h-neurons by alpha."""
        import torch
        from torch.nn import functional as F

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
                    device = act.device
                    idx = idx_tensor.to(device)
                    # Cast to float32 before scaling to avoid bfloat16 overflow at high alpha
                    orig_dtype = act.dtype
                    act_f32 = act.to(torch.float32)
                    act_f32[:, :, idx] = act_f32[:, :, idx] * a
                    act = act_f32.to(orig_dtype)
                    return orig_mlp.down_proj(act)
                return patched_forward

            mlp.forward = make_patched_forward(mlp, neuron_idx_tensor, alpha)

    def restore(self):
        """Restore original MLP forwards."""
        for layer_idx, orig_forward in self._original_forwards.items():
            self.model.model.layers[layer_idx].mlp.forward = orig_forward
        self._original_forwards.clear()


# ===========================================================================
# Phase 1: H-Neuron Identification
# ===========================================================================
@app.function(
    image=image,
    gpu="A100-80GB:2",
    secrets=[hf_secret],
    timeout=43200,  # 12 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=196608,
)
def run_70b_phase1():
    """Identify h-neurons on Llama-3.1-70B-Instruct via CETT + sparse probing.

    Same methodology as the 8B pipeline (01_identify_h_neurons.py):
      1. Generate TriviaQA completions, filter by consistency
      2. Compute per-neuron CETT features via forward hooks
      3. Two-stage sparse probing: univariate ranking + L1 logistic regression
      4. Output: 70b_h_neurons.json
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tqdm import tqdm

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Restore any checkpoints from volume (preemption resilience)
    _restore_from_volume([
        "70b_triviaqa_checkpoint.json",
        "70b_triviaqa_samples.json",
        "70b_cett_features.npz",
        "70b_cett_features_partial.npz",
        "70b_h_neurons.json",
    ])

    # Idempotent: skip if already done
    if Path("output/70b_h_neurons.json").exists():
        print("70b_h_neurons.json already exists. Skipping phase 1.")
        return

    print("=" * 60)
    print("70B PHASE 1: H-Neuron Identification via CETT + Sparse Probing")
    print("=" * 60)
    sys.stdout.flush()

    # Load model
    model, tokenizer = _load_model()
    n_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size
    print(f"Will probe {n_layers} layers x {intermediate_size} neurons "
          f"= {n_layers * intermediate_size:,} total FFN neurons")
    sys.stdout.flush()

    # ---- Step 1: TriviaQA consistency-filtered samples ----
    samples_path = OUTPUT_DIR / "70b_triviaqa_samples.json"
    if samples_path.exists():
        print(f"Loading cached samples from {samples_path}")
        with open(samples_path) as f:
            samples = json.load(f)
    else:
        samples = _generate_triviaqa_samples_70b(model, tokenizer, OUTPUT_DIR)
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Saved {len(samples)} samples to {samples_path}")
        _save_to_volume()

    y = np.array([s["label"] for s in samples])
    print(f"Samples: {len(samples)} (faithful={int((y==0).sum())}, hallucinated={int((y==1).sum())})")
    sys.stdout.flush()

    # ---- Step 2: Compute CETT features ----
    features_path = OUTPUT_DIR / "70b_cett_features.npz"
    if features_path.exists():
        print(f"Loading cached CETT features from {features_path}")
        data = np.load(features_path)
        X = data["X"]
    else:
        X = _compute_cett_features_70b(model, tokenizer, samples, n_layers, intermediate_size)
        np.savez_compressed(features_path, X=X, y=y)
        print(f"Saved CETT features to {features_path} (shape: {X.shape})")
        _save_to_volume()

    # ---- Step 3: Sparse probing ----
    print("\nFitting sparse probe...")
    sys.stdout.flush()
    h_neurons, cv_scores = _fit_sparse_probe_70b(X, y, n_layers, intermediate_size)

    # Save results
    result = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "intermediate_size": intermediate_size,
        "n_samples": len(samples),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "n_h_neurons": len(h_neurons),
        "pct_of_total": len(h_neurons) / (n_layers * intermediate_size) * 100,
        "h_neurons": h_neurons,
    }

    h_neurons_path = OUTPUT_DIR / "70b_h_neurons.json"
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

    _save_to_volume()
    model_volume.commit()
    print("Phase 1 complete.")


def _generate_triviaqa_samples_70b(model, tokenizer, output_dir: Path) -> list[dict]:
    """Generate TriviaQA completions and filter by consistency."""
    import torch
    from datasets import load_dataset
    from tqdm import tqdm

    print("Loading TriviaQA validation set...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    questions = []
    for row in ds:
        q = row["question"]
        answers = row["answer"]["aliases"] + [row["answer"]["value"]]
        answers = list(set(a.lower().strip() for a in answers))
        questions.append({"question": q, "answers": answers})
        if len(questions) >= 2000:
            break
    print(f"Loaded {len(questions)} TriviaQA questions")

    def check_answer(generated: str, gold_answers: list[str]) -> bool:
        gen_lower = generated.lower().strip()
        return any(a in gen_lower for a in gold_answers)

    # Resume from checkpoint
    checkpoint_path = output_dir / "70b_triviaqa_checkpoint.json"
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        consistent_correct = [s for s in ckpt["samples"] if s["label"] == 0]
        consistent_incorrect = [s for s in ckpt["samples"] if s["label"] == 1]
        start_idx = ckpt.get("questions_processed", 0)
        print(f"Resuming: {len(consistent_correct)} correct, "
              f"{len(consistent_incorrect)} incorrect, {start_idx} questions processed")
    else:
        consistent_correct = []
        consistent_incorrect = []

    min_correct = int(N_COMPLETIONS * CONSISTENCY_THRESHOLD)
    last_checkpoint = len(consistent_correct) + len(consistent_incorrect)

    # Determine the device for the first embedding layer (for multi-GPU)
    first_device = next(model.parameters()).device

    for i, q in enumerate(tqdm(questions[start_idx:], desc="Generating completions",
                               initial=start_idx, total=len(questions))):
        if len(consistent_correct) >= N_PER_CLASS and len(consistent_incorrect) >= N_PER_CLASS:
            break

        prompt = f"Answer the following question in a few words.\n\nQuestion: {q['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(first_device)

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

        if n_correct >= min_correct and len(consistent_correct) < N_PER_CLASS:
            consistent_correct.append({
                "question": q["question"],
                "answers": q["answers"],
                "prompt": prompt,
                "completion": completions[0],
                "label": 0,
                "n_correct": n_correct,
            })
        elif n_correct <= (N_COMPLETIONS - min_correct) and len(consistent_incorrect) < N_PER_CLASS:
            consistent_incorrect.append({
                "question": q["question"],
                "answers": q["answers"],
                "prompt": prompt,
                "completion": completions[0],
                "label": 1,
                "n_correct": n_correct,
            })

        total_found = len(consistent_correct) + len(consistent_incorrect)
        if total_found % 50 == 0 and total_found > 0:
            print(f"  Correct: {len(consistent_correct)}, Incorrect: {len(consistent_incorrect)}")
            sys.stdout.flush()

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
            sys.stdout.flush()
            # Also save to volume for preemption resilience
            try:
                _save_to_volume()
            except Exception as e:
                print(f"  Volume checkpoint warning: {e}")

    print(f"Final: {len(consistent_correct)} faithful, {len(consistent_incorrect)} hallucinated")
    return consistent_correct + consistent_incorrect


def _compute_cett_features_70b(model, tokenizer, samples: list[dict],
                                n_layers: int, intermediate_size: int):
    """Compute CETT features for all FFN neurons across all samples.

    Same algorithm as 8B but with architecture-aware dimensions (read from model.config).

    For SwiGLU (Llama):
        neuron j activation = silu(gate_proj(x))_j * up_proj(x)_j
        contribution to residual = a_j * down_proj.weight[:, j]
        CETT_{j,t} = |a_j| * ||down_proj.weight[:, j]||_2 / ||FFN_output||_2
    """
    import numpy as np
    import torch
    from tqdm import tqdm

    n_features = 2 * n_layers * intermediate_size
    X = np.zeros((len(samples), n_features), dtype=np.float32)

    # Resume from partial checkpoint if available
    start_sample_idx = 0
    partial_path = Path("output/70b_cett_features_partial.npz")
    if partial_path.exists():
        try:
            partial_data = np.load(partial_path)
            n_done = int(partial_data.get("completed", 0))
            if n_done > 0 and n_done <= len(samples):
                X[:n_done] = partial_data["X"][:n_done]
                start_sample_idx = n_done
                print(f"Resumed CETT from partial checkpoint: {n_done}/{len(samples)} samples already done")
                sys.stdout.flush()
        except Exception as e:
            print(f"Could not load partial CETT checkpoint: {e}. Starting from scratch.")


    # Precompute down_proj weight norms for each layer
    print("Precomputing down_proj weight norms...")
    down_proj_norms = []
    for layer_idx in range(n_layers):
        mlp = model.model.layers[layer_idx].mlp
        # Compute on the device where this layer lives
        w_norms = mlp.down_proj.weight.norm(dim=0).detach().float().cpu().numpy()
        down_proj_norms.append(w_norms)
        if (layer_idx + 1) % 20 == 0:
            print(f"  Layer {layer_idx + 1}/{n_layers} norms computed")
    print(f"Down_proj norms computed for {n_layers} layers")
    sys.stdout.flush()

    first_device = next(model.parameters()).device

    for sample_idx, sample in enumerate(tqdm(samples[start_sample_idx:], desc="Computing CETT",
                                              initial=start_sample_idx, total=len(samples))):
        sample_idx = sample_idx + start_sample_idx  # Adjust for resume offset
        prompt_text = sample["prompt"]
        full_text = prompt_text + sample["completion"]

        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(first_device)
        n_prompt_tokens = prompt_ids.shape[1]
        n_total_tokens = full_ids.shape[1]
        n_answer_tokens = n_total_tokens - n_prompt_tokens

        # Storage for per-layer CETT means
        cett_answer = np.zeros((n_layers, intermediate_size), dtype=np.float32)
        cett_prompt = np.zeros((n_layers, intermediate_size), dtype=np.float32)

        # Register hooks
        hook_handles = []
        hook_storage = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                x = input[0]  # (1, seq, hidden)
                gate = module.gate_proj(x)
                up = module.up_proj(x)
                act = torch.nn.functional.silu(gate) * up  # (1, seq, intermediate)

                # |a_j| per token
                act_abs = act.abs().squeeze(0).detach().float().cpu().numpy()

                # ||FFN_output||_2 per token
                ffn_out = output.squeeze(0).detach().float()
                ffn_norms = ffn_out.norm(dim=-1).cpu().numpy()
                ffn_norms = np.maximum(ffn_norms, 1e-8)

                # CETT_{j,t} = |a_j| * ||w_j||_2 / ||FFN_out||_2
                w_norms = down_proj_norms[layer_idx]
                cett = act_abs * w_norms[np.newaxis, :] / ffn_norms[:, np.newaxis]

                hook_storage[layer_idx] = cett
            return hook_fn

        for layer_idx in range(n_layers):
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
        for layer_idx in range(n_layers):
            cett = hook_storage[layer_idx]
            if n_answer_tokens > 0:
                cett_answer[layer_idx] = cett[n_prompt_tokens:].mean(axis=0)
            if n_prompt_tokens > 0:
                cett_prompt[layer_idx] = cett[:n_prompt_tokens].mean(axis=0)

        # Flatten: [answer_features, prompt_features]
        X[sample_idx, :n_layers * intermediate_size] = cett_answer.flatten()
        X[sample_idx, n_layers * intermediate_size:] = cett_prompt.flatten()

        if (sample_idx + 1) % 25 == 0:
            print(f"  Processed {sample_idx + 1}/{len(samples)} samples")
            sys.stdout.flush()

        # Periodic save of partial features (every 50 samples) for preemption resilience
        if (sample_idx + 1) % 50 == 0:
            try:
                partial_path = Path("output/70b_cett_features_partial.npz")
                y_partial = np.array([s["label"] for s in samples[:sample_idx + 1]])
                np.savez_compressed(partial_path, X=X[:sample_idx + 1], y=y_partial,
                                    completed=sample_idx + 1)
                print(f"  Partial checkpoint: {sample_idx + 1} samples saved")
                sys.stdout.flush()
                _save_to_volume()
            except Exception as e:
                print(f"  Partial save warning: {e}")

    return X


def _fit_sparse_probe_70b(X, y, n_layers: int, intermediate_size: int):
    """Two-stage sparse probing: univariate ranking + L1 logistic regression.

    Same methodology as 8B but handles the larger feature space
    (70B: 80 layers x 28,672 = 2,293,760 neurons x 2 feature types = 4,587,520 features).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    sys.stdout.flush()

    n_features = X.shape[1]
    mask_0 = y == 0
    mask_1 = y == 1

    # Stage 1: Per-neuron effect sizes (Cohen's d)
    print("Stage 1: Computing per-neuron effect sizes...")
    sys.stdout.flush()

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
        if (start // chunk_size) % 10 == 0:
            print(f"  Processed {end:,}/{n_features:,} features "
                  f"({100*end/n_features:.1f}%)")
            sys.stdout.flush()

    # Select top candidates
    positive_mask = effect_sizes > 0.3
    n_candidates_by_threshold = int(positive_mask.sum())
    top_k = min(2000, n_candidates_by_threshold)
    if top_k < 100:
        top_k = 2000
    top_indices = np.argsort(-effect_sizes)[:top_k]
    top_indices = top_indices[effect_sizes[top_indices] > 0]

    print(f"  {n_candidates_by_threshold:,} features with effect size > 0.3")
    print(f"  Selected {len(top_indices):,} candidate features for stage 2")
    sys.stdout.flush()

    # Stage 2: L1-regularized logistic regression
    print("\nStage 2: Sparse probing on candidate features...")
    sys.stdout.flush()

    X_candidates = X[:, top_indices].copy()
    means = X_candidates.mean(axis=0)
    X_candidates -= means
    stds = X_candidates.std(axis=0)
    stds[stds < 1e-8] = 1.0
    X_candidates /= stds

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(clf, X_candidates, y, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    sys.stdout.flush()

    # Extract h-neurons
    coefs_candidates = clf.coef_[0]
    coefs = np.zeros(X.shape[1])
    coefs[top_indices] = coefs_candidates
    h_neuron_flat_indices = np.where(coefs > 1e-6)[0]
    print(f"H-neurons found: {len(h_neuron_flat_indices)} / {n_layers * intermediate_size} "
          f"({len(h_neuron_flat_indices) / (n_layers * intermediate_size) * 100:.4f}%)")

    # Map flat indices to (layer, neuron, feature_type)
    h_neurons = []
    for flat_idx in h_neuron_flat_indices:
        if flat_idx < n_layers * intermediate_size:
            layer_idx = flat_idx // intermediate_size
            neuron_idx = flat_idx % intermediate_size
            feature_type = "answer"
        else:
            adjusted = flat_idx - n_layers * intermediate_size
            layer_idx = adjusted // intermediate_size
            neuron_idx = adjusted % intermediate_size
            feature_type = "prompt"
        h_neurons.append({
            "layer": int(layer_idx),
            "neuron": int(neuron_idx),
            "feature_type": feature_type,
            "coefficient": float(coefs[flat_idx]),
        })

    # Deduplicate
    unique_neurons = {}
    for hn in h_neurons:
        key = (hn["layer"], hn["neuron"])
        if key not in unique_neurons or hn["coefficient"] > unique_neurons[key]["coefficient"]:
            unique_neurons[key] = hn
    h_neurons_deduped = sorted(unique_neurons.values(), key=lambda x: -x["coefficient"])

    print(f"Unique (layer, neuron) h-neurons: {len(h_neurons_deduped)}")
    return h_neurons_deduped, cv_scores


# ===========================================================================
# Phase 2: Ablation on Physician Test Set
# ===========================================================================
@app.function(
    image=image,
    gpu="A100-80GB:2",
    secrets=[hf_secret],
    timeout=43200,  # 12 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=196608,
)
def run_70b_ablation():
    """Run ablation sweep on physician test set with 8 alpha levels.

    Requires run_70b_phase1 output (70b_h_neurons.json).
    Outputs: 70b_ablation_results.json with per-case results and aggregate metrics.
    """
    import numpy as np
    import torch

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Restore prerequisites and any partial results
    _restore_from_volume([
        "70b_h_neurons.json",
        "70b_ablation_results.json",
        "70b_ablation_partial.json",
    ])

    # Idempotent check
    output_path = OUTPUT_DIR / "70b_ablation_results.json"
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        # Check if all alphas are completed
        completed_alphas = set(r["alpha"] for r in existing.get("per_case_results", []))
        if len(completed_alphas) >= len(ALPHA_LEVELS):
            print("70b_ablation_results.json already complete. Skipping.")
            return

    # Check prerequisite
    h_neurons_path = OUTPUT_DIR / "70b_h_neurons.json"
    if not h_neurons_path.exists():
        raise RuntimeError("70b_h_neurons.json not found. Run run_70b_phase1 first.")

    with open(h_neurons_path) as f:
        h_data = json.load(f)
    h_neurons = h_data["h_neurons"]
    print(f"Loaded {len(h_neurons)} h-neurons from 70B identification")

    # Load test data
    with open("data/physician_test.json") as f:
        cases = json.load(f)
    print(f"Physician test: {len(cases)} cases")

    print("=" * 60)
    print("70B PHASE 2: Ablation Sweep on Physician Test Set")
    print(f"Alpha levels: {ALPHA_LEVELS}")
    print("=" * 60)
    sys.stdout.flush()

    # Load model
    model, tokenizer = _load_model()

    ablator = HNeuronAblator70B(model, h_neurons)

    # Resume from partial results
    all_case_results = []
    completed_pairs = set()
    partial_path = OUTPUT_DIR / "70b_ablation_partial.json"
    if partial_path.exists():
        with open(partial_path) as f:
            all_case_results = json.load(f)
        completed_pairs = {(r["alpha"], r["case_index"]) for r in all_case_results}
        completed_alphas_so_far = sorted(set(r["alpha"] for r in all_case_results))
        print(f"Resuming: {len(all_case_results)} results from alphas {completed_alphas_so_far}")

    for alpha in ALPHA_LEVELS:
        # Check if this alpha is fully done
        alpha_done = all(
            (alpha, ci) in completed_pairs for ci in range(len(cases))
        )
        if alpha_done:
            print(f"\n  alpha={alpha}: already completed, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Alpha = {alpha}")
        print(f"{'='*60}")
        sys.stdout.flush()

        ablator.apply(alpha)

        n_errors = 0
        for ci, case in enumerate(cases):
            if (alpha, ci) in completed_pairs:
                continue

            message = case.get("message", case.get("prompt", ""))
            prompt = build_chat_prompt(tokenizer, message)

            try:
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                )
                # Move to model device (first parameter device for device_map="auto")
                first_device = next(model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

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
                parsed = parse_response_70b(response)
            except Exception as exc:
                n_errors += 1
                print(f"  [WARN] Case {ci} inference error (alpha={alpha}): {exc}. Fallback: benign.")
                torch.cuda.empty_cache()
                response = f"[INFERENCE_ERROR: {str(exc)[:80]}]"
                parsed = {"detection": 0, "severity": "Benign", "action": "None"}

            # Map ground truth action
            gt_action = case.get("action_truth", case.get("esi_action_text", ""))
            gt_action_category = _map_action_to_category(gt_action)

            all_case_results.append({
                "case_index": ci,
                "alpha": alpha,
                "detection_truth": case.get("detection_truth", 0),
                "action_truth": gt_action,
                "action_truth_category": gt_action_category,
                "predicted_detection": parsed["detection"],
                "predicted_severity": parsed["severity"],
                "predicted_action": parsed["action"],
                "action_correct": parsed["action"] == gt_action_category,
                "response": response[:500],
            })

            if (ci + 1) % 20 == 0:
                n_det = sum(
                    1 for r in all_case_results
                    if r["alpha"] == alpha and r["predicted_detection"] == 1
                )
                print(f"    Case {ci+1}/{len(cases)}: {n_det} detected so far at alpha={alpha}")
                sys.stdout.flush()

        ablator.restore()

        if n_errors > 0:
            print(f"  [INFO] alpha={alpha}: {n_errors} inference errors fell back to benign.")

        # Incremental save after each alpha
        with open(partial_path, "w") as f:
            json.dump(all_case_results, f, indent=2)
        try:
            _save_to_volume()
        except Exception as e:
            print(f"  Volume save warning: {e}")

        # Print alpha summary
        alpha_results = [r for r in all_case_results if r["alpha"] == alpha]
        tp = sum(1 for r in alpha_results if r["detection_truth"] == 1 and r["predicted_detection"] == 1)
        fn = sum(1 for r in alpha_results if r["detection_truth"] == 1 and r["predicted_detection"] == 0)
        tn = sum(1 for r in alpha_results if r["detection_truth"] == 0 and r["predicted_detection"] == 0)
        fp = sum(1 for r in alpha_results if r["detection_truth"] == 0 and r["predicted_detection"] == 1)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  Alpha {alpha}: TP={tp} FN={fn} TN={tn} FP={fp} "
              f"Sens={sens:.3f} Spec={spec:.3f}")
        sys.stdout.flush()

    # Compute aggregate metrics at each alpha
    aggregate_metrics = _compute_ablation_metrics(all_case_results)

    # Assemble final output
    final_result = {
        "model": MODEL_ID,
        "n_h_neurons": len(h_neurons),
        "n_cases": len(cases),
        "alpha_levels": ALPHA_LEVELS,
        "aggregate_metrics": aggregate_metrics,
        "per_case_results": all_case_results,
    }

    with open(output_path, "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"\nSaved ablation results to {output_path}")

    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()

    _save_to_volume()
    print("Phase 2 (ablation) complete.")


def _compute_ablation_metrics(all_case_results: list[dict]) -> list[dict]:
    """Compute sensitivity, specificity, MCC, and action accuracy at each alpha."""
    import numpy as np

    metrics_by_alpha = []

    alphas_present = sorted(set(r["alpha"] for r in all_case_results))
    for alpha in alphas_present:
        alpha_results = [r for r in all_case_results if r["alpha"] == alpha]

        tp = sum(1 for r in alpha_results if r["detection_truth"] == 1 and r["predicted_detection"] == 1)
        fn = sum(1 for r in alpha_results if r["detection_truth"] == 1 and r["predicted_detection"] == 0)
        tn = sum(1 for r in alpha_results if r["detection_truth"] == 0 and r["predicted_detection"] == 0)
        fp = sum(1 for r in alpha_results if r["detection_truth"] == 0 and r["predicted_detection"] == 1)

        n = len(alpha_results)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # MCC
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        # Action accuracy (among cases where detection was correct)
        action_correct = sum(1 for r in alpha_results if r.get("action_correct", False))
        action_acc = action_correct / n if n > 0 else 0.0

        # Per-action-category accuracy
        action_categories = ["Call 911/ED", "Contact Doctor", "Routine Follow-up", "No Action/Self-care"]
        per_action = {}
        for cat in action_categories:
            cat_cases = [r for r in alpha_results if r.get("action_truth_category") == cat]
            if cat_cases:
                cat_correct = sum(1 for r in cat_cases if r.get("predicted_action") == cat)
                per_action[cat] = {
                    "n": len(cat_cases),
                    "correct": cat_correct,
                    "accuracy": cat_correct / len(cat_cases),
                }
            else:
                per_action[cat] = {"n": 0, "correct": 0, "accuracy": 0.0}

        metrics_by_alpha.append({
            "alpha": alpha,
            "n": n,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "mcc": round(mcc, 4),
            "action_accuracy": round(action_acc, 4),
            "per_action_accuracy": per_action,
        })

        print(f"  Alpha={alpha}: Sens={sens:.3f} Spec={spec:.3f} MCC={mcc:.3f} "
              f"ActionAcc={action_acc:.3f}")

    return metrics_by_alpha


# ===========================================================================
# Phase 3: Logit Lens Analysis
# ===========================================================================
@app.function(
    image=image,
    gpu="A100-80GB:2",
    secrets=[hf_secret],
    timeout=21600,  # 6 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=196608,
)
def run_70b_logit_lens():
    """Logit lens analysis on Llama-3.1-70B-Instruct.

    Projects hidden states at each transformer layer to vocabulary space to track
    where the triage decision "locks in". Compares across alpha levels.

    Requires: 70b_h_neurons.json (from phase 1).
    Optionally uses: 70b_ablation_results.json (to select FN/TP/TN cases).
    Outputs: 70b_logit_lens.json
    """
    import numpy as np
    import torch
    from torch.nn import functional as F

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Restore from volume
    _restore_from_volume([
        "70b_h_neurons.json",
        "70b_ablation_results.json",
        "70b_logit_lens.json",
    ])

    # Idempotent
    output_path = OUTPUT_DIR / "70b_logit_lens.json"
    if output_path.exists():
        print("70b_logit_lens.json already exists. Skipping.")
        return

    # Check prerequisite
    h_neurons_path = OUTPUT_DIR / "70b_h_neurons.json"
    if not h_neurons_path.exists():
        raise RuntimeError("70b_h_neurons.json not found. Run run_70b_phase1 first.")

    with open(h_neurons_path) as f:
        h_data = json.load(f)
    h_neurons = h_data["h_neurons"]
    print(f"Loaded {len(h_neurons)} h-neurons")

    # Load test cases
    with open("data/physician_test.json") as f:
        all_cases = json.load(f)
    print(f"Physician test: {len(all_cases)} cases")

    print("=" * 60)
    print("70B PHASE 3: Logit Lens Analysis")
    print("=" * 60)
    sys.stdout.flush()

    # Select cases: prioritize FN and TP from ablation results
    selected_indices = _select_logit_lens_cases(OUTPUT_DIR, all_cases)
    cases = [all_cases[i] for i in selected_indices]
    print(f"Selected {len(cases)} cases for logit lens analysis")

    # Load model
    model, tokenizer = _load_model()
    n_layers = model.config.num_hidden_layers

    # Build ablator and logit lens extractor
    ablator = HNeuronAblator70B(model, h_neurons)

    # Get token group IDs
    group_ids = _get_token_group_ids(tokenizer, SAFETY_TOKEN_GROUPS)
    print(f"\nToken groups mapped to IDs:")
    for gname, ids in group_ids.items():
        sample_tokens = [tokenizer.decode([tid]) for tid in ids[:5]]
        print(f"  {gname}: {len(ids)} tokens (e.g., {sample_tokens})")

    # Run logit lens for each alpha
    all_results = {}
    first_device = next(model.parameters()).device

    for alpha in ANALYSIS_ALPHAS:
        print(f"\n{'='*60}")
        print(f"Logit Lens at alpha = {alpha}")
        print(f"{'='*60}")
        sys.stdout.flush()

        alpha_results = []
        for ci, (orig_idx, case) in enumerate(zip(selected_indices, cases)):
            if ci % 10 == 0:
                print(f"  Case {ci+1}/{len(cases)}...")
                sys.stdout.flush()

            ablator.apply(alpha)

            message = case.get("message", case.get("prompt", ""))
            prompt = build_chat_prompt(tokenizer, message)
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(first_device)

            # Extract per-layer hidden states
            layer_logits = _extract_logit_lens(model, inputs["input_ids"], n_layers)

            # Compute group probs at each layer
            per_layer = {}
            for layer_idx, logits in layer_logits.items():
                gp = _compute_group_probs(logits, group_ids)
                top_tokens = _compute_top_tokens(logits, tokenizer, k=5)
                per_layer[str(layer_idx)] = {
                    "group_probs": gp,
                    "top_tokens": top_tokens,
                }

            # Generate response at this alpha
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            ablator.restore()

            alpha_results.append({
                "case_index": int(orig_idx),
                "alpha": alpha,
                "detection_truth": case.get("detection_truth", 0),
                "action_truth": case.get("action_truth", "None"),
                "message_preview": message[:200],
                "response": response,
                "per_layer": per_layer,
            })

        all_results[str(alpha)] = alpha_results
        print(f"  Completed {len(alpha_results)} cases at alpha={alpha}")

    # Divergence analysis
    divergence = _analyze_decision_locking(all_results, n_layers)

    # Assemble final output
    final_result = {
        "model": MODEL_ID,
        "n_h_neurons": len(h_neurons),
        "n_cases": len(cases),
        "analysis_alphas": ANALYSIS_ALPHAS,
        "n_layers": n_layers,
        "results_by_alpha": all_results,
        "divergence_analysis": divergence,
    }

    with open(output_path, "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"\nSaved logit lens results to {output_path}")

    # Print summary
    if divergence:
        hazard_cases = {k: v for k, v in divergence.items() if v["detection_truth"] == 1}
        benign_cases = {k: v for k, v in divergence.items() if v["detection_truth"] == 0}
        if hazard_cases:
            import numpy as _np
            max_div_layers = [v["max_divergence_layer"] for v in hazard_cases.values()]
            mean_kl = _np.mean([v["max_kl"] for v in hazard_cases.values()])
            print(f"\nHazard cases (n={len(hazard_cases)}):")
            print(f"  Mean max divergence layer: {_np.mean(max_div_layers):.1f}")
            print(f"  Mean max KL: {mean_kl:.4f}")
        if benign_cases:
            import numpy as _np
            max_div_layers = [v["max_divergence_layer"] for v in benign_cases.values()]
            print(f"\nBenign cases (n={len(benign_cases)}):")
            print(f"  Mean max divergence layer: {_np.mean(max_div_layers):.1f}")

    _save_to_volume()
    print("Phase 3 (logit lens) complete.")


def _select_logit_lens_cases(output_dir: Path, all_cases: list[dict]) -> list[int]:
    """Select a balanced subset of cases for logit lens: FN, TP, TN."""
    import numpy as np

    rng = np.random.RandomState(SEED)

    ablation_path = output_dir / "70b_ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            abl_data = json.load(f)
        # Get baseline (alpha=1.0) results
        baseline = [
            r for r in abl_data.get("per_case_results", [])
            if abs(r["alpha"] - 1.0) < 0.01
        ]
        if baseline:
            fn_indices = [r["case_index"] for r in baseline
                          if r["detection_truth"] == 1 and r["predicted_detection"] == 0]
            tp_indices = [r["case_index"] for r in baseline
                          if r["detection_truth"] == 1 and r["predicted_detection"] == 1]
            tn_indices = [r["case_index"] for r in baseline
                          if r["detection_truth"] == 0 and r["predicted_detection"] == 0]
            print(f"From 70B ablation baseline: {len(fn_indices)} FN, "
                  f"{len(tp_indices)} TP, {len(tn_indices)} TN")

            selected = []
            if fn_indices:
                selected.extend(fn_indices[:20])
            if tp_indices:
                n_tp = min(15, MAX_LOGIT_LENS_CASES - len(selected))
                selected.extend(
                    rng.choice(tp_indices, min(n_tp, len(tp_indices)), replace=False).tolist()
                )
            if tn_indices:
                n_tn = min(15, MAX_LOGIT_LENS_CASES - len(selected))
                selected.extend(
                    rng.choice(tn_indices, min(n_tn, len(tn_indices)), replace=False).tolist()
                )
            if selected:
                return selected

    # Fallback
    return list(range(min(MAX_LOGIT_LENS_CASES, len(all_cases))))


def _extract_logit_lens(model, input_ids, n_layers: int) -> dict:
    """Run forward pass with hooks and project each layer's hidden states through LM head."""
    import torch

    hidden_states = {}
    hooks = []

    for i, layer in enumerate(model.model.layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states[layer_idx] = output[0].detach()
                else:
                    hidden_states[layer_idx] = output.detach()
            return hook_fn
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        with torch.no_grad():
            model(input_ids)

        lm_head = model.lm_head
        norm = model.model.norm

        layer_logits = {}
        for layer_idx, hidden in hidden_states.items():
            h = hidden[:, -1, :]  # last token
            # Move to the device of the norm layer for processing
            h_device = norm.weight.device
            h = h.to(h_device)
            h_normed = norm(h)
            # Move to LM head device
            lm_device = next(lm_head.parameters()).device
            h_normed = h_normed.to(lm_device)
            logits = lm_head(h_normed).squeeze(0)
            layer_logits[layer_idx] = logits.float().cpu()

        return layer_logits
    finally:
        for h in hooks:
            h.remove()


def _get_token_group_ids(tokenizer, groups: dict) -> dict:
    """Map each token group to a set of token IDs."""
    group_ids = {}
    for group_name, keywords in groups.items():
        ids = set()
        for kw in keywords:
            for variant in [kw, kw.lower(), kw.upper(), kw.capitalize(),
                            f" {kw}", f" {kw.lower()}"]:
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                ids.update(tokens)
        group_ids[group_name] = sorted(ids)
    return group_ids


def _compute_group_probs(logits, group_ids: dict) -> dict:
    """Compute mean probability for each token group."""
    import torch
    from torch.nn import functional as F

    probs = F.softmax(logits.detach().float(), dim=-1)
    group_probs = {}
    for group_name, ids in group_ids.items():
        if ids:
            group_probs[group_name] = float(probs[ids].sum().item())
        else:
            group_probs[group_name] = 0.0
    return group_probs


def _compute_top_tokens(logits, tokenizer, k: int = 10) -> list:
    """Return top-k tokens by probability."""
    import torch
    from torch.nn import functional as F

    probs = F.softmax(logits.float(), dim=-1)
    topk = torch.topk(probs, k)
    return [
        {"token": tokenizer.decode([idx.item()]), "token_id": idx.item(),
         "prob": float(p.item())}
        for idx, p in zip(topk.indices, topk.values)
    ]


def _analyze_decision_locking(results_by_alpha: dict, n_layers: int) -> dict:
    """Analyze at which layer the triage decision diverges between alpha levels.

    Computes KL divergence between safety token distributions at alpha=1.0 vs alpha=2.0.
    """
    import numpy as np

    baseline_key = "1.0"
    amplified_key = "2.0"

    if baseline_key not in results_by_alpha or amplified_key not in results_by_alpha:
        return {}

    baseline_results = results_by_alpha[baseline_key]
    amplified_results = results_by_alpha[amplified_key]

    analysis = {}
    for bl, amp in zip(baseline_results, amplified_results):
        case_idx = bl["case_index"]
        kl_per_layer = []
        reassure_shift = []

        for layer_idx in range(n_layers):
            bl_probs = bl["per_layer"].get(str(layer_idx), {}).get("group_probs", {})
            amp_probs = amp["per_layer"].get(str(layer_idx), {}).get("group_probs", {})

            bl_reassure = bl_probs.get("reassure", 0.0) + bl_probs.get("routine", 0.0)
            amp_reassure = amp_probs.get("reassure", 0.0) + amp_probs.get("routine", 0.0)
            reassure_shift.append(amp_reassure - bl_reassure)

            groups = sorted(set(list(bl_probs.keys()) + list(amp_probs.keys())))
            p = np.array([bl_probs.get(g, 1e-10) for g in groups], dtype=np.float64)
            q = np.array([amp_probs.get(g, 1e-10) for g in groups], dtype=np.float64)
            p = p / p.sum()
            q = q / q.sum()
            kl = float(np.sum(p * np.log(p / np.clip(q, 1e-10, None))))
            kl_per_layer.append(kl)

        analysis[str(case_idx)] = {
            "kl_per_layer": kl_per_layer,
            "reassure_shift_per_layer": reassure_shift,
            "max_divergence_layer": int(np.argmax(kl_per_layer)),
            "max_kl": float(max(kl_per_layer)),
            "detection_truth": bl["detection_truth"],
        }

    return analysis


# ===========================================================================
# Model preload utility (run once to cache model weights to volume)
# ===========================================================================
@app.function(
    image=image,
    gpu="A100-80GB:2",
    secrets=[hf_secret],
    timeout=7200,  # 2 hours — enough for 70B download
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=196608,
)
def preload_model():
    """Download Llama-3.1-70B-Instruct to the model cache volume.

    Run this once before phase1 to avoid download overhead inside the timed function.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    print(f"Downloading {MODEL_ID} to /model-cache ...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model downloaded and loaded in {time.time() - t0:.1f}s")
    print(f"Architecture: {model.config.num_hidden_layers} layers, "
          f"intermediate_size={model.config.intermediate_size}")

    # Quick smoke test
    from transformers import pipeline
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    print(f"Smoke test: '{tokenizer.decode(out[0], skip_special_tokens=True)}'")

    model_volume.commit()
    print("Model cached to volume. Ready for phase1.")


# ===========================================================================
# Local entrypoint: spawn phases sequentially with --detach support
# ===========================================================================
@app.local_entrypoint()
def main(phase: str = "all"):
    """Spawn 70B pipeline phases.

    Usage:
        modal run 70b_h_neuron_pipeline.py           # run all phases
        modal run 70b_h_neuron_pipeline.py --phase preload
        modal run 70b_h_neuron_pipeline.py --phase phase1
        modal run 70b_h_neuron_pipeline.py --phase phase2
        modal run 70b_h_neuron_pipeline.py --phase phase3

    For detached (background) runs, use:
        modal run --detach 70b_h_neuron_pipeline.py --phase phase1
    """
    if phase in ("preload", "all"):
        print("Running preload_model...")
        preload_model.remote()
        print("Preload complete.")

    if phase in ("phase1", "all"):
        print("Running run_70b_phase1...")
        run_70b_phase1.remote()
        print("Phase 1 complete.")

    if phase in ("phase2", "all"):
        print("Running run_70b_ablation...")
        run_70b_ablation.remote()
        print("Phase 2 complete.")

    if phase in ("phase3", "all"):
        print("Running run_70b_logit_lens...")
        run_70b_logit_lens.remote()
        print("Phase 3 complete.")
