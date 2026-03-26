#!/usr/bin/env python3
"""Modal cloud execution for h-neuron ablation triage pipeline.

Two-phase execution:
  Phase 1 (A100-80GB): Identify h-neurons via CETT + sparse probing (~2-4 hrs)
  Phase 2 (A10G x10):  Ablation inference sweep on triage test sets (~1-2 hrs)

Usage:
    cd packaging/h_neuron_triage

    # Full pipeline (fire-and-forget):
    python -m modal run modal_pipeline.py::launch

    # Check status:
    python -m modal run modal_pipeline.py::status

    # Download results:
    python -m modal run modal_pipeline.py::download

    # Phase 1 only (h-neuron identification):
    python -m modal run modal_pipeline.py::run_phase1

    # Phase 2 only (ablation inference, requires phase 1 output):
    python -m modal run modal_pipeline.py::run_phase2
"""

import json
import os
import shutil
import time
from pathlib import Path

import modal

app = modal.App("h-neuron-triage-v4")

# Volumes (v3 has existing Llama-3.1-8B-Instruct results from phases 1-5)
results_volume = modal.Volume.from_name("h-neuron-results-v3", create_if_missing=True)
model_volume = modal.Volume.from_name("llama-3.1-8b-instruct-cache", create_if_missing=True)
finetuned_volume = modal.Volume.from_name("h-neuron-finetuned-llama", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

# Image with all dependencies (includes peft/bitsandbytes for fine-tuning)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "transformers>=4.38.0",
        "accelerate>=0.25.0",
        "huggingface-hub>=0.20.0",
        "safetensors>=0.4.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
    )
    .add_local_dir(".", "/app", ignore=[".venv", "__pycache__", "*.pyc", ".git"], copy=True)
)

def _restore_finetuned_model():
    """Restore fine-tuned model from dedicated volume or results volume fallback."""
    ft_dest = Path("output/finetuned_model")
    ft_dest.mkdir(parents=True, exist_ok=True)
    ft_src = Path("/finetuned")
    if (ft_src / "config.json").exists():
        for f in ft_src.iterdir():
            if f.is_file():
                shutil.copy(str(f), str(ft_dest / f.name))
        print("Restored finetuned model from /finetuned volume")
    else:
        ft_results = Path("/results/output/finetuned_model")
        if ft_results.exists():
            for f in ft_results.iterdir():
                if f.is_file():
                    shutil.copy(str(f), str(ft_dest / f.name))
            print("Restored finetuned model from /results volume")
    if not (ft_dest / "config.json").exists():
        raise RuntimeError("Fine-tuned model not found. Run phase 5 first.")


def _save_to_volume(vol):
    """Copy output/ to the results volume."""
    for dirname in ["output", "figures", "tables"]:
        src = Path(dirname)
        if src.exists():
            dest = Path(f"/results/{dirname}")
            shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
            n = len(list(src.iterdir()))
            print(f"  Saved {dirname}/: {n} files -> volume")
    vol.commit()


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=21600,  # 6 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=65536,
)
def run_phase1():
    """Phase 1: Identify h-neurons (A100-80GB, ~2-4 hours)."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"

    # Restore checkpoint from volume if available (preemption resilience)
    os.makedirs("output", exist_ok=True)
    for fname in ["triviaqa_checkpoint.json", "triviaqa_samples.json",
                   "cett_features.npz", "h_neurons.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")
            print(f"Restored {fname} from volume")

    # If h_neurons.json already exists, skip re-running
    if Path("output/h_neurons.json").exists():
        print("h_neurons.json already exists in volume. Skipping phase 1.")
        return

    print("=" * 60)
    print("PHASE 1: H-Neuron Identification")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-u", "01_identify_h_neurons.py"],
        capture_output=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Phase 1 failed with code {result.returncode}")

    _save_to_volume(results_volume)
    model_volume.commit()
    print("Phase 1 complete.")


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=86400,  # 24 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=65536,
)
def run_phase2():
    """Phase 2: Ablation inference sweep (A10G, ~1-2 hours)."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["DATA_DIR"] = "data"

    # Copy from volume: h-neurons + any partial ablation results
    os.makedirs("output", exist_ok=True)
    for fname in ["h_neurons.json", "ablation_results.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")
            print(f"Restored {fname} from volume")
    if not Path("output/h_neurons.json").exists():
        raise RuntimeError("h_neurons.json not found in volume. Run phase 1 first.")

    print("=" * 60)
    print("PHASE 2: Ablation Inference Sweep")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-u", "02_ablation_inference.py"],
        capture_output=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Phase 2 failed with code {result.returncode}")

    _save_to_volume(results_volume)
    print("Phase 2 complete.")


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=21600,  # 6 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=65536,
)
def run_logit_lens():
    """Phase 3: Logit Lens analysis (A100-80GB, ~1-2 hours)."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["DATA_DIR"] = "data"

    # Copy from volume
    os.makedirs("output", exist_ok=True)
    for fname in ["h_neurons.json", "ablation_results.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")
            print(f"Restored {fname} from volume")
    if not Path("output/h_neurons.json").exists():
        raise RuntimeError("h_neurons.json not found. Run phase 1 first.")

    print("=" * 60)
    print("PHASE 3: Logit Lens Analysis")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-u", "05_logit_lens.py"],
        capture_output=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Logit lens failed with code {result.returncode}")

    _save_to_volume(results_volume)
    print("Logit lens complete.")


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=43200,  # 12 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=32768,
)
def run_single_random_set(set_idx: int):
    """Run a single random neuron control set on one GPU."""
    import sys
    import numpy as np
    import torch
    from torch.nn import functional as F
    from collections import Counter
    from pathlib import Path
    import transformers

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.makedirs("output", exist_ok=True)

    # Load h-neurons
    h_path = Path("/results/output/h_neurons.json")
    shutil.copy(str(h_path), "output/h_neurons.json")
    with open("output/h_neurons.json") as f:
        h_data = json.load(f)
    h_neurons = h_data["h_neurons"]

    # Load test cases
    with open("data/physician_test.json") as f:
        cases = json.load(f)

    print(f"=== Random Set {set_idx}: {len(h_neurons)} h-neurons, {len(cases)} cases ===")

    # Generate the specific random set (deterministic from seed + set_idx)
    N_LAYERS = 32
    NEURONS_PER_LAYER = 14336
    rng = np.random.RandomState(42)
    layer_counts = Counter(hn["layer"] for hn in h_neurons)

    # Advance RNG to the right set
    for skip in range(set_idx):
        for layer, count in layer_counts.items():
            h_in_layer = set(hn["neuron"] for hn in h_neurons if hn["layer"] == layer)
            available = [n for n in range(NEURONS_PER_LAYER) if n not in h_in_layer]
            rng.choice(available, size=count, replace=False)

    random_neurons = []
    for layer, count in layer_counts.items():
        h_in_layer = set(hn["neuron"] for hn in h_neurons if hn["layer"] == layer)
        available = [n for n in range(NEURONS_PER_LAYER) if n not in h_in_layer]
        chosen = rng.choice(available, size=count, replace=False)
        for n in chosen:
            random_neurons.append({"layer": layer, "neuron": int(n)})
    print(f"  Generated {len(random_neurons)} random neurons")

    # Load model
    MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    # Build ablator
    layer_neurons_map = {}
    for rn in random_neurons:
        layer = rn["layer"]
        if layer not in layer_neurons_map:
            layer_neurons_map[layer] = []
        layer_neurons_map[layer].append(rn["neuron"])

    original_forwards = {}

    def apply_ablation(alpha):
        restore_ablation()
        for layer_idx, neuron_indices in layer_neurons_map.items():
            mlp = model.model.layers[layer_idx].mlp
            original_forwards[layer_idx] = mlp.forward
            idx_t = torch.tensor(sorted(set(neuron_indices)), dtype=torch.long, device=device)
            def make_patched(orig_mlp, idx, a):
                def patched(x):
                    gate = orig_mlp.gate_proj(x)
                    up = orig_mlp.up_proj(x)
                    act = F.silu(gate) * up
                    # Cast to float32 before scaling to avoid bfloat16 overflow at high alpha
                    orig_dtype = act.dtype
                    act_f32 = act.to(torch.float32)
                    act_f32[:, :, idx] = act_f32[:, :, idx] * a
                    act = act_f32.to(orig_dtype)
                    return orig_mlp.down_proj(act)
                return patched
            mlp.forward = make_patched(mlp, idx_t, alpha)

    def restore_ablation():
        for li, fwd in original_forwards.items():
            model.model.layers[li].mlp.forward = fwd
        original_forwards.clear()

    # Import config
    sys.path.insert(0, "/app")
    from config import SYSTEM_PROMPT, SAFETY_HINT, EMERGENCY_KEYWORDS, URGENT_KEYWORDS

    def build_prompt(message):
        return f"{SYSTEM_PROMPT}{SAFETY_HINT}\n\nPatient message: {message}\n\nAssessment:"

    def parse_response(text):
        text_lower = text.lower()
        if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
            return {"detection": 1, "severity": "Emergent"}
        if any(kw in text_lower for kw in URGENT_KEYWORDS):
            return {"detection": 1, "severity": "Urgent"}
        return {"detection": 0, "severity": "Benign"}

    # Run ablation at each alpha
    ALPHAS = [0.0, 1.0, 2.0, 3.0]
    results = []
    for alpha in ALPHAS:
        print(f"  Alpha = {alpha}")
        apply_ablation(alpha)
        n_errors = 0
        for i, case in enumerate(cases):
            message = case.get("message", case.get("prompt", ""))
            prompt = build_prompt(message)
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=150, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                parsed = parse_response(response)
            except Exception as exc:
                n_errors += 1
                print(f"  [WARN] Set {set_idx} case {i} alpha={alpha} error: {exc}. Fallback: benign.")
                torch.cuda.empty_cache()
                response = f"[INFERENCE_ERROR: {str(exc)[:60]}]"
                parsed = {"detection": 0, "severity": "Benign"}
            results.append({
                "set_idx": set_idx,
                "case_index": i,
                "alpha": alpha,
                "detection_truth": case.get("detection_truth", 0),
                "detection_pred": parsed["detection"],
                "response": response[:200],
            })
        restore_ablation()
        if n_errors > 0:
            print(f"    [INFO] alpha={alpha}: {n_errors} inference errors fell back to benign.")
        print(f"    Done: {len([r for r in results if r['alpha'] == alpha])} cases")

    # Save to volume
    out_path = Path(f"/results/output/random_set_{set_idx}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2)
    results_volume.commit()
    print(f"  Saved {len(results)} results to volume as random_set_{set_idx}.json")
    return {"set_idx": set_idx, "n_results": len(results)}


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=43200,  # 12 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=65536,
)
def run_controls():
    """Run Experiments B + C only (CETT transfer + logit lens control)."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["DATA_DIR"] = "data"

    # Copy from volume
    os.makedirs("output", exist_ok=True)
    for fname in ["h_neurons.json", "ablation_results.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")
            print(f"Restored {fname} from volume")
    if not Path("output/h_neurons.json").exists():
        raise RuntimeError("h_neurons.json not found. Run phase 1 first.")

    print("=" * 60)
    print("PHASE 4: Control Experiments (B + C only)")
    print("=" * 60)

    sys.path.insert(0, "/app")
    from importlib import import_module
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Restore completed results from volume
    for fname in ["cett_transfer_validation.json", "logit_lens_control_groups.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")
            print(f"Restored {fname} from volume")

    spec = import_module("07_control_experiments")

    try:
        spec.main(skip_experiment_a=True)
    except Exception as e:
        print(f"Control experiments error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _save_to_volume(results_volume)

    print("Control experiments B+C complete.")


@app.function(
    image=image,
    gpu=None,
    timeout=1800,
    volumes={"/results": results_volume},
)
def run_evaluation():
    """Phase 4: Evaluate metrics + generate figures (CPU only)."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["OUTPUT_DIR"] = "output"

    # Copy results from volume
    os.makedirs("output", exist_ok=True)
    for fname in ["ablation_results.json", "h_neurons.json",
                   "logit_lens_results.json", "logit_lens_divergence.json"]:
        src = Path(f"/results/output/{fname}")
        if src.exists():
            shutil.copy(str(src), f"output/{fname}")

    print("Running evaluation...")
    subprocess.run([sys.executable, "03_evaluate_triage.py"], check=True)

    print("Generating ablation figures...")
    subprocess.run([sys.executable, "04_analysis_figures.py"], check=True)

    print("Generating logit lens figures...")
    subprocess.run([sys.executable, "06_logit_lens_figures.py"], check=True)

    _save_to_volume(results_volume)
    print("Evaluation complete.")


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=86400,  # 24 hours
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=32768,
)
def run_all_random_sets():
    """Run all random neuron control sets sequentially with incremental saves."""
    import sys
    import numpy as np
    import torch
    from torch.nn import functional as F
    from collections import Counter
    import transformers

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.makedirs("output", exist_ok=True)

    # Load h-neurons
    h_path = Path("/results/output/h_neurons.json")
    shutil.copy(str(h_path), "output/h_neurons.json")
    with open("output/h_neurons.json") as f:
        h_data = json.load(f)
    h_neurons = h_data["h_neurons"]

    # Load test cases
    with open("data/physician_test.json") as f:
        cases = json.load(f)
    print(f"Loaded {len(h_neurons)} h-neurons, {len(cases)} cases")

    # Check which sets are already completed in volume
    N_RANDOM_SETS_CHECK = int(os.environ.get("N_RANDOM_SETS", "100"))
    completed_sets = set()
    for i in range(N_RANDOM_SETS_CHECK):
        vol_path = Path(f"/results/output/random_set_{i}.json")
        if vol_path.exists():
            completed_sets.add(i)
    if completed_sets:
        print(f"  {len(completed_sets)} sets already completed, skipping")

    # Load model once
    MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"
    print(f"Loading model {MODEL_ID}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print("Model loaded.")

    sys.path.insert(0, "/app")
    from config import SYSTEM_PROMPT, SAFETY_HINT, EMERGENCY_KEYWORDS, URGENT_KEYWORDS

    def build_prompt(message):
        return f"{SYSTEM_PROMPT}{SAFETY_HINT}\n\nPatient message: {message}\n\nAssessment:"

    def parse_response(text):
        text_lower = text.lower()
        if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
            return {"detection": 1, "severity": "Emergent"}
        if any(kw in text_lower for kw in URGENT_KEYWORDS):
            return {"detection": 1, "severity": "Urgent"}
        return {"detection": 0, "severity": "Benign"}

    N_RANDOM_SETS = int(os.environ.get("N_RANDOM_SETS", "100"))
    N_LAYERS = 32
    NEURONS_PER_LAYER = 14336
    layer_counts = Counter(hn["layer"] for hn in h_neurons)
    ALPHAS = [0.0, 1.0, 2.0, 3.0]

    def build_chat_prompt_local(message):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT + SAFETY_HINT},
            {"role": "user", "content": f"Patient message: {message}"},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    original_forwards = {}

    def apply_ablation(neurons, alpha):
        restore_ablation()
        layer_map = {}
        for rn in neurons:
            layer = rn["layer"]
            if layer not in layer_map:
                layer_map[layer] = []
            layer_map[layer].append(rn["neuron"])
        for layer_idx, neuron_indices in layer_map.items():
            mlp = model.model.layers[layer_idx].mlp
            original_forwards[layer_idx] = mlp.forward
            idx_t = torch.tensor(sorted(set(neuron_indices)), dtype=torch.long, device=device)
            def make_patched(orig_mlp, idx, a):
                def patched(x):
                    gate = orig_mlp.gate_proj(x)
                    up = orig_mlp.up_proj(x)
                    act = F.silu(gate) * up
                    act[:, :, idx] = act[:, :, idx] * a
                    return orig_mlp.down_proj(act)
                return patched
            mlp.forward = make_patched(mlp, idx_t, alpha)

    def restore_ablation():
        for li, fwd in original_forwards.items():
            model.model.layers[li].mlp.forward = fwd
        original_forwards.clear()

    for set_idx in range(N_RANDOM_SETS):
        if set_idx in completed_sets:
            continue

        # Generate random neurons for this set
        rng = np.random.RandomState(42 + set_idx)
        random_neurons = []
        for layer, count in layer_counts.items():
            h_in_layer = set(hn["neuron"] for hn in h_neurons if hn["layer"] == layer)
            available = [n for n in range(NEURONS_PER_LAYER) if n not in h_in_layer]
            chosen = rng.choice(available, size=count, replace=False)
            for n in chosen:
                random_neurons.append({"layer": layer, "neuron": int(n)})

        print(f"\n=== Random Set {set_idx} ({len(random_neurons)} neurons) ===")

        results = []
        for alpha in ALPHAS:
            print(f"  Alpha = {alpha}")
            apply_ablation(random_neurons, alpha)
            for i, case in enumerate(cases):
                message = case.get("message", case.get("prompt", ""))
                prompt = build_chat_prompt_local(message)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=256, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                parsed = parse_response(response)
                results.append({
                    "set_idx": set_idx,
                    "case_index": i,
                    "alpha": alpha,
                    "detection_truth": case.get("detection_truth", 0),
                    "detection_pred": parsed["detection"],
                    "response": response[:200],
                })
            restore_ablation()
            print(f"    Done: {len([r for r in results if r['alpha'] == alpha])} cases")

        # Save this set to volume immediately
        out_path = Path(f"/results/output/random_set_{set_idx}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out_path), "w") as f:
            json.dump(results, f, indent=2)
        results_volume.commit()
        print(f"  Saved set {set_idx} ({len(results)} results) to volume")

    print(f"\nAll {N_RANDOM_SETS} random sets complete.")


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[hf_secret],
    timeout=7200,  # 2 hours — one set takes ~45min on A100-80GB
    volumes={"/results": results_volume, "/model-cache": model_volume},
    memory=32768,
)
def run_one_random_set_v2(set_idx: int):
    """Run a single random neuron control set — corrected RNG, prompt, and token length.

    Uses identical logic to run_all_random_sets:
      - rng = np.random.RandomState(42 + set_idx)  (NOT sequential advance from seed 42)
      - tokenizer.apply_chat_template()             (NOT raw text prompt)
      - max_new_tokens=256                          (NOT 150)
    """
    import sys
    import numpy as np
    import torch
    from torch.nn import functional as F
    from collections import Counter
    import transformers

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.makedirs("output", exist_ok=True)

    # Skip if already done
    vol_path = Path(f"/results/output/random_set_{set_idx}.json")
    if vol_path.exists():
        print(f"Set {set_idx} already completed, skipping.")
        return {"set_idx": set_idx, "status": "skipped"}

    # Load h-neurons
    shutil.copy("/results/output/h_neurons.json", "output/h_neurons.json")
    with open("output/h_neurons.json") as f:
        h_neurons = json.load(f)["h_neurons"]

    # Load test cases
    with open("data/physician_test.json") as f:
        cases = json.load(f)
    print(f"Set {set_idx}: {len(h_neurons)} h-neurons, {len(cases)} cases")

    # Generate random neurons — CORRECT: independent seed per set_idx
    N_LAYERS = 32
    NEURONS_PER_LAYER = 14336
    rng = np.random.RandomState(42 + set_idx)
    layer_counts = Counter(hn["layer"] for hn in h_neurons)
    random_neurons = []
    for layer, count in layer_counts.items():
        h_in_layer = set(hn["neuron"] for hn in h_neurons if hn["layer"] == layer)
        available = [n for n in range(NEURONS_PER_LAYER) if n not in h_in_layer]
        chosen = rng.choice(available, size=count, replace=False)
        for n in chosen:
            random_neurons.append({"layer": layer, "neuron": int(n)})
    print(f"  Generated {len(random_neurons)} random neurons")

    # Load model
    MODEL_ID = "unsloth/Llama-3.1-8B-Instruct"
    print(f"Loading model {MODEL_ID}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print("Model loaded.")

    sys.path.insert(0, "/app")
    from config import SYSTEM_PROMPT, SAFETY_HINT, EMERGENCY_KEYWORDS, URGENT_KEYWORDS

    def build_chat_prompt(message):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT + SAFETY_HINT},
            {"role": "user", "content": f"Patient message: {message}"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def parse_response(text):
        text_lower = text.lower()
        if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
            return {"detection": 1, "severity": "Emergent"}
        if any(kw in text_lower for kw in URGENT_KEYWORDS):
            return {"detection": 1, "severity": "Urgent"}
        return {"detection": 0, "severity": "Benign"}

    original_forwards = {}

    def apply_ablation(neurons, alpha):
        restore_ablation()
        layer_map = {}
        for rn in neurons:
            layer_map.setdefault(rn["layer"], []).append(rn["neuron"])
        for layer_idx, neuron_indices in layer_map.items():
            mlp = model.model.layers[layer_idx].mlp
            original_forwards[layer_idx] = mlp.forward
            idx_t = torch.tensor(sorted(set(neuron_indices)), dtype=torch.long, device=device)
            def make_patched(orig_mlp, idx, a):
                def patched(x):
                    gate = orig_mlp.gate_proj(x)
                    up = orig_mlp.up_proj(x)
                    act = F.silu(gate) * up
                    act[:, :, idx] = act[:, :, idx] * a
                    return orig_mlp.down_proj(act)
                return patched
            mlp.forward = make_patched(mlp, idx_t, alpha)

    def restore_ablation():
        for li, fwd in original_forwards.items():
            model.model.layers[li].mlp.forward = fwd
        original_forwards.clear()

    ALPHAS = [0.0, 1.0, 2.0, 3.0]
    results = []
    for alpha in ALPHAS:
        print(f"  Alpha = {alpha}")
        apply_ablation(random_neurons, alpha)
        for i, case in enumerate(cases):
            message = case.get("message", case.get("prompt", ""))
            prompt = build_chat_prompt(message)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_response(response)
            results.append({
                "set_idx": set_idx,
                "case_index": i,
                "alpha": alpha,
                "detection_truth": case.get("detection_truth", 0),
                "detection_pred": parsed["detection"],
                "response": response[:200],
            })
        restore_ablation()
        print(f"    Done: {len([r for r in results if r['alpha'] == alpha])} cases")

    # Save to volume
    vol_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(vol_path), "w") as f:
        json.dump(results, f, indent=2)
    results_volume.commit()
    print(f"  Saved set {set_idx} ({len(results)} results) to volume")
    return {"set_idx": set_idx, "n_results": len(results), "status": "done"}


@app.local_entrypoint()
def run_parallel_random_sets():
    """Spawn one run_one_random_set_v2 container per missing set (fire-and-forget)."""
    import subprocess
    import time

    # Determine completed sets from volume via CLI
    try:
        out = subprocess.check_output(
            ["modal", "volume", "ls", "h-neuron-results-v3", "output"],
            text=True, stderr=subprocess.DEVNULL,
        )
        completed = set()
        for line in out.splitlines():
            if "random_set_" in line:
                try:
                    idx = int(line.strip().split("random_set_")[1].replace(".json", "").split()[0])
                    completed.add(idx)
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        print(f"[WARN] Could not read volume listing: {e}. Using empty completed set.")
        completed = set()

    missing = [i for i in range(100) if i not in completed]
    if not missing:
        print("All 100 sets already complete.")
        return

    print(f"Spawning {len(missing)} parallel jobs for sets: {missing}")
    handles = {}
    for idx in missing:
        h = run_one_random_set_v2.spawn(idx)
        handles[idx] = h
        print(f"  Spawned set {idx}: {h.object_id}")

    print(f"\nAll {len(missing)} jobs live on Modal. Polling for completion...")
    done = set()
    errors = set()
    while len(done) + len(errors) < len(missing):
        for idx, h in list(handles.items()):
            if idx in done or idx in errors:
                continue
            try:
                h.get(timeout=10)
                done.add(idx)
                print(f"  [DONE] set {idx} ({len(done)}/{len(missing)})")
            except Exception as e:
                if "TimeoutError" in type(e).__name__:
                    pass
                else:
                    errors.add(idx)
                    print(f"  [ERROR] set {idx}: {e}")
        remaining = len(missing) - len(done) - len(errors)
        if remaining > 0:
            print(f"  Status: {len(done)} done, {len(errors)} errors, {remaining} running...")
            time.sleep(30)

    print(f"\nComplete: {len(done)}/{len(missing)} succeeded, {len(errors)} failed.")
    if errors:
        print(f"Failed sets: {sorted(errors)}")


# ===========================================================================
# NEW PHASES: Fine-tuning + Medical h-neuron + 2x2 Ablation
# ===========================================================================

@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=21600,  # 6 hours
    volumes={"/results": results_volume, "/model-cache": model_volume,
             "/finetuned": finetuned_volume},
    memory=65536,
)
def run_finetune():
    """Phase 5: QLoRA fine-tuning of Llama-3.1-8B-Instruct."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"

    os.makedirs("output", exist_ok=True)

    # Check if fine-tuned model already saved in volume
    ft_config = Path("/finetuned/config.json")
    if ft_config.exists():
        print("Fine-tuned model already in volume. Restoring...")
        ft_dest = Path("output/finetuned_model")
        ft_dest.mkdir(parents=True, exist_ok=True)
        for f in Path("/finetuned").iterdir():
            if f.is_file():
                shutil.copy(str(f), str(ft_dest / f.name))
        print("Restored fine-tuned model. Skipping training.")
        return

    print("=" * 60)
    print("PHASE 5: QLoRA Fine-Tuning")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-u", "08_finetune_triage.py"],
        capture_output=False, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Fine-tuning failed with code {result.returncode}")

    # Save fine-tuned model to dedicated volume
    ft_src = Path("output/finetuned_model")
    if ft_src.exists():
        for f in ft_src.iterdir():
            if f.is_file():
                shutil.copy(str(f), f"/finetuned/{f.name}")
        finetuned_volume.commit()
        print("Saved fine-tuned model to volume")

    _save_to_volume(results_volume)
    model_volume.commit()
    print("Fine-tuning complete.")


@app.function(
    image=image,
    gpu="any",
    secrets=[hf_secret],
    timeout=21600,  # 6 hours
    volumes={"/results": results_volume, "/model-cache": model_volume,
             "/finetuned": finetuned_volume},
    memory=65536,
)
def run_evaluate_finetuned():
    """Phase 6: Evaluate fine-tuned model baseline."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["DATA_DIR"] = "data"
    # Llama-3.1-8B fits in bf16 on A10G (24GB); no quantization needed

    os.makedirs("output", exist_ok=True)

    _restore_finetuned_model()

    # Restore checkpoint files from volume (for resume after preemption)
    for ckpt in ["finetuned_baseline_physician.json",
                  "finetuned_baseline_realworld.json"]:
        vol_path = Path(f"/results/output/{ckpt}")
        local_path = Path(f"output/{ckpt}")
        if vol_path.exists() and not local_path.exists():
            shutil.copy(str(vol_path), str(local_path))
            print(f"Restored checkpoint: {ckpt}")

    # Remove stale final metrics (will be recomputed)
    for p in [Path("output/finetuned_baseline_metrics.json"),
              Path("/results/output/finetuned_baseline_metrics.json")]:
        if p.exists():
            p.unlink()

    print("=" * 60)
    print("PHASE 6: Evaluate Fine-Tuned Baseline")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-u", "09_evaluate_finetuned.py"],
        capture_output=False, env=env,
    )

    # Save even if it fails (checkpoints are valuable)
    _save_to_volume(results_volume)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed with code {result.returncode}")

    print("Evaluation complete.")


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=43200,  # 12 hours (L4 ~40 cases/hr → 480 cases per run; needs 3 runs for 1280 cases)
    volumes={"/results": results_volume, "/model-cache": model_volume,
             "/finetuned": finetuned_volume},
    memory=65536,
)
def run_medical_h_neurons():
    """Phase 7: Identify medical h-neurons via CETT on fine-tuned model."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["FINETUNED_MODEL_DIR"] = "output/finetuned_model"
    os.environ["TRAIN_DATA"] = "data/combined_train.json"

    os.makedirs("output", exist_ok=True)

    _restore_finetuned_model()

    # Restore any partial results (including intra-step checkpoint)
    for fname in ["medical_probe_responses.json", "medical_probe_dataset.json",
                   "medical_cett_features.npz", "medical_h_neurons.json",
                   "medical_probe_responses_partial.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")
            print(f"Restored {fname}")

    if Path("output/medical_h_neurons.json").exists():
        print("Medical h-neurons already identified. Skipping.")
        return

    print("=" * 60)
    print("PHASE 7: Medical H-Neuron Identification")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    # Use Popen so we can periodically commit checkpoints while subprocess runs
    import time as _time
    proc = subprocess.Popen(
        [sys.executable, "-u", "10_identify_medical_h_neurons.py"],
        env=env,
    )
    commit_interval = 300  # commit every 5 minutes
    last_commit = _time.time()
    while proc.poll() is None:
        _time.sleep(10)
        if _time.time() - last_commit >= commit_interval:
            try:
                _save_to_volume(results_volume)
                print(f"Periodic volume commit ({int((_time.time() - last_commit)/60):.0f}m interval)")
            except Exception as e:
                print(f"  Periodic commit warning: {e}")
            last_commit = _time.time()
    proc.wait()
    returncode = proc.returncode

    # Final save to volume
    _save_to_volume(results_volume)

    if returncode != 0:
        raise RuntimeError(f"Medical h-neuron ID failed with code {returncode}")

    print("Medical h-neuron identification complete.")


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[hf_secret],
    timeout=86400,  # 24 hours
    volumes={"/results": results_volume, "/model-cache": model_volume,
             "/finetuned": finetuned_volume},
    memory=65536,
)
def run_two_by_two():
    """Phase 8: 2x2 factorial ablation sweep."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["DATA_DIR"] = "data"

    os.makedirs("output", exist_ok=True)

    _restore_finetuned_model()

    # Restore h-neurons and partial results
    for fname in ["h_neurons.json", "medical_h_neurons.json",
                   "two_by_two_results.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")
            print(f"Restored {fname}")

    if not Path("output/h_neurons.json").exists():
        raise RuntimeError("h_neurons.json not found. Run phase 1 first.")
    if not Path("output/medical_h_neurons.json").exists():
        raise RuntimeError("medical_h_neurons.json not found. Run phase 7 first.")

    print("=" * 60)
    print("PHASE 8: 2x2 Factorial Ablation")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    import time as _time
    proc = subprocess.Popen(
        [sys.executable, "-u", "11_two_by_two_ablation.py"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    commit_interval = 300  # commit every 5 minutes
    last_commit = _time.time()
    output_lines = []

    import select
    import io
    while proc.poll() is None:
        # Read available output
        if proc.stdout:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                print(decoded)
                output_lines.append(decoded)
                if len(output_lines) > 200:
                    output_lines.pop(0)
        if _time.time() - last_commit >= commit_interval:
            try:
                _save_to_volume(results_volume)
                print(f"Periodic volume commit (two_by_two)")
            except Exception as e:
                print(f"  Periodic commit warning: {e}")
            last_commit = _time.time()
        _time.sleep(1)

    # Drain remaining output
    if proc.stdout:
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace").rstrip()
            print(decoded)
            output_lines.append(decoded)

    proc.wait()
    returncode = proc.returncode
    _save_to_volume(results_volume)
    if returncode != 0:
        tail = "\n".join(output_lines[-50:])
        raise RuntimeError(f"2x2 ablation failed with code {returncode}.\nLast output:\n{tail}")

    print("2x2 ablation complete.")


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[hf_secret],
    timeout=14400,  # 4 hours — plenty for 500 cases
    volumes={"/results": results_volume, "/model-cache": model_volume,
             "/finetuned": finetuned_volume},
    memory=65536,
)
def run_arm_d_alpha3_fix():
    """Targeted fix: run ONLY Arm D / realworld / alpha=3.0.

    Runs inline (no subprocess) for reliability.  Key safety features:
      - Per-case try/except: one inf/nan case cannot kill the run
      - float32 amplification: avoids bfloat16 overflow (max ~65504)
      - Intra-arm resume: if this function itself is interrupted, restart
        picks up from the last saved case_index
      - Volume commit every 25 cases + final commit
    """
    import json
    import numpy as np
    import sys
    import torch
    from torch.nn import functional as F
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ARM_NAME = "D_finetuned_medical"
    DATASET_NAME = "realworld"
    ALPHA = 3.0
    MAX_NEW_TOKENS = 256
    SEED = 42
    COMMIT_EVERY = 25  # save to volume every N cases

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    _restore_finetuned_model()

    # --- Restore existing results from volume ---
    results_path = Path("output/two_by_two_results.json")
    vol_results = Path("/results/output/two_by_two_results.json")
    if vol_results.exists():
        shutil.copy(str(vol_results), str(results_path))
        print(f"Restored two_by_two_results.json from volume")

    all_results = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing records")

    # Check if this specific combo is fully done
    existing_d_rw_a3 = [
        r for r in all_results
        if r["arm"] == ARM_NAME and r["dataset"] == DATASET_NAME
        and r["alpha"] == ALPHA
    ]
    existing_indices = {r["case_index"] for r in existing_d_rw_a3}
    print(f"Arm D / realworld / alpha=3.0: {len(existing_d_rw_a3)} cases already done")

    # --- Restore data files ---
    for fname in ["h_neurons.json", "medical_h_neurons.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")
            print(f"Restored {fname}")

    for fname in ["physician_test.json", "realworld_test.json"]:
        vol_src = Path(f"/results/data/{fname}")
        local_src = Path(f"data/{fname}")
        if not local_src.exists() and vol_src.exists():
            shutil.copy(str(vol_src), str(local_src))
            print(f"Restored {fname}")

    # --- Load and subset realworld cases (same logic + seed as 11_two_by_two_ablation.py) ---
    with open("data/realworld_test.json") as f:
        realworld_all = json.load(f)

    rng = np.random.RandomState(SEED)
    hazardous_idx = [i for i, c in enumerate(realworld_all) if c.get("detection_truth") == 1]
    benign_idx = [i for i, c in enumerate(realworld_all) if c.get("detection_truth") == 0]
    n_benign_sample = min(500 - len(hazardous_idx), len(benign_idx))
    sampled_benign = rng.choice(benign_idx, n_benign_sample, replace=False).tolist()
    stratified_idx = sorted(hazardous_idx + sampled_benign)
    realworld_cases = [realworld_all[i] for i in stratified_idx]
    print(f"Realworld subset: {len(realworld_cases)} cases "
          f"({len(hazardous_idx)} hazardous + {n_benign_sample} benign)")

    remaining = [i for i in range(len(realworld_cases)) if i not in existing_indices]
    if not remaining:
        print("Arm D / realworld / alpha=3.0 already complete. Nothing to do.")
        return

    print(f"Remaining: {len(remaining)} cases to run (alpha={ALPHA})")

    # --- Load medical h-neurons ---
    with open("output/medical_h_neurons.json") as f:
        medical_neurons = json.load(f)["medical_h_neurons"]
    print(f"Medical h-neurons: {len(medical_neurons)}")

    # --- Load fine-tuned model ---
    ft_model_dir = "output/finetuned_model"
    if not Path(ft_model_dir + "/config.json").exists():
        raise RuntimeError(f"Fine-tuned model not found at {ft_model_dir}")
    print(f"Loading fine-tuned model from {ft_model_dir} ...")

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(ft_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ft_model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # --- Build ablator with float32-safe patched_forward ---
    layer_neurons: dict = {}
    for hn in medical_neurons:
        layer = hn["layer"]
        if layer not in layer_neurons:
            layer_neurons[layer] = []
        layer_neurons[layer].append(hn["neuron"])
    for layer in layer_neurons:
        layer_neurons[layer] = sorted(set(layer_neurons[layer]))
    print(f"Ablator: {len(layer_neurons)} layers, "
          f"{sum(len(v) for v in layer_neurons.values())} neurons, alpha={ALPHA}")

    original_forwards = {}
    for layer_idx, neuron_indices in layer_neurons.items():
        mlp = model.model.layers[layer_idx].mlp
        original_forwards[layer_idx] = mlp.forward
        neuron_idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)

        def make_patched_forward(orig_mlp, idx_tensor, a):
            def patched_forward(x):
                gate = orig_mlp.gate_proj(x)
                up = orig_mlp.up_proj(x)
                act = F.silu(gate) * up
                idx = idx_tensor.to(act.device)
                # float32 to prevent bfloat16 overflow at alpha=3.0
                orig_dtype = act.dtype
                act_f32 = act.to(torch.float32)
                act_f32[:, :, idx] = act_f32[:, :, idx] * a
                act = act_f32.to(orig_dtype)
                return orig_mlp.down_proj(act)
            return patched_forward

        mlp.forward = make_patched_forward(mlp, neuron_idx_tensor, ALPHA)

    # --- Per-case inference with exception handling ---
    sys.path.insert(0, "/app")
    from config import SYSTEM_PROMPT, SAFETY_HINT, EMERGENCY_KEYWORDS, URGENT_KEYWORDS

    def build_prompt(msg):
        system = SYSTEM_PROMPT + SAFETY_HINT
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Patient message: {msg}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def parse_resp(text):
        tl = text.lower()
        if "no clinical hazard" in tl or "no hazard detected" in tl:
            return {"detection": 0, "severity": "Benign", "action": "None"}
        if "hazard detected" in tl:
            if "call 911" in tl or "call 988" in tl or "emergency" in tl:
                return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
            elif "contact" in tl and ("doctor" in tl or "provider" in tl):
                return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
            elif "routine" in tl or "follow-up" in tl or "follow up" in tl:
                return {"detection": 1, "severity": "Routine", "action": "Routine Follow-up"}
            return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
        if any(kw in tl for kw in EMERGENCY_KEYWORDS):
            return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
        if any(kw in tl for kw in URGENT_KEYWORDS):
            return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
        for phrase in ["requires immediate", "seek immediate", "requires emergency",
                       "life-threatening", "immediate medical", "immediately go",
                       "go to the nearest", "emergency evaluation"]:
            if phrase in tl:
                return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
        for phrase in ["clinical concern", "medical attention", "requires attention",
                       "should be evaluated", "see your doctor", "see a doctor",
                       "contact your", "consult your", "do not wait"]:
            if phrase in tl:
                return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
        return {"detection": 0, "severity": "Benign", "action": "None"}

    new_results = []
    n_errors = 0
    import time as _time
    last_commit = _time.time()

    for i in remaining:
        case = realworld_cases[i]
        message = case.get("message", case.get("prompt", ""))
        prompt = build_prompt(message)
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
            parsed = parse_resp(response)
        except Exception as exc:
            n_errors += 1
            print(f"  [WARN] Case {i}: {exc}. Fallback: benign.", flush=True)
            torch.cuda.empty_cache()
            response = f"[INFERENCE_ERROR: {str(exc)[:80]}]"
            parsed = {"detection": 0, "severity": "Benign", "action": "None"}

        new_results.append({
            "arm": ARM_NAME, "dataset": DATASET_NAME, "alpha": ALPHA,
            "case_index": i,
            "detection_truth": case.get("detection_truth", 0),
            "detection_pred": parsed["detection"],
            "action_truth": case.get("action_truth", "None"),
            "action_pred": parsed["action"],
            "severity_pred": parsed["severity"],
            "response": response[:300],
        })

        # Periodic commit to volume
        if len(new_results) % COMMIT_EVERY == 0:
            combined = all_results + new_results
            with open(results_path, "w") as f:
                json.dump(combined, f)
            shutil.copy(str(results_path), "/results/output/two_by_two_results.json")
            results_volume.commit()
            elapsed = int(_time.time() - last_commit)
            n_det = sum(1 for r in new_results if r["detection_pred"] == 1)
            print(f"  [{len(new_results)}/{len(remaining)}] committed "
                  f"({elapsed}s since last) | det={n_det}/{len(new_results)}", flush=True)
            last_commit = _time.time()

    # Restore original forwards
    for layer_idx, orig_fwd in original_forwards.items():
        model.model.layers[layer_idx].mlp.forward = orig_fwd

    # Final save
    combined = all_results + new_results
    with open(results_path, "w") as f:
        json.dump(combined, f)
    shutil.copy(str(results_path), "/results/output/two_by_two_results.json")
    results_volume.commit()

    n_det = sum(1 for r in new_results if r["detection_pred"] == 1)
    n_truth = sum(1 for r in new_results if r["detection_truth"] == 1)
    tp = sum(1 for r in new_results if r["detection_pred"] == 1 and r["detection_truth"] == 1)
    print(f"\n=== Arm D / realworld / alpha=3.0 COMPLETE ===")
    print(f"  {len(new_results)} cases run | {n_errors} errors (benign fallback)")
    print(f"  Detected {n_det} hazards | True hazards: {n_truth} | TP: {tp}")
    if n_truth > 0:
        print(f"  Sensitivity: {tp/n_truth:.3f}")
    print(f"  Total records in JSON: {len(combined)}")


@app.function(
    image=image,
    gpu=None,
    timeout=3600,
    volumes={"/results": results_volume},
)
def run_vulnerability_analysis():
    """Phase 9: Case-level vulnerability analysis (CPU only)."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["DATA_DIR"] = "data"

    os.makedirs("output", exist_ok=True)
    for fname in ["two_by_two_results.json", "two_by_two_summary.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")

    print("=" * 60)
    print("PHASE 9: Vulnerability Analysis")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-u", "12_vulnerability_analysis.py"],
        capture_output=False, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Vulnerability analysis failed with code {result.returncode}")

    _save_to_volume(results_volume)
    print("Vulnerability analysis complete.")


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    timeout=21600,
    volumes={"/results": results_volume, "/model-cache": model_volume,
             "/finetuned": finetuned_volume},
    memory=65536,
)
def run_finetuned_logit_lens():
    """Phase 10: Logit lens comparison (base vs fine-tuned)."""
    import subprocess
    import sys

    os.chdir("/app")
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"
    os.environ["OUTPUT_DIR"] = "output"
    os.environ["DATA_DIR"] = "data"

    os.makedirs("output", exist_ok=True)

    _restore_finetuned_model()

    # Restore h-neurons
    for fname in ["h_neurons.json", "medical_h_neurons.json",
                   "two_by_two_results.json"]:
        vol_src = Path(f"/results/output/{fname}")
        if vol_src.exists():
            shutil.copy(str(vol_src), f"output/{fname}")

    print("=" * 60)
    print("PHASE 10: Fine-Tuned Logit Lens")
    print("=" * 60)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [sys.executable, "-u", "13_finetuned_logit_lens.py"],
        capture_output=False, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Logit lens failed with code {result.returncode}")

    _save_to_volume(results_volume)
    print("Fine-tuned logit lens complete.")


# ===========================================================================
# Local entrypoints
# ===========================================================================

@app.local_entrypoint()
def run_remaining_random_sets_parallel():
    """Run sets 38-99 in parallel — one GPU container per set for maximum speed and reliability."""
    import modal
    vol = modal.Volume.from_name("h-neuron-results-v3")
    # Discover which sets are already complete in the volume
    completed = set()
    for entry in vol.listdir("output/"):
        name = entry.path.split("/")[-1]
        if name.startswith("random_set_") and name.endswith(".json"):
            try:
                idx = int(name.replace("random_set_", "").replace(".json", ""))
                completed.add(idx)
            except ValueError:
                pass
    remaining = [i for i in range(100) if i not in completed]
    print(f"Completed sets: {sorted(completed)}")
    print(f"Remaining sets to run: {remaining} ({len(remaining)} total)")
    if not remaining:
        print("All 100 sets complete.")
        return
    # .map() spawns one isolated container per set in parallel
    print(f"Launching {len(remaining)} parallel GPU containers...")
    results = list(run_single_random_set.map(remaining, order_outputs=False))
    print(f"All {len(results)} sets complete.")
    for r in sorted(results, key=lambda x: x.get("set_idx", 0)):
        print(f"  Set {r['set_idx']}: {r['n_results']} results")


@app.local_entrypoint()
def run_random_control():
    """Launch the all-in-one random control function."""
    print("Launching random neuron control (all 5 sets, sequential with incremental saves)...")
    run_all_random_sets.remote()


@app.local_entrypoint()
def launch():
    """Run the original 4-phase pipeline (TriviaQA h-neurons only)."""
    print("Starting h-neuron triage pipeline (base model only)...")
    t0 = time.time()

    print("\n[1/4] Phase 1: H-Neuron Identification...")
    run_phase1.remote()
    print(f"  Phase 1 done in {(time.time() - t0) / 60:.1f} min")

    t1 = time.time()
    print("\n[2/4] Phase 2: Ablation Inference...")
    run_phase2.remote()
    print(f"  Phase 2 done in {(time.time() - t1) / 60:.1f} min")

    t2 = time.time()
    print("\n[3/4] Phase 3: Logit Lens Analysis...")
    run_logit_lens.remote()
    print(f"  Logit lens done in {(time.time() - t2) / 60:.1f} min")

    t3 = time.time()
    print("\n[4/4] Evaluation and Figures...")
    run_evaluation.remote()
    print(f"  Eval done in {(time.time() - t3) / 60:.1f} min")

    total = (time.time() - t0) / 60
    print(f"\nPipeline complete in {total:.1f} min total.")


@app.local_entrypoint()
def launch_v3():
    """Run the full 2x2 factorial pipeline (fine-tuning + medical h-neurons).

    Phases:
      1. TriviaQA h-neuron identification (base model)
      2. Ablation inference (base model, for Arm A)
      3. QLoRA fine-tuning
      4. Evaluate fine-tuned baseline (Gate 1 check)
      5. Medical h-neuron identification (Gate 2 check)
      6. 2x2 factorial ablation (Gate 3 check)
      7. Vulnerability analysis
      8. Fine-tuned logit lens
    """
    print("=" * 60)
    print("H-NEURON TRIAGE PIPELINE V3 (2x2 FACTORIAL)")
    print("Llama-3.1-8B-Instruct")
    print("=" * 60)
    t0 = time.time()

    def elapsed():
        return f"{(time.time() - t0) / 60:.1f}m"

    print(f"\n[1/8] TriviaQA H-Neuron Identification...")
    run_phase1.remote()
    print(f"  Done ({elapsed()})")

    print(f"\n[2/8] Base Model Ablation Inference...")
    run_phase2.remote()
    print(f"  Done ({elapsed()})")

    print(f"\n[3/8] QLoRA Fine-Tuning...")
    run_finetune.remote()
    print(f"  Done ({elapsed()})")

    print(f"\n[4/8] Fine-Tuned Baseline Evaluation (GATE 1)...")
    run_evaluate_finetuned.remote()
    print(f"  Done ({elapsed()})")
    print("  >>> CHECK: sensitivity >= 60%? If not, pivot strategy. <<<")

    print(f"\n[5/8] Medical H-Neuron Identification (GATE 2)...")
    run_medical_h_neurons.remote()
    print(f"  Done ({elapsed()})")
    print("  >>> CHECK: probe AUC > 0.65? If not, pivot strategy. <<<")

    print(f"\n[6/8] 2x2 Factorial Ablation (GATE 3)...")
    run_two_by_two.remote()
    print(f"  Done ({elapsed()})")
    print("  >>> CHECK: >= 3 FN->TP conversions in Arm D? <<<")

    print(f"\n[7/8] Vulnerability Analysis...")
    run_vulnerability_analysis.remote()
    print(f"  Done ({elapsed()})")

    print(f"\n[8/8] Fine-Tuned Logit Lens...")
    run_finetuned_logit_lens.remote()
    print(f"  Done ({elapsed()})")

    print(f"\nFull pipeline complete in {elapsed()}.")
    print("Run 'python -m modal run modal_pipeline.py::download' to get results.")


@app.local_entrypoint()
def launch_remaining():
    """Run phases 6-10 only (fine-tuned eval through logit lens).

    Assumes phases 1-5 already complete on the results volume.
    """
    print("=" * 60)
    print("REMAINING PHASES (6-10)")
    print("=" * 60)
    t0 = time.time()

    def elapsed():
        return f"{(time.time() - t0) / 60:.1f}m"

    print(f"\n[6] Fine-Tuned Baseline Evaluation (GATE 1)...")
    run_evaluate_finetuned.remote()
    print(f"  Done ({elapsed()})")

    print(f"\n[7] Medical H-Neuron Identification (GATE 2)...")
    run_medical_h_neurons.remote()
    print(f"  Done ({elapsed()})")

    print(f"\n[8] 2x2 Factorial Ablation (GATE 3)...")
    run_two_by_two.remote()
    print(f"  Done ({elapsed()})")

    print(f"\n[9] Vulnerability Analysis...")
    run_vulnerability_analysis.remote()
    print(f"  Done ({elapsed()})")

    print(f"\n[10] Fine-Tuned Logit Lens...")
    run_finetuned_logit_lens.remote()
    print(f"  Done ({elapsed()})")

    print(f"\nPhases 6-10 complete in {elapsed()}.")


@app.local_entrypoint()
def status():
    """Check what's in the results volume."""
    print("Contents of h-neuron-results volume:")
    for dirpath, dirnames, filenames in os.walk("/results"):
        level = dirpath.replace("/results", "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(dirpath)}/")
        sub_indent = "  " * (level + 1)
        for f in filenames:
            fpath = os.path.join(dirpath, f)
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"{sub_indent}{f} ({size_str})")


@app.local_entrypoint()
def download():
    """Download results from volume to local directories."""
    for dirname in ["output", "figures", "tables"]:
        src = Path(f"/results/{dirname}")
        if not src.exists():
            print(f"  {dirname}/ not found in volume, skipping")
            continue
        dest = Path(dirname)
        dest.mkdir(parents=True, exist_ok=True)
        for fpath in src.rglob("*"):
            if fpath.is_file():
                rel = fpath.relative_to(src)
                local_path = dest / rel
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(fpath), str(local_path))
                print(f"  {dirname}/{rel}")
    print("Download complete.")
