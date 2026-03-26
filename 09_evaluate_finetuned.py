#!/usr/bin/env python3
"""Step 9: Evaluate fine-tuned model baseline on physician and real-world test sets.

Runs greedy inference with the same prompt/parser as the base model pipeline,
establishing the fine-tuned baseline before any h-neuron modulation.

Usage:
    python 09_evaluate_finetuned.py

Output:
    output/finetuned_baseline_physician.json  -- per-case results (200 cases)
    output/finetuned_baseline_realworld.json  -- per-case results (2000 cases)
    output/finetuned_baseline_metrics.json    -- aggregate metrics
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
FINETUNED_MODEL_DIR = OUTPUT_DIR / "finetuned_model"
MAX_NEW_TOKENS = 256  # Llama-3.1 produces direct responses (no chain-of-thought)
SEED = 42

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SYSTEM_PROMPT, SAFETY_HINT, EMERGENCY_KEYWORDS, URGENT_KEYWORDS,
    ACTION_MAP, ACTION_LABELS,
)


def build_chat_prompt(tokenizer, message: str) -> str:
    """Build prompt using chat template (must match fine-tuning format)."""
    system = SYSTEM_PROMPT + SAFETY_HINT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Patient message: {message}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_response(text: str) -> dict:
    """Parse model response into detection/severity/action.

    Llama-3.1 fine-tuned model produces direct trained-format responses:
      "HAZARD DETECTED. ..." or "No clinical hazard detected. ..."
    Falls back to keyword matching for edge cases.
    """
    text_lower = text.lower()

    # Check trained output markers first
    if "hazard detected" in text_lower:
        # Make sure it's not "no ... hazard detected"
        if "no clinical hazard" in text_lower or "no hazard detected" in text_lower:
            return {"detection": 0, "severity": "Benign", "action": "None"}
        if "call 911" in text_lower or "call 988" in text_lower or "emergency" in text_lower:
            return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
        elif "contact" in text_lower and ("doctor" in text_lower or "provider" in text_lower):
            return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}
        elif "routine" in text_lower or "follow-up" in text_lower or "follow up" in text_lower:
            return {"detection": 1, "severity": "Routine", "action": "Routine Follow-up"}
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}

    if "no clinical hazard" in text_lower or "no hazard" in text_lower:
        return {"detection": 0, "severity": "Benign", "action": "None"}

    # Keyword fallback
    if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
        return {"detection": 1, "severity": "Emergent", "action": "Call 911/988"}
    if any(kw in text_lower for kw in URGENT_KEYWORDS):
        return {"detection": 1, "severity": "Urgent", "action": "Contact Doctor"}

    return {"detection": 0, "severity": "Benign", "action": "None"}


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    rad = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, (center - rad) / denom), min(1.0, (center + rad) / denom))


def compute_metrics(results: list) -> dict:
    """Compute triage metrics from per-case results."""
    y_true = np.array([r["detection_truth"] for r in results])
    y_pred = np.array([r["detection_pred"] for r in results])

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)

    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / max(denom, 1e-10)

    return {
        "n": len(results),
        "n_hazard": int(y_true.sum()),
        "n_benign": int((1 - y_true).sum()),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": round(sens, 4),
        "sensitivity_ci": [round(x, 4) for x in sens_ci],
        "specificity": round(spec, 4),
        "specificity_ci": [round(x, 4) for x in spec_ci],
        "mcc": round(mcc, 4),
    }


def run_inference(model, tokenizer, cases, device, label="", checkpoint_path=None):
    """Run greedy inference on a list of cases with checkpoint/resume support."""
    # Resume from checkpoint if available
    results = []
    start_idx = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        start_idx = len(results)
        if start_idx > 0:
            print(f"  Resuming from checkpoint: {start_idx}/{len(cases)} done")

    for i in range(start_idx, len(cases)):
        case = cases[i]
        if i % 50 == 0:
            print(f"  [{label}] Case {i}/{len(cases)}...")
            sys.stdout.flush()

        message = case.get("message", case.get("prompt", ""))
        prompt = build_chat_prompt(tokenizer, message)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = parse_response(response)

        results.append({
            "case_index": i,
            "detection_truth": case.get("detection_truth", 0),
            "detection_pred": parsed["detection"],
            "action_truth": case.get("action_truth", "None"),
            "action_pred": parsed["action"],
            "severity_pred": parsed["severity"],
            "response": response[:300],
            "raw_response_len": len(response),
        })

        # Save checkpoint every 25 cases (local + volume)
        if checkpoint_path and (i + 1) % 25 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)
            # Also save to Modal volume if mounted
            vol_path = Path("/results") / checkpoint_path
            if vol_path.parent.parent.exists():
                vol_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(checkpoint_path, str(vol_path))

    # Final checkpoint save
    if checkpoint_path:
        with open(checkpoint_path, "w") as f:
            json.dump(results, f)
        vol_path = Path("/results") / checkpoint_path
        if vol_path.parent.parent.exists():
            vol_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(checkpoint_path, str(vol_path))

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "finetuned_baseline_metrics.json"
    # Always re-run (prompt format may have changed)

    # Load fine-tuned model (use 4-bit quantization if on smaller GPU)
    print(f"Loading fine-tuned model from {FINETUNED_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_MODEL_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = os.environ.get("USE_4BIT", "0") == "1"
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("Using 4-bit quantization for smaller GPU")
        model = AutoModelForCausalLM.from_pretrained(
            str(FINETUNED_MODEL_DIR),
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(FINETUNED_MODEL_DIR),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    device = next(model.parameters()).device
    print("Model loaded.")

    # Load test sets
    physician_path = DATA_DIR / "physician_test.json"
    realworld_path = DATA_DIR / "realworld_test.json"

    with open(physician_path) as f:
        physician_cases = json.load(f)
    with open(realworld_path) as f:
        realworld_cases = json.load(f)
    print(f"Loaded {len(physician_cases)} physician + {len(realworld_cases)} real-world cases")

    # Run inference on physician test (with checkpoint resume)
    phys_checkpoint = OUTPUT_DIR / "finetuned_baseline_physician.json"
    phys_done = phys_checkpoint.exists() and len(json.load(open(phys_checkpoint))) == len(physician_cases)

    if phys_done:
        with open(phys_checkpoint) as f:
            phys_results = json.load(f)
        print(f"Physician test already complete ({len(phys_results)} cases)")
    else:
        print("\n" + "=" * 60)
        print("PHYSICIAN TEST SET")
        print("=" * 60)
        phys_results = run_inference(
            model, tokenizer, physician_cases, device, "physician",
            checkpoint_path=str(phys_checkpoint))

    phys_metrics = compute_metrics(phys_results)
    print(f"\nPhysician metrics: sens={phys_metrics['sensitivity']}, "
          f"spec={phys_metrics['specificity']}, mcc={phys_metrics['mcc']}")

    # Run inference on real-world test (with checkpoint resume)
    rw_checkpoint = OUTPUT_DIR / "finetuned_baseline_realworld.json"
    rw_done = rw_checkpoint.exists() and len(json.load(open(rw_checkpoint))) == len(realworld_cases)

    if rw_done:
        with open(rw_checkpoint) as f:
            rw_results = json.load(f)
        print(f"Real-world test already complete ({len(rw_results)} cases)")
    else:
        print("\n" + "=" * 60)
        print("REAL-WORLD TEST SET")
        print("=" * 60)
        rw_results = run_inference(
            model, tokenizer, realworld_cases, device, "realworld",
            checkpoint_path=str(rw_checkpoint))

    rw_metrics = compute_metrics(rw_results)
    print(f"\nReal-world metrics: sens={rw_metrics['sensitivity']}, "
          f"spec={rw_metrics['specificity']}, mcc={rw_metrics['mcc']}")

    # Save combined metrics
    all_metrics = {
        "physician": phys_metrics,
        "realworld": rw_metrics,
        "model": str(FINETUNED_MODEL_DIR),
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    print(f"{'':20s} {'Sensitivity':>12s} {'Specificity':>12s} {'MCC':>8s}")
    print(f"{'Base DS-R1-8B':20s} {'TBD':>12s} {'TBD':>12s} {'TBD':>8s}")
    print(f"{'Fine-tuned':20s} {phys_metrics['sensitivity']:>12.3f} "
          f"{phys_metrics['specificity']:>12.3f} {phys_metrics['mcc']:>8.3f}")
    print(f"{'GPT-5.1 Safety':20s} {'0.836':>12s} {'0.956':>12s} {'0.802':>8s}")


if __name__ == "__main__":
    main()
