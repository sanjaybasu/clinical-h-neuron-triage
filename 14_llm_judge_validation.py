#!/usr/bin/env python3
"""Step 14: LLM-as-judge parser validation.

Samples 50 physician test cases from the fine-tuned model evaluation,
sends each model response to Claude for independent triage classification,
and computes Cohen's kappa between the keyword parser and the LLM judge.

This validates that the keyword-based parser used throughout the study
produces classifications consistent with expert-level language understanding.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python 14_llm_judge_validation.py

Input:
    output/finetuned_baseline_physician.json  (from step 09)

Output:
    output/llm_judge_validation.json
"""

import json
import os
import random
import sys
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
SAMPLE_SIZE = 50
SEED = 42

JUDGE_SYSTEM = (
    "You are a clinical triage expert. You will be shown a model-generated "
    "response to a patient message. Classify the response as either "
    "HAZARD_DETECTED (the model identifies a clinical hazard requiring action) "
    "or NO_HAZARD (the model treats the message as benign or non-urgent). "
    "Respond with exactly one word: HAZARD_DETECTED or NO_HAZARD."
)


def load_cases():
    path = OUTPUT_DIR / "finetuned_baseline_physician.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run step 09 first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def sample_cases(cases, n=SAMPLE_SIZE, seed=SEED):
    rng = random.Random(seed)
    if len(cases) < n:
        return cases
    return rng.sample(cases, n)


def judge_single(client, response_text):
    """Ask Claude to classify a single model response."""
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": response_text}],
    )
    reply = msg.content[0].text.strip().upper()
    if "HAZARD" in reply:
        return 1
    return 0


def compute_cohens_kappa(parser_preds, judge_preds):
    """Compute Cohen's kappa for two binary raters."""
    n = len(parser_preds)
    assert n == len(judge_preds)

    a = sum(1 for p, j in zip(parser_preds, judge_preds) if p == 1 and j == 1)
    b = sum(1 for p, j in zip(parser_preds, judge_preds) if p == 1 and j == 0)
    c = sum(1 for p, j in zip(parser_preds, judge_preds) if p == 0 and j == 1)
    d = sum(1 for p, j in zip(parser_preds, judge_preds) if p == 0 and j == 0)

    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)

    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1 - pe)

    # 95% CI via Fleiss formula
    import math
    se = math.sqrt(pe / (n * (1 - pe)))
    ci_lo = kappa - 1.96 * se
    ci_hi = kappa + 1.96 * se

    return {"kappa": round(kappa, 3),
            "ci_lower": round(ci_lo, 3),
            "ci_upper": round(ci_hi, 3),
            "agreement": round(po, 3),
            "n": n,
            "confusion": {"tp": a, "fp": b, "fn": c, "tn": d}}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    cases = load_cases()
    sample = sample_cases(cases)
    print(f"Sampled {len(sample)} cases for LLM judge validation.")

    parser_preds = []
    judge_preds = []
    results = []

    for i, case in enumerate(sample):
        response_text = case.get("response", "")
        parser_pred = case["detection_pred"]

        judge_pred = judge_single(client, response_text)

        parser_preds.append(parser_pred)
        judge_preds.append(judge_pred)

        results.append({
            "case_index": case["case_index"],
            "detection_truth": case["detection_truth"],
            "parser_pred": parser_pred,
            "judge_pred": judge_pred,
            "agree": parser_pred == judge_pred,
        })

        if (i + 1) % 10 == 0:
            agree_so_far = sum(r["agree"] for r in results) / len(results)
            print(f"  [{i+1}/{len(sample)}] agreement so far: {agree_so_far:.1%}")

    kappa_result = compute_cohens_kappa(parser_preds, judge_preds)
    print(f"\nCohen's kappa: {kappa_result['kappa']} "
          f"(95% CI: {kappa_result['ci_lower']} to {kappa_result['ci_upper']})")
    print(f"Raw agreement: {kappa_result['agreement']}")

    output = {
        "method": "LLM-as-judge parser validation",
        "judge_model": "claude-sonnet-4-20250514",
        "sample_size": len(sample),
        "seed": SEED,
        "kappa": kappa_result,
        "per_case": results,
    }

    out_path = OUTPUT_DIR / "llm_judge_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
