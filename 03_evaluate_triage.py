#!/usr/bin/env python3
"""Step 3: Compute triage safety metrics for each alpha level.

Metrics:
  - Sensitivity (Wilson CI)
  - Specificity (Wilson CI)
  - MCC
  - Action accuracy (bootstrap CI)
  - Critical under-triage rate (Wilson CI)
  - FN correction rate (baseline FN -> TP)
  - TP disruption rate (baseline TP -> FN)
  - McNemar's test vs baseline (alpha=1.0)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


OUTPUT_DIR = Path("output")

ACTION_MAP = {
    "None": 0, "Benign": 0,
    "Routine Follow-up": 1, "Routine": 1,
    "Contact Doctor": 2, "Urgent": 2,
    "Call 911/988": 3, "Emergent": 3,
}


def wilson_ci(count: int, total: int, alpha: float = 0.05):
    """Wilson score interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    lo, hi = proportion_confint(count, total, alpha=alpha, method="wilson")
    return (float(lo), float(hi))


def bootstrap_ci(y_true, y_pred, metric_func, n_boot=2000, seed=42):
    """Bootstrap 95% CI for a metric."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            s = metric_func(np.array(y_true)[idx], np.array(y_pred)[idx])
            scores.append(s)
        except Exception:
            pass
    if not scores:
        return (0.0, 0.0)
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def mcc(y_true, y_pred):
    """Matthews correlation coefficient."""
    from sklearn.metrics import matthews_corrcoef
    return matthews_corrcoef(y_true, y_pred)


def mcnemar_test(y_baseline, y_ablated, y_true):
    """McNemar's test comparing two classifiers on the same data."""
    # b = baseline correct, ablated wrong; c = baseline wrong, ablated correct
    b = sum(1 for yb, ya, yt in zip(y_baseline, y_ablated, y_true)
            if yb == yt and ya != yt)
    c = sum(1 for yb, ya, yt in zip(y_baseline, y_ablated, y_true)
            if yb != yt and ya == yt)
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}
    # Exact binomial test (better for small counts)
    p_value = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) < 25 else None
    # Chi-squared approximation
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    p_chi2 = 1 - stats.chi2.cdf(chi2, 1)
    return {
        "statistic": float(chi2),
        "p_value": float(p_value if p_value is not None else p_chi2),
        "b_baseline_correct_ablated_wrong": b,
        "c_baseline_wrong_ablated_correct": c,
    }


def evaluate_alpha(results_at_alpha: list[dict], baseline_results: list[dict] = None):
    """Compute all metrics for results at a single alpha level."""
    y_true_det = np.array([r["detection_truth"] for r in results_at_alpha])
    y_pred_det = np.array([r["predicted_detection"] for r in results_at_alpha])

    y_true_act = np.array([ACTION_MAP.get(r["action_truth"], 0) for r in results_at_alpha])
    y_pred_act = np.array([ACTION_MAP.get(r["predicted_action"], 0) for r in results_at_alpha])

    # Detection metrics
    tp = int(np.sum((y_true_det == 1) & (y_pred_det == 1)))
    fn = int(np.sum((y_true_det == 1) & (y_pred_det == 0)))
    fp = int(np.sum((y_true_det == 0) & (y_pred_det == 1)))
    tn = int(np.sum((y_true_det == 0) & (y_pred_det == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)
    mcc_val = mcc(y_true_det, y_pred_det) if len(set(y_true_det)) > 1 and len(set(y_pred_det)) > 1 else 0.0
    mcc_ci = bootstrap_ci(y_true_det, y_pred_det, mcc)

    # Action accuracy
    act_acc = float(np.mean(y_true_act == y_pred_act))
    act_acc_ci = bootstrap_ci(y_true_act, y_pred_act, lambda yt, yp: float(np.mean(yt == yp)))

    # Critical under-triage: among Urgent/Emergent cases, how many triaged below true level
    critical_mask = y_true_act >= 2
    if critical_mask.sum() > 0:
        under_triage = int(np.sum(y_pred_act[critical_mask] < y_true_act[critical_mask]))
        crit_rate = under_triage / int(critical_mask.sum())
        crit_ci = wilson_ci(under_triage, int(critical_mask.sum()))
    else:
        under_triage = 0
        crit_rate = 0.0
        crit_ci = (0.0, 0.0)

    metrics = {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens,
        "sensitivity_ci_lo": sens_ci[0],
        "sensitivity_ci_hi": sens_ci[1],
        "specificity": spec,
        "specificity_ci_lo": spec_ci[0],
        "specificity_ci_hi": spec_ci[1],
        "mcc": mcc_val,
        "mcc_ci_lo": mcc_ci[0],
        "mcc_ci_hi": mcc_ci[1],
        "action_accuracy": act_acc,
        "action_accuracy_ci_lo": act_acc_ci[0],
        "action_accuracy_ci_hi": act_acc_ci[1],
        "critical_under_triage": crit_rate,
        "critical_under_triage_ci_lo": crit_ci[0],
        "critical_under_triage_ci_hi": crit_ci[1],
        "n_critical_cases": int(critical_mask.sum()),
        "n_under_triaged": under_triage,
    }

    # Comparison to baseline
    if baseline_results is not None:
        y_base_det = np.array([r["predicted_detection"] for r in baseline_results])
        # FN correction: baseline FN that become TP
        baseline_fn_mask = (y_true_det == 1) & (y_base_det == 0)
        if baseline_fn_mask.sum() > 0:
            fn_corrected = int(np.sum(y_pred_det[baseline_fn_mask] == 1))
            fn_correction_rate = fn_corrected / int(baseline_fn_mask.sum())
        else:
            fn_corrected = 0
            fn_correction_rate = 0.0

        # TP disruption: baseline TP that become FN
        baseline_tp_mask = (y_true_det == 1) & (y_base_det == 1)
        if baseline_tp_mask.sum() > 0:
            tp_disrupted = int(np.sum(y_pred_det[baseline_tp_mask] == 0))
            tp_disruption_rate = tp_disrupted / int(baseline_tp_mask.sum())
        else:
            tp_disrupted = 0
            tp_disruption_rate = 0.0

        # McNemar's test
        mcnemar = mcnemar_test(y_base_det.tolist(), y_pred_det.tolist(), y_true_det.tolist())

        metrics.update({
            "fn_corrected": fn_corrected,
            "fn_correction_rate": fn_correction_rate,
            "tp_disrupted": tp_disrupted,
            "tp_disruption_rate": tp_disruption_rate,
            "mcnemar_statistic": mcnemar["statistic"],
            "mcnemar_p_value": mcnemar["p_value"],
            "mcnemar_b": mcnemar["b_baseline_correct_ablated_wrong"],
            "mcnemar_c": mcnemar["c_baseline_wrong_ablated_correct"],
        })

    return metrics


def main():
    results_path = OUTPUT_DIR / "ablation_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run 02_ablation_inference.py first.")
        sys.exit(1)

    with open(results_path) as f:
        all_results = json.load(f)

    print(f"Loaded {len(all_results)} results")

    # Group by dataset and alpha
    grouped = {}
    for r in all_results:
        key = (r["dataset"], r["alpha"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    datasets = sorted(set(r["dataset"] for r in all_results))
    alphas = sorted(set(r["alpha"] for r in all_results))
    print(f"Datasets: {datasets}")
    print(f"Alpha levels: {alphas}")

    # Compute metrics
    all_metrics = []
    for ds in datasets:
        # Get baseline (alpha=1.0)
        baseline_key = (ds, 1.0)
        baseline_results = grouped.get(baseline_key)

        for alpha in alphas:
            key = (ds, alpha)
            if key not in grouped:
                continue
            results = grouped[key]
            print(f"\n{ds} alpha={alpha}: {len(results)} cases")

            bl = baseline_results if alpha != 1.0 else None
            metrics = evaluate_alpha(results, baseline_results=bl)
            metrics["dataset"] = ds
            metrics["alpha"] = alpha
            metrics["n_cases"] = len(results)
            all_metrics.append(metrics)

            print(f"  Sensitivity: {metrics['sensitivity']:.3f} "
                  f"({metrics['sensitivity_ci_lo']:.3f}-{metrics['sensitivity_ci_hi']:.3f})")
            print(f"  Specificity: {metrics['specificity']:.3f}")
            print(f"  MCC: {metrics['mcc']:.3f}")
            print(f"  Action accuracy: {metrics['action_accuracy']:.3f}")
            print(f"  Critical under-triage: {metrics['critical_under_triage']:.3f}")
            if "mcnemar_p_value" in metrics:
                print(f"  McNemar vs baseline: p={metrics['mcnemar_p_value']:.4f}")

    # Save
    df = pd.DataFrame(all_metrics)
    cols_order = [
        "dataset", "alpha", "n_cases",
        "sensitivity", "sensitivity_ci_lo", "sensitivity_ci_hi",
        "specificity", "specificity_ci_lo", "specificity_ci_hi",
        "mcc", "mcc_ci_lo", "mcc_ci_hi",
        "action_accuracy", "action_accuracy_ci_lo", "action_accuracy_ci_hi",
        "critical_under_triage", "critical_under_triage_ci_lo", "critical_under_triage_ci_hi",
        "tp", "fn", "fp", "tn",
        "n_critical_cases", "n_under_triaged",
    ]
    extra_cols = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + extra_cols]

    output_path = OUTPUT_DIR / "triage_metrics_by_alpha.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved metrics to {output_path}")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Sensitivity by dataset and alpha")
    print("=" * 100)
    for ds in datasets:
        print(f"\n{ds}:")
        ds_df = df[df["dataset"] == ds].sort_values("alpha")
        for _, row in ds_df.iterrows():
            mc = f"  McNemar p={row.get('mcnemar_p_value', 'N/A'):.4f}" if "mcnemar_p_value" in row and pd.notna(row.get("mcnemar_p_value")) else ""
            print(f"  alpha={row['alpha']:.2f}: "
                  f"Sens={row['sensitivity']:.3f} ({row['sensitivity_ci_lo']:.3f}-{row['sensitivity_ci_hi']:.3f}), "
                  f"Spec={row['specificity']:.3f}, "
                  f"MCC={row['mcc']:.3f}, "
                  f"CritUnderTriage={row['critical_under_triage']:.3f}"
                  f"{mc}")


if __name__ == "__main__":
    main()
