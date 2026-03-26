# Clinical H-Neuron Triage

Code and data for:

> **"Clinical fine-tuning silently replaces the safety circuit in language models, impairing emergency detection"**
> Basu S, Patel SY, Sheth P, Muralidharan B, Elamaran N, Kinra A, Kharwadkar S, Farhat-Sabet R, Morgan J, Batniji R. Under review, 2026.

## Overview

We fine-tuned Llama-3.1-8B-Instruct on 1,280 labeled Medicaid triage messages and discovered that fine-tuning created a completely new set of over-compliance neurons (h-neurons) with zero overlap with the base-model circuit (Jaccard = 0.000). This circuit replacement—not modification—explains why fine-tuning catastrophically collapsed hazard detection sensitivity (0.409 → 0.053) while improving specificity (0.691 → 0.971).

## Repository Structure

```
├── 01_identify_h_neurons.py        # CETT computation + sparse L1 probing (base model)
├── 02_ablation_inference.py        # Ablation sweep (alpha 0.0–3.0) on triage test sets
├── 03_evaluate_triage.py           # Metrics: sensitivity, specificity, MCC
├── 04_analysis_figures.py          # Figures and comparison tables
├── 05_logit_lens.py                # Logit lens trajectory analysis
├── 06_logit_lens_figures.py        # Logit lens visualization
├── 07_control_experiments.py       # Random neuron controls + CETT transfer validation
├── 08_finetune_triage.py           # QLoRA fine-tuning pipeline
├── 09_evaluate_finetuned.py        # Fine-tuned model baseline evaluation
├── 10_identify_medical_h_neurons.py # Medical h-neuron identification (fine-tuned model)
├── 11_two_by_two_ablation.py       # 2×2 factorial ablation sweep
├── 12_vulnerability_analysis.py    # Case-level rescued FN analysis
├── 13_finetuned_logit_lens.py      # Logit lens on fine-tuned model
├── 14_llm_judge_validation.py      # LLM-judge validation of triage responses
├── config.py                       # Hyperparameters and paths
├── modal_pipeline.py               # Cloud orchestration (Modal)
├── data/
│   ├── physician_test_github.json  # 200 physician-adjudicated test cases (de-identified)
│   └── realworld_test_github.csv   # 2,000 real-world Medicaid test cases (de-identified)
├── results/
│   ├── h_neurons.json              # Base-model h-neurons (n=213)
│   ├── medical_h_neurons.json      # Fine-tuned model h-neurons (n=5)
│   ├── ablation_results.json       # Full ablation sweep results
│   ├── triage_metrics_by_alpha.csv # Metrics at each ablation level
│   ├── two_by_two_results.json     # 2×2 factorial results
│   ├── two_by_two_summary.json     # 2×2 summary statistics
│   ├── finetuned_baseline_metrics.json # Fine-tuned model baseline
│   └── comparison_with_baselines.csv   # Comparison across all systems
└── figures/                        # Publication figures (PDF + PNG)
```

## Reproduction

```bash
pip install modal torch transformers peft bitsandbytes
modal setup

# Run full pipeline on Modal (A100-80GB)
modal run modal_pipeline.py::run_all_phases

# Or run individual steps locally
python 01_identify_h_neurons.py    # requires A100-80GB
python 02_ablation_inference.py    # requires A10G
python 03_evaluate_triage.py       # CPU only
```

## Key Results

| Model | Sensitivity | Specificity | MCC |
|-------|------------|-------------|-----|
| GPT-4.1 Safety (best baseline) | 1.000 | 0.950 | 0.970 |
| Llama-3.1-8B base | 0.409 | 0.691 | 0.095 |
| Llama-3.1-8B fine-tuned | 0.053 | 0.971 | 0.068 |
| Base + h-neuron suppression (α=0.5) | 0.386 | 0.650 | 0.028 |

Fine-tuning created 5 new h-neurons in layers 23–30 (output-adjacent) with zero overlap with the 213 base-model neurons in layers 10–14 (Jaccard = 0.000). Suppressing these post-fine-tuning neurons does not rescue hazard detection sensitivity.

## Citation

```bibtex
@article{basu2026clinical,
  title={Clinical fine-tuning silently replaces the safety circuit in language models, impairing emergency detection},
  author={Basu, Sanjay and Patel, Sadiq Y and Sheth, Parth and Muralidharan, Bhairavi and Elamaran, Namrata and Kinra, Aakriti and Kharwadkar, Sagar and Farhat-Sabet, Ryan and Morgan, John and Batniji, Rajaie},
  year={2026},
  note={Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE).
