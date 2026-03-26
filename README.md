# Clinical H-Neuron Triage

Code, data, and results for:

> **"Clinical fine-tuning silently replaces the safety circuit in language models, catastrophically impairing emergency detection"**
> Basu S, Patel SY, Sheth P, et al. 2026 (under review).

## Overview

This repository contains the complete pipeline for:
1. **CETT-based sparse probing** to identify h-neurons (over-compliance neurons) in Meta-Llama-3.1-8B-Instruct
2. **Graded ablation** (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}) on physician-created and real-world Medicaid triage test sets
3. **Logit lens analysis** tracing mechanistic pathway of h-neuron modulation
4. **QLoRA fine-tuning** and identification of domain-specific medical h-neurons
5. **2×2 factorial experiment**: fine-tuning status × h-neuron domain
6. **Scale replication** in Llama-3.1-70B-Instruct
7. **Permutation control**: null distribution from 100 independently seeded random neuron sets

**Key finding:** Fine-tuning on a combined training corpus (22.7% hazardous) improved specificity by 40% (0.691→0.971; McNemar p=8.57×10⁻⁵) while collapsing hazard detection sensitivity by 87% (0.409→0.053; McNemar p=4.98×10⁻¹¹). CETT-based mechanistic auditing identified this circuit replacement before behavioral testing.

## Repository Structure

```
.
├── data/
│   ├── physician_test.json      # 200 physician-created triage scenarios (132 hazardous, 68 benign)
│   ├── realworld_test.csv       # 2,000 de-identified Medicaid patient messages (165 hazardous)
│   └── fine_tuning_set.json     # 1,280 labeled training cases (290 hazardous, 22.7%)
├── results/
│   ├── h_neurons.json           # 213 identified base-model h-neurons with layer/coefficient info
│   ├── medical_h_neurons.json   # 5 fine-tuning-induced medical h-neurons
│   ├── ablation_results.json    # Phase 2 ablation results (8 alpha levels × 2 test sets)
│   ├── triage_metrics_by_alpha.csv  # Sensitivity/specificity/MCC by alpha level
│   ├── two_by_two_results.json  # 2×2 factorial raw results
│   ├── two_by_two_summary.json  # 2×2 factorial summary metrics
│   ├── finetuned_baseline_metrics.json  # Fine-tuned model baseline performance
│   └── comparison_with_baselines.csv   # Comparison with GPT-5.1, Guardrail, CQL systems
├── figures/                     # All manuscript figures (PNG)
├── 01_identify_h_neurons.py     # CETT extraction + sparse probing (Modal A100-80GB)
├── 02_ablation_inference.py     # Scaled ablation inference (Modal A10G)
├── 03_evaluate_triage.py        # Sensitivity/specificity/MCC computation
├── 04_analysis_figures.py       # Figure generation scripts
├── 05_logit_lens.py             # Logit lens analysis
├── 06_logit_lens_figures.py     # Logit lens figure generation
├── 07_control_experiments.py    # Permutation control + CETT transfer analysis
├── 08_finetune_triage.py        # QLoRA fine-tuning pipeline
├── 09_evaluate_finetuned.py     # Fine-tuned model evaluation
├── 10_identify_medical_h_neurons.py  # Medical-domain CETT probing
├── 11_two_by_two_ablation.py    # 2×2 factorial ablation
├── 12_vulnerability_analysis.py # Case-level rescued FN analysis
├── 13_finetuned_logit_lens.py   # Logit lens on fine-tuned model
├── 14_llm_judge_validation.py   # LLM-as-judge validation (Claude Opus)
├── 70b_h_neuron_pipeline.py     # Llama-3.1-70B h-neuron identification
├── 70b_baseline_test.py         # 70B baseline ablation sweep
├── make_fig_permutation_histogram.py  # Extended Data Figure 5 generation
├── config.py                    # Shared configuration (keywords, alpha levels, paths)
├── modal_pipeline.py            # Cloud GPU orchestration (Modal)
└── orchestrator.py              # Phase-gated pipeline runner
```

## Reproducing the Results

### Requirements

```bash
pip install modal scipy numpy matplotlib scikit-learn transformers peft bitsandbytes unsloth
```

GPU compute is orchestrated via [Modal](https://modal.com). You will need a Modal account and a Hugging Face token with access to `meta-llama/Meta-Llama-3.1-8B-Instruct`.

```bash
modal setup
modal secret create huggingface HF_TOKEN=your_token_here
```

### Full Pipeline

```bash
# Phase 1: H-neuron identification (A100-80GB, ~2-3 hrs)
python -m modal run modal_pipeline.py::phase1_identify

# Phase 2: Ablation sweep (A10G ×10, ~2-3 hrs)
python -m modal run modal_pipeline.py::phase2_ablation

# Phase 3: Logit lens (A100-80GB, ~1-2 hrs)
python -m modal run modal_pipeline.py::phase3_logit_lens

# Extension 1: Fine-tuning + 2×2 factorial (A100-80GB, ~6-8 hrs)
python -m modal run modal_pipeline.py::extension1_finetuning

# Extension 2: 70B replication (A100-80GB, ~4-6 hrs)
python -m modal run modal_pipeline.py::extension2_70b

# Download results to local results/ directory
python -m modal run modal_pipeline.py::download_results
```

### Local Analysis (no GPU required)

With results already in `results/`, generate all figures and metrics locally:

```bash
python 03_evaluate_triage.py      # Recompute metrics from raw results
python 04_analysis_figures.py     # Regenerate figures
python make_fig_permutation_histogram.py  # Permutation control histogram
```

### Estimated GPU Cost

| Phase | GPU | ~Hours | ~Cost |
|-------|-----|--------|-------|
| H-neuron identification | A100-80GB | 2–3 | $8–12 |
| Ablation sweep | A10G ×10 | 2–3 | $4–6 |
| Logit lens | A100-80GB | 1–2 | $4–8 |
| Fine-tuning + 2×2 | A100-80GB | 6–8 | $24–32 |
| 70B replication | A100-80GB | 4–6 | $16–24 |
| **Total** | | **~15–22 hrs** | **~$56–82** |

## Data

### Physician-Created Test Set (`data/physician_test.json`)

200 synthetic clinical vignettes constructed by board-certified physicians spanning the Emergency Severity Index (ESI) severity spectrum. Each vignette includes:
- `message`: Patient-reported symptom description
- `detection_truth`: Ground truth hazard label (1=hazardous, 0=benign)
- `esi_level`: ESI severity level (1–5)
- `hazard_category`: Clinical hazard type (cardiac, suicide_risk, pediatric, etc.)

### Real-World Test Set (`data/realworld_test.csv`)

2,000 de-identified patient messages from a Medicaid population health program (Waymark Care). Messages were de-identified by removing all 18 HIPAA identifiers before analysis. Approved by WCG IRB (protocol #20253751) with waiver of informed consent.

Fields include `message`, `hazard_detection` (ground truth), `esi_level`, and clinician action metadata.

### Fine-Tuning Set (`data/fine_tuning_set.json`)

1,280 labeled triage cases used for QLoRA fine-tuning: 480 physician-created scenarios (50.0% hazardous) + 800 de-identified real-world Medicaid messages (6.2% hazardous; combined 22.7% hazardous).

## Key Results

| Metric | Base Model | Fine-Tuned Model |
|--------|-----------|-----------------|
| Sensitivity (hazard detection) | 0.409 (95% CI 0.329–0.494) | 0.053 (95% CI 0.026–0.105) |
| Specificity | 0.691 (95% CI 0.574–0.788) | 0.971 (95% CI 0.899–0.992) |
| MCC | 0.098 | 0.054 |
| McNemar p (sensitivity change) | — | 4.98×10⁻¹¹ |
| H-neurons identified | 213 (layers 10–14) | 5 (layers 23–30) |
| Jaccard overlap (base vs. fine-tuned) | 0.000 | — |

## Citation

```bibtex
@article{basu2026clinical,
  title={Clinical fine-tuning silently replaces the safety circuit in language models, catastrophically impairing emergency detection},
  author={Basu, Sanjay and Patel, Sadiq Y and Sheth, Parth and Muralidharan, Bhairavi and Elamaran, Namrata and Kinra, Aakriti and Kharwadkar, Sagar and Farhat-Sabet, Ryan and Morgan, John and Batniji, Rajaie},
  note={Under review},
  year={2026},
  note={Under review}
}
```

## License

Code: MIT License. Data: CC BY 4.0 (de-identified patient data released under IRB-approved protocol).

## Ethics

This study was approved by WCG IRB (Princeton, NJ; protocol #20253751) with waiver of informed consent. The real-world test set comprises de-identified patient messages from a pre-existing Medicaid population health program; no new patient data were collected.
