# Clinical H-Neuron Triage

Code and data for:

> **"Clinical fine-tuning silently replaces the safety circuit in language models, impairing emergency detection"**
> Basu S, Patel SY, Sheth P, Muralidharan B, Elamaran N, Kinra A, Kharwadkar S, Farhat-Sabet R, Morgan J, Batniji R. Under review, 2026.

## Overview

We fine-tuned Llama-3.1-8B-Instruct on 1,280 labeled Medicaid triage messages using a standard QLoRA protocol without class weighting — replicating the default clinical deployment pipeline — and discovered that fine-tuning created a completely new set of over-compliance neurons (h-neurons) in anatomically distinct output-adjacent layers, with zero overlap with the base-model circuit. This circuit replacement explains why fine-tuning collapsed hazard detection sensitivity while improving specificity.

The key finding is structural, not behavioral. CETT probing identifies the circuit location and layer distribution independently of whether sensitivity collapsed severely or mildly; the spatial dissociation between base-model h-neurons (layers 10–14) and fine-tuned h-neurons (layers 23–30) is improbable by chance (p=0.001).

## Study Design

The study was designed to pre-empt alternative explanations with a pre-specified 2×2 factorial (fine-tuning status × h-neuron domain):

- Arm A (base + TriviaQA h-neurons): Control — general-purpose probing has weak, non-directional effects on triage
- Arm B (base + medical h-neurons): Validates medical h-neurons as genuine over-compliance mediators in the base model itself — suppression improves sensitivity, amplification degrades it (bidirectional ρ=−1.000).
- Arm C (fine-tuned + TriviaQA h-neurons): Domain-specificity control — wrong-domain probing loses directionality after fine-tuning
- Arm D (fine-tuned + medical h-neurons): Primary endpoint — reversed polarity (ρ=+1.000), confirming circuit replacement rather than suppression

Scale replication: CETT probing was replicated in Llama-3.1-70B. The layer distribution differs from the 8B model (distributed across all 80 layers vs. concentrated in 10–14), confirming scale-dependent circuit architecture.

Architectural generalizability: The SwiGLU activation decomposition underlying CETT is shared by Mistral, Qwen-2.5, Yi-1.5, DeepSeek-V3, and other major open-weight models used in clinical fine-tuning, making the methodology directly portable without modification.

## Repository Structure

```
├── 01_identify_h_neurons.py        # CETT computation + sparse L1 probing (base model)
├── 02_ablation_inference.py        # Ablation sweep (alpha 0.0–3.0) on triage test sets
├── 03_evaluate_triage.py           # Metrics: sensitivity, specificity, MCC
├── 04_analysis_figures.py          # Figures and comparison tables
├── 05_logit_lens.py                # Logit lens trajectory analysis
├── 06_logit_lens_figures.py        # Logit lens visualization
├── 07_control_experiments.py       # Random neuron controls + CETT transfer validation
├── 08_finetune_triage.py           # QLoRA fine-tuning pipeline (standard, no class weighting)
├── 09_evaluate_finetuned.py        # Fine-tuned model baseline evaluation
├── 10_identify_medical_h_neurons.py # Medical h-neuron identification (fine-tuned model)
├── 11_two_by_two_ablation.py       # 2×2 factorial ablation sweep
├── 12_vulnerability_analysis.py    # Case-level rescued FN analysis
├── 13_finetuned_logit_lens.py      # Logit lens on fine-tuned model
├── 14_llm_judge_validation.py      # LLM-judge validation of triage responses
├── 70b_h_neuron_pipeline.py        # Llama-3.1-70B replication pipeline
├── 70b_baseline_test.py            # 70B baseline triage evaluation
├── config.py                       # Hyperparameters and paths
├── modal_pipeline.py               # Cloud orchestration (Modal)
├── data/                           # De-identified datasets (not tracked in git; available on request)
├── output/                         # Pipeline outputs (not tracked in git; reproducible via scripts)
└── figures/                        # Publication figures (not tracked in git; reproducible via scripts)
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
