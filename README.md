# Clinical H-Neuron Triage

Code and data for:

> **"Clinical fine-tuning silently replaces the safety circuit in language models, impairing emergency detection"**
> Basu S, Patel SY, Sheth P, Muralidharan B, Elamaran N, Kinra A, Kharwadkar S, Farhat-Sabet R, Morgan J, Batniji R. Under review, 2026.

## Overview

We fine-tuned Llama-3.1-8B-Instruct on 1,280 labeled Medicaid triage messages using a **standard QLoRA protocol without class weighting** — replicating the default clinical deployment pipeline — and discovered that fine-tuning created a completely new set of over-compliance neurons (h-neurons) in anatomically distinct output-adjacent layers, with zero overlap with the base-model circuit. This circuit replacement explains why fine-tuning collapsed hazard detection sensitivity (0.409 → 0.053) while improving specificity (0.691 → 0.971).

**The key finding is structural, not behavioral.** CETT probing identifies the circuit location and layer distribution independently of whether sensitivity collapsed severely or mildly; the spatial dissociation between base-model h-neurons (layers 10–14) and fine-tuned h-neurons (layers 23–30) is improbable by chance (p=0.001) and would be detectable before any behavioral testing.

## Study Design

The study was designed to pre-empt alternative explanations with a pre-specified 2×2 factorial (fine-tuning status × h-neuron domain):

- **Arm A** (base + TriviaQA h-neurons): Control — general-purpose probing has weak, non-directional effects on triage
- **Arm B** (base + medical h-neurons): Validates medical h-neurons as genuine over-compliance mediators *in the base model itself* — suppression improves sensitivity, amplification degrades it (bidirectional ρ=−1.000). This demonstrates the 5 neurons identified post-fine-tuning are not training artifacts.
- **Arm C** (fine-tuned + TriviaQA h-neurons): Domain-specificity control — wrong-domain probing loses directionality after fine-tuning
- **Arm D** (fine-tuned + medical h-neurons): Primary endpoint — reversed polarity (ρ=+1.000), confirming circuit replacement rather than suppression

**Scale replication:** CETT probing was replicated in Llama-3.1-70B, identifying 724 h-neurons (0.032% of total; probe AUC 0.910). The layer distribution differs from the 8B model (distributed across all 80 layers vs. concentrated in 10–14), confirming scale-dependent circuit architecture and that re-identification is required at each scale.

**Architectural generalizability:** The SwiGLU activation decomposition underlying CETT is shared by Mistral, Qwen-2.5, Yi-1.5, DeepSeek-V3, and other major open-weight models used in clinical fine-tuning, making the methodology directly portable without modification.

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
├── data/
│   ├── combined_train.json         # 1,280 training cases (physician + real-world)
│   └── realworld_test.json         # 2,000 real-world Medicaid test cases (de-identified)
├── output/
│   ├── h_neurons.json              # Base-model h-neurons (n=213, layers 10–14)
│   ├── medical_h_neurons.json      # Fine-tuned model h-neurons (n=5, layers 23–30)
│   ├── ablation_results.json       # Full ablation sweep results
│   ├── triage_metrics_by_alpha.csv # Metrics at each ablation level
│   ├── two_by_two_results.json     # 2×2 factorial results
│   ├── two_by_two_summary.json     # 2×2 summary statistics
│   ├── random_sets/                # 87 permutation control sets (random neuron ablation)
│   └── triviaqa_inputs/            # TriviaQA consistency-filter data
└── figures/                        # Publication figures (fig1–fig5, efig1–efig5, PDF + PNG)
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

| Model / Condition | Sensitivity | Specificity | MCC |
|---|---|---|---|
| GPT-4.1 Safety (best baseline) | 1.000 | 0.950 | 0.970 |
| Llama-3.1-8B base | 0.409 | 0.691 | 0.095 |
| Llama-3.1-8B fine-tuned (standard QLoRA) | 0.053 | 0.971 | 0.068 |
| Base + h-neuron suppression (α=0.5) | 0.386 | 0.650 | 0.028 |

**Circuit replacement evidence:**
- Base h-neurons: 213 neurons in layers 10–14 (middle, factual retrieval zone)
- Fine-tuned h-neurons: 5 neurons in layers 23–30 (output-adjacent)
- Jaccard overlap = 0.000; spatial concentration p=0.001 vs. random null
- Arm B validation: medical h-neurons function as causal over-compliance mediators in the *base* model (bidirectional ρ=−1.000), confirming they are not fine-tuning artifacts
- Arm D reversal: fine-tuned model shows opposite polarity (ρ=+1.000), consistent only with circuit replacement

**70B replication:**
- 724 h-neurons identified (0.032% of 2,293,760 total FFN neurons; probe AUC 0.910)
- Layer distribution shifts from concentrated (8B: layers 10–14) to distributed (70B: all 80 layers)
- Aggregate behavioral suppression effect is null at 70B scale (bidirectional case-level changes cancel), consistent with distributed over-compliance encoding; domain-specific fine-tuning of the 70B remains to be tested

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
