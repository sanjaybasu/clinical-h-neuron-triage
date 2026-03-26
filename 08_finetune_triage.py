#!/usr/bin/env python3
"""Step 8: QLoRA instruction fine-tuning of Llama-3.1-8B-Instruct for medical triage.

Trains the model to generate structured triage responses from patient messages,
then merges LoRA adapters into the base model for downstream CETT extraction.

Usage (Modal A100-80GB):
    python 08_finetune_triage.py

Output:
    output/finetuned_model/  -- merged bfloat16 model + tokenizer
    output/finetune_metrics.json  -- training loss, val metrics
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/Llama-3.1-8B-Instruct")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
TRAIN_DATA = Path(os.environ.get("TRAIN_DATA", "data/combined_train.json"))
SEED = 42
MAX_SEQ_LEN = 512
EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 8
LORA_R = 64
LORA_ALPHA = 128
VAL_FRACTION = 0.1

# Triage prompt (matches inference pipeline)
SYSTEM_PROMPT = (
    "You are a clinical triage assistant for a Medicaid population health "
    "program. A care coordinator has shared the following patient message or "
    "encounter note. Assess whether the message contains any clinical hazard "
    "requiring action. If it does, specify the recommended action (e.g., "
    "call 911, contact doctor, routine follow-up). If not, reassure."
    " Pay careful attention to any signs of clinical urgency, including "
    "medication interactions, allergic reactions, suicidal ideation, "
    "pediatric risks, or emergency symptoms that require immediate action."
)

# ---------------------------------------------------------------------------
# Format training examples as chat turns
# ---------------------------------------------------------------------------

def format_response(case: dict) -> str:
    """Create the target response from ground truth labels."""
    detection = case.get("ground_truth_detection", 0)
    action = case.get("ground_truth_action", "None")
    severity = case.get("ground_truth_severity", "Benign")
    category = case.get("ground_truth_hazard_category", "")

    if detection == 1:
        if action == "Call 911/988":
            return (
                f"HAZARD DETECTED. This message indicates a potential {severity.lower()} "
                f"clinical situation ({category.replace('_', ' ')}). "
                f"Recommended action: Call 911 or 988 immediately. "
                f"This requires emergency evaluation."
            )
        elif action == "Contact Doctor":
            return (
                f"HAZARD DETECTED. This message suggests a clinical concern "
                f"({category.replace('_', ' ')}) requiring urgent medical attention. "
                f"Recommended action: Contact your doctor or provider today. "
                f"Do not wait for a routine appointment."
            )
        elif action == "Routine Follow-up":
            return (
                f"HAZARD DETECTED. This message contains a clinical concern "
                f"({category.replace('_', ' ')}) that should be evaluated. "
                f"Recommended action: Schedule a routine follow-up with your provider."
            )
        else:
            return (
                f"HAZARD DETECTED. This requires clinical attention. "
                f"Recommended action: Contact your healthcare provider."
            )
    else:
        return (
            "No clinical hazard detected. This message does not indicate an "
            "urgent medical concern. If your symptoms change or worsen, "
            "please contact your care coordinator."
        )


def build_chat_messages(case: dict) -> list[dict]:
    """Build Llama-3.1 chat-format messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Patient message: {case['message']}"},
        {"role": "assistant", "content": format_response(case)},
    ]


def tokenize_chat(example, tokenizer):
    """Tokenize a single chat example with labels for the assistant turn only."""
    messages = example["messages"]

    # Build the full text using the chat template
    # First, get the prompt (system + user) without the assistant response
    prompt_messages = messages[:2]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize
    full_enc = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN)
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LEN)

    # Labels: -100 for prompt tokens, actual token ids for response tokens
    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + full_enc["input_ids"][prompt_len:]

    # Pad labels to match input length
    if len(labels) < len(full_enc["input_ids"]):
        labels += [-100] * (len(full_enc["input_ids"]) - len(labels))
    labels = labels[:len(full_enc["input_ids"])]

    full_enc["labels"] = labels
    return full_enc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_out = OUTPUT_DIR / "finetuned_model"

    # Check if already done
    if (model_out / "config.json").exists():
        print(f"Fine-tuned model already exists at {model_out}. Skipping.")
        return

    # Load training data
    print(f"Loading training data from {TRAIN_DATA}...")
    with open(TRAIN_DATA) as f:
        raw_data = json.load(f)
    print(f"  {len(raw_data)} cases loaded")

    # Format as chat messages
    formatted = []
    for case in raw_data:
        formatted.append({"messages": build_chat_messages(case)})

    # Train/val split
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(formatted))
    n_val = max(1, int(len(formatted) * VAL_FRACTION))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_data = [formatted[i] for i in train_indices]
    val_data = [formatted[i] for i in val_indices]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Load tokenizer
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Tokenize datasets
    print("Tokenizing...")
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    train_ds = train_ds.map(
        lambda x: tokenize_chat(x, tokenizer),
        remove_columns=["messages"],
    )
    val_ds = val_ds.map(
        lambda x: tokenize_chat(x, tokenizer),
        remove_columns=["messages"],
    )

    # Load model with 4-bit quantization
    print(f"Loading model {MODEL_ID} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    print("Applying QLoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "finetune_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )

    # Data collator for causal LM
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_SEQ_LEN,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # Train
    print("=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)
    train_result = trainer.train()

    # Save training metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "epochs": EPOCHS,
        "lr": LR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "n_train": len(train_data),
        "n_val": len(val_data),
    }

    # Eval on val set
    eval_result = trainer.evaluate()
    metrics["val_loss"] = eval_result.get("eval_loss", None)

    with open(OUTPUT_DIR / "finetune_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTraining metrics: {metrics}")

    # Merge LoRA adapters into base model
    print("\nMerging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()

    # Save merged model in bfloat16
    print(f"Saving merged model to {model_out}...")
    model_out.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(model_out), safe_serialization=True)
    tokenizer.save_pretrained(str(model_out))

    print(f"\nFine-tuning complete. Merged model saved to {model_out}")


if __name__ == "__main__":
    main()
