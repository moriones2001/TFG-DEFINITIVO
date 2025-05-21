#!/usr/bin/env python3
# train_adapter.py
# Purpose: Load dataset.jsonl, fine-tune a LoRA adapter on the base model, and save it under adapters/latest

import os
import sys
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Load environment variables
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "pysentimiento/robertuito-hate-speech")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./adapters/latest")
DATASET_FILE = "dataset.jsonl"

# 0. Check if we have any training examples
if not os.path.exists(DATASET_FILE) or os.stat(DATASET_FILE).st_size == 0:
    print(f"No training data found in {DATASET_FILE}, skipping adapter training.")
    sys.exit(0)

# 1. Load tokenizer and base model
print(f"Loading base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# Forzamos single-label classification en lugar de multi-label
base_model.config.problem_type = "single_label_classification"

# 2. Prepare dataset
print(f"Loading dataset from {DATASET_FILE}")
# load_dataset expects a dict split
ds = load_dataset("json", data_files={"train": DATASET_FILE})["train"]

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tokenize_fn, batched=True)
ds = ds.rename_column("label", "labels")
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 3. Configure LoRA
print("Configuring LoRA adapter...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(base_model, lora_config)

# 4. Training
print("Starting training...")
args = TrainingArguments(
    output_dir=ADAPTER_PATH,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="no",
    seed=42
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    tokenizer=tokenizer
)
trainer.train()

# 5. Save adapter
model.save_pretrained(ADAPTER_PATH)
print(f"Saved new adapter to {ADAPTER_PATH}")
