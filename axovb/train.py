#!/usr/bin/env python3
"""
Fine-tune DistilBERT (66M params) for AXOVB — version bump classification.

Task: text-classification (4 classes: patch, minor, major, none)
Input: git diff text (truncated to 512 tokens)
Output: {patch, minor, major, none} + confidence

Usage:
    python train.py --data ./data/train.jsonl --output ./model/
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl, train_test_split, print_stats

def main():
    parser = argparse.ArgumentParser(description="Fine-tune AXOVB bump classifier")
    parser.add_argument("--data", required=True, help="Path to training data (JSONL)")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--base-model", default="distilbert-base-uncased",
                        help="Base model to fine-tune (default: distilbert-base-uncased, 66M)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Label mapping
    LABEL2ID = {"none": 0, "patch": 1, "minor": 2, "major": 3}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    # Load data
    print("Loading data...")
    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} samples")

    # Convert to dataset format
    def to_hf(item):
        return {
            "text": item["text"],
            "label": LABEL2ID.get(item["label"], 0),
        }

    hf_data = [to_hf(item) for item in data]
    train, test = train_test_split(hf_data)

    train_ds = Dataset.from_list(train)
    test_ds = Dataset.from_list(test)

    print_stats(train, test, "label")

    # Tokenize
    print(f"Loading tokenizer: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    # Model
    print(f"Loading model: {args.base_model} (66M params)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Train
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining {args.base_model} for {args.epochs} epochs...")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()

    trainer.train()

    # Evaluate
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    # Save
    print(f"\nSaving model to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nModel saved to {args.output}")
    print(f"To export to ONNX: python ../shared/export_onnx.py --model {args.output} --output {args.output}-onnx/ --task text-classification")

if __name__ == "__main__":
    main()
