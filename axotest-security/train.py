#!/usr/bin/env python3
"""
Fine-tune DistilBERT (66M params) for AXOTEST security scanning.

Task: text-classification (2 classes: secure, vulnerable)
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl, train_test_split, print_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
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

    LABEL2ID = {"secure": 0, "vulnerable": 1}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} samples")

    hf_data = [{"text": d["text"], "label": LABEL2ID.get(d["label"], 0)} for d in data]
    train, test = train_test_split(hf_data)
    train_ds = Dataset.from_list(train)
    test_ds = Dataset.from_list(test)
    print_stats(train, test, "label")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    print(f"Loading model: {args.base_model} (66M params)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, warmup_steps=10, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", report_to="none")

    trainer = Trainer(model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics)

    print(f"\nTraining {args.base_model} for {args.epochs} epochs...")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")
    print(f"Export: python ../shared/export_onnx.py --model {args.output} --output {args.output}-onnx/ --task text-classification")

if __name__ == "__main__":
    main()
