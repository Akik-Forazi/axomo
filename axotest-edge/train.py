#!/usr/bin/env python3
"""
Fine-tune DistilBERT (66M params) for AXOTEST edge case detection.

Task: token-classification (NER-style, 3 labels: O, B-BOUNDARY, I-BOUNDARY)
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
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification
    )
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import f1_score

    LABEL2ID = {"O": 0, "B-BOUNDARY": 1, "I-BOUNDARY": 2}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} samples")

    hf_data = [{"tokens": d["tokens"], "labels": [LABEL2ID.get(l, 0) for l in d["labels"]]} for d in data]
    train, test = train_test_split(hf_data)
    train_ds = Dataset.from_list(train)
    test_ds = Dataset.from_list(test)
    print_stats(train, test, "labels")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, is_split_into_words=True)

    def tokenize_and_align(examples):
        tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = []
        for i, labels in enumerate(examples["labels"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            prev = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned.append(-100)
                elif word_idx != prev:
                    aligned.append(labels[word_idx])
                else:
                    aligned.append(labels[word_idx])
                prev = word_idx
            all_labels.append(aligned)
        tokenized["labels"] = all_labels
        return tokenized

    train_ds = train_ds.map(tokenize_and_align, batched=True)
    test_ds = test_ds.map(tokenize_and_align, batched=True)

    print(f"Loading model: {args.base_model} (66M params)...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_preds = []
        true_labels = []
        for pred, label in zip(predictions, labels):
            for p, l in zip(pred, label):
                if l != -100:
                    true_preds.append(p)
                    true_labels.append(l)
        f1 = f1_score(true_labels, true_preds, average="weighted")
        return {"f1": f1}

    training_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, warmup_steps=10, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", report_to="none")

    trainer = Trainer(model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics)

    print(f"\nTraining {args.base_model} for {args.epochs} epochs...")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")
    print(f"Export: python ../shared/export_onnx.py --model {args.output} --output {args.output}-onnx/ --task token-classification")

if __name__ == "__main__":
    main()
