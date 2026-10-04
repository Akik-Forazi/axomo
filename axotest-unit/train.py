#!/usr/bin/env python3
"""
Fine-tune CodeT5-small (60M params) for AXOTEST unit test generation.

Task: text2text-generation (seq2seq)
Input: source code → Output: test code

Usage:
    python train.py --data ./data/train.jsonl --output ./model/
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl, train_test_split, print_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Salesforce/codet5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    from transformers import (
        AutoTokenizer, T5ForConditionalGeneration,
        TrainingArguments, Trainer, DataCollatorForSeq2Seq
    )
    from datasets import Dataset
    import numpy as np

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} samples")

    hf_data = [{"input": d["input"], "output": d["output"]} for d in data]
    train, test = train_test_split(hf_data)
    train_ds = Dataset.from_list(train)
    test_ds = Dataset.from_list(test)
    print_stats(train, test, "output")

    print(f"Loading tokenizer: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(examples):
        inputs = tokenizer(examples["input"], truncation=True, max_length=512, padding="max_length")
        labels = tokenizer(examples["output"], truncation=True, max_length=512, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    print(f"Loading model: {args.base_model} (60M params)...")
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode for BLEU (simplified — just measure token overlap)
        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        exact_match = sum(p == l for p, l in zip(pred_texts, label_texts)) / len(label_texts)
        return {"exact_match": exact_match}

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    print(f"\nTraining {args.base_model} for {args.epochs} epochs...")
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")
    print(f"Export to ONNX: python ../shared/export_onnx.py --model {args.output} --output {args.output}-onnx/ --task text2text-generation")

if __name__ == "__main__":
    main()
