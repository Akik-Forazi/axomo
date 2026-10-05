#!/usr/bin/env python3
"""
Fine-tune distilgpt2 (82M params) for AXOVB — natural language explanation generator.

Instead of hardcoded strings like "Exported symbol(s) removed — breaking change",
this model generates contextual explanations:

  "I checked the code graph and found that authMiddleware was removed from
   the public API. This breaks 15 callers across 3 modules. Major bump."

Task: causal language modeling (text generation)
Base: distilgpt2 (82M params) — smallest GPT-2 variant
Input: "Features: {...}\nBump: major\nExplanation:"
Output: " I found 5 changed symbols including removed exports..."

The model is small (82M), runs on CPU, generates 1-3 sentences in ~100ms.

Usage:
    python train_explainer.py --data ./data/explainer_train.jsonl --output ./explainer/
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl, train_test_split

def main():
    parser = argparse.ArgumentParser(description="Fine-tune AXOVB explainer (distilgpt2, 82M)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="distilgpt2", help="distilgpt2 = 82M params")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    import torch

    # Load data
    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} samples")

    # Format as full text: prompt + response
    texts = [item["prompt"] + item["response"] for item in data]
    train_texts, test_texts = train_test_split(texts)
    train_ds = Dataset.from_dict({"text": train_texts})
    test_ds = Dataset.from_dict({"text": test_texts})

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Load tokenizer + model
    print(f"Loading tokenizer: {args.base_model} (82M params)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,} ({total_params/1e6:.0f}M)")
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.0f} MB")

    # Tokenize
    def tokenize(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    # Train
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        fp16=False,  # CPU
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print(f"\nTraining {args.base_model} for {args.epochs} epochs...")
    print(f"  Samples: {len(train_ds)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max length: {args.max_length}")
    print()

    trainer.train()

    # Save
    print(f"\nSaving model to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # Test generation
    print(f"\n── Sample generations ──────────────────────────")
    model.eval()
    test_prompts = [
        "Features: {\"num_changed_symbols\": 5, \"num_callers_affected\": 15, \"has_removed_exports\": 1, \"has_new_exports\": 0}\nBump: major\nExplanation:",
        "Features: {\"num_changed_symbols\": 3, \"num_callers_affected\": 0, \"has_removed_exports\": 0, \"has_new_exports\": 1}\nBump: minor\nExplanation:",
        "Features: {\"num_changed_symbols\": 2, \"num_callers_affected\": 5, \"has_removed_exports\": 0, \"has_new_exports\": 0}\nBump: patch\nExplanation:",
        "Features: {\"num_changed_symbols\": 0, \"num_callers_affected\": 0, \"has_removed_exports\": 0, \"has_new_exports\": 0, \"num_doc_files\": 3}\nBump: none\nExplanation:",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  {response[len(prompt):].strip()[:150]}")
        print()

    print(f"────────────────────────────────────────────────")
    print(f"\nModel saved to {args.output}")
    print(f"Export to ONNX: python ../shared/export_onnx.py --model {args.output} --output {args.output}-onnx/ --task text-generation")

if __name__ == "__main__":
    main()
