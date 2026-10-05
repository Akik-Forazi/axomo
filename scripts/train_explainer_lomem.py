#!/usr/bin/env python3
"""
Ultra-low-memory training of distilgpt2 explainer — for 4GB RAM environments.
- batch_size 2, max_length 64
- gradient_accumulation_steps 8 (effective batch 16)
- gradient_checkpointing for activation memory savings
- Captures full loss + perplexity curves and sample generations.
"""
import argparse, os, sys, json, time
sys.path.insert(0, '/home/z/my-project/axomo/shared')
from utils import load_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    import torch

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} samples, using limit={args.limit}", flush=True)
    data = data[:args.limit]
    texts = [item["prompt"] + item["response"] for item in data]

    n = len(texts)
    n_test = max(10, n // 10)
    test_texts = texts[:n_test]
    train_texts = texts[n_test:]
    train_ds = Dataset.from_dict({"text": train_texts})
    test_ds = Dataset.from_dict({"text": test_texts})
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)

    print(f"Loading tokenizer: {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.base_model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,} ({total_params/1e6:.0f}M)", flush=True)
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.0f} MB", flush=True)

    model.gradient_checkpointing_enable()
    print(f"Gradient checkpointing: ENABLED (saves activation memory)", flush=True)

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

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=5,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=False,
        logging_steps=1,
        logging_first_step=True,
        disable_tqdm=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print(f"\nTraining {args.base_model} for {args.epochs} epoch(s)...", flush=True)
    print(f"  Samples: {len(train_ds)}", flush=True)
    print(f"  Batch size: {args.batch_size} (effective: {args.batch_size * args.grad_accum})", flush=True)
    print(f"  Max length: {args.max_length}", flush=True)
    print(f"  Optimizer steps: ~{len(train_ds) // (args.batch_size * args.grad_accum) * args.epochs}", flush=True)
    print(flush=True)

    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0

    logs = trainer.state.log_history
    train_losses = [(l["step"], l["loss"]) for l in logs if "loss" in l]
    eval_losses = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]

    print(f"\n=== TRAINING PEAKS ===", flush=True)
    print(f"Train time: {train_time:.1f}s ({train_time/60:.1f}m)", flush=True)
    print(f"Total optimizer steps: {train_result.global_step}", flush=True)
    print(f"\nTrain loss curve (step, loss):", flush=True)
    for step, loss in train_losses:
        print(f"  step={step:4d}  loss={loss:.4f}", flush=True)
    print(f"\nEval loss curve (step, eval_loss, perplexity):", flush=True)
    for step, loss in eval_losses:
        ppl = torch.exp(torch.tensor(loss)).item()
        print(f"  step={step:4d}  eval_loss={loss:.4f}  ppl={ppl:.2f}", flush=True)

    peak_loss = max(l for _, l in train_losses) if train_losses else None
    final_loss = train_losses[-1][1] if train_losses else None
    peak_eval = max(l for _, l in eval_losses) if eval_losses else None
    final_eval = eval_losses[-1][1] if eval_losses else None
    if train_losses:
        print(f"\nTrain loss: peak={peak_loss:.4f}  final={final_loss:.4f}  drop={peak_loss-final_loss:.4f}", flush=True)
    if eval_losses:
        print(f"Eval  loss: peak={peak_eval:.4f}  final={final_eval:.4f}  drop={peak_eval-final_eval:.4f}", flush=True)

    print(f"\nSaving model to {args.output}...", flush=True)
    os.makedirs(args.output, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    metrics = {
        "model": args.base_model,
        "params": total_params,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "gradient_checkpointing": True,
        "train_time_seconds": train_time,
        "global_step": train_result.global_step,
        "train_loss_curve": train_losses,
        "eval_loss_curve": eval_losses,
        "peak_train_loss": peak_loss,
        "final_train_loss": final_loss,
        "peak_eval_loss": peak_eval,
        "final_eval_loss": final_eval,
    }
    with open(os.path.join(args.output, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {args.output}/training_metrics.json", flush=True)

    print(f"\n=== SAMPLE GENERATIONS ===", flush=True)
    model.eval()
    test_prompts = [
        'Features: {"num_changed_symbols": 5, "num_callers_affected": 15, "has_removed_exports": 1, "has_new_exports": 0}\nBump: major\nExplanation:',
        'Features: {"num_changed_symbols": 3, "num_callers_affected": 0, "has_removed_exports": 0, "has_new_exports": 1}\nBump: minor\nExplanation:',
        'Features: {"num_changed_symbols": 2, "num_callers_affected": 5, "has_removed_exports": 0, "has_new_exports": 0}\nBump: patch\nExplanation:',
        'Features: {"num_changed_symbols": 0, "num_callers_affected": 0, "has_removed_exports": 0, "has_new_exports": 0, "num_doc_files": 3}\nBump: none\nExplanation:',
    ]
    samples = []
    for p in test_prompts:
        inputs = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0], skip_special_tokens=True)[len(p):].strip()
        samples.append({"prompt": p, "response": response[:200]})
        print(f"PROMPT: {p[:80]}...", flush=True)
        print(f"  -> {response[:200]}", flush=True)
        print(flush=True)

    with open(os.path.join(args.output, "sample_generations.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Sample generations saved to {args.output}/sample_generations.json", flush=True)
    print(f"\nModel saved to {args.output}", flush=True)

if __name__ == "__main__":
    main()
