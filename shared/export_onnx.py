#!/usr/bin/env python3
"""
Export a HuggingFace model to ONNX format using torch.onnx.export(dynamo=False).
Unpacks tokenizer BatchEncoding into individual tensors.

Usage:
    python export_onnx.py --model ./model/ --output ./onnx/ --task text-classification
"""
import argparse, os, sys

def export_to_onnx(model_path: str, output_dir: str, task: str = "text-classification"):
    import torch
    from transformers import AutoTokenizer
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Create dummy input
    dummy = tokenizer("sample text", return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    input_names = ["input_ids", "attention_mask"]
    dummy_inputs = (input_ids, attention_mask)

    if task == "text-classification":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    elif task == "token-classification":
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    elif task == "text2text-generation":
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        print(f"Unsupported task: {task}"); sys.exit(1)

    model.eval()
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"Exporting to {onnx_path} (dynamo=False)...")
    try:
        torch.onnx.export(
            model, dummy_inputs, onnx_path,
            input_names=input_names, output_names=["logits"],
            dynamo=False, opset_version=17,
        )
    except TypeError:
        # Fallback for older torch versions without dynamo param
        torch.onnx.export(model, dummy_inputs, onnx_path,
            input_names=input_names, output_names=["logits"], opset_version=17)

    tokenizer.save_pretrained(output_dir)
    import json
    config = {
        "task": task, "model_type": model.config.model_type,
        "max_length": getattr(model.config, "max_position_embeddings", 512),
        "id2label": dict(model.config.id2label) if hasattr(model.config, "id2label") else {},
        "label2id": dict(model.config.label2id) if hasattr(model.config, "label2id") else {},
    }
    with open(os.path.join(output_dir, "axodex_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(output_dir) for f in fs)
    print(f"\nExport complete! Size: {total/1024/1024:.1f} MB")
    print(f"Use with: --model-path {output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", default="text-classification",
                   choices=["text-classification", "text2text-generation", "token-classification"])
    a = p.parse_args()
    export_to_onnx(a.model, a.output, a.task)
