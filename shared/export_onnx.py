#!/usr/bin/env python3
"""
Export a HuggingFace model to ONNX format for use with
@huggingface/transformers in Node.js.

Usage:
    python export_onnx.py --model ./fine-tuned-model/ --output ./onnx/ --task text-classification
"""
import argparse
import os
import sys

def export_to_onnx(model_path: str, output_dir: str, task: str = "text-classification"):
    """Export a HuggingFace model to ONNX format."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from optimum.onnxruntime import ORTModelForSequenceClassification
    import torch

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if task == "text-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_path, export=True
        )
    elif task == "text2text-generation":
        from transformers import AutoModelForSeq2SeqLM
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_path, export=True)
    elif task == "token-classification":
        from transformers import AutoModelForTokenClassification
        from optimum.onnxruntime import ORTModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        ort_model = ORTModelForTokenClassification.from_pretrained(model_path, export=True)
    else:
        print(f"Unsupported task: {task}")
        sys.exit(1)

    print(f"Saving ONNX model to {output_dir}...")
    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save a config.json that specifies the task
    import json
    config = {
        "task": task,
        "model_type": model.config.model_type,
        "max_length": getattr(model.config, "max_position_embeddings", 512),
    }
    with open(os.path.join(output_dir, "axodex_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Report file size
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    print(f"\nExport complete!")
    print(f"  Output: {output_dir}")
    print(f"  Size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Task: {task}")
    print(f"\nTo use with AXOVB/AXOTEST:")
    print(f"  --model-path {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--output", required=True, help="Output directory for ONNX")
    parser.add_argument("--task", default="text-classification",
                       choices=["text-classification", "text2text-generation", "token-classification"])
    args = parser.parse_args()
    export_to_onnx(args.model, args.output, args.task)
