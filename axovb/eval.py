#!/usr/bin/env python3
"""
Evaluate the AXOVB bump classifier model.

Measures: accuracy, F1, precision, recall, inference speed.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl

def main():
    parser = argparse.ArgumentParser(description="Evaluate AXOVB model")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--data", required=True, help="Test data (JSONL)")
    args = parser.parse_args()

    from transformers import pipeline
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} test samples")

    print(f"Loading model: {args.model}...")
    classifier = pipeline("text-classification", model=args.model, device=-1)  # CPU

    # Run inference
    print("Running inference...")
    predictions = []
    labels = []
    times = []

    LABEL2ID = {"none": 0, "patch": 1, "minor": 2, "major": 3}

    for item in data:
        text = item["text"]
        true_label = LABEL2ID.get(item["label"], 0)

        start = time.time()
        result = classifier(text[:2000])
        elapsed = time.time() - start
        times.append(elapsed)

        pred_label = LABEL2ID.get(result[0]["label"], 0) if result else 0
        predictions.append(pred_label)
        labels.append(true_label)

    # Metrics
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    cm = confusion_matrix(labels, predictions)

    print(f"\n── AXOVB Model Evaluation ────────────────────")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  F1:        {f1*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  Avg inference: {sum(times)/len(times)*1000:.1f}ms")
    print(f"  P95 inference: {sorted(times)[int(len(times)*0.95)]*1000:.1f}ms")
    print(f"  Confusion matrix:")
    labels_names = ["none", "patch", "minor", "major"]
    print(f"    {'':>8} " + " ".join(f"{n:>8}" for n in labels_names))
    for i, row in enumerate(cm):
        print(f"    {labels_names[i]:>8} " + " ".join(f"{v:>8}" for v in row))
    print(f"─────────────────────────────────────────────\n")

    target = 0.90
    if acc >= target:
        print(f"  ✓ PASS — accuracy {acc*100:.1f}% ≥ {target*100:.0f}% target")
    else:
        print(f"  ✗ FAIL — accuracy {acc*100:.1f}% < {target*100:.0f}% target")

if __name__ == "__main__":
    main()
