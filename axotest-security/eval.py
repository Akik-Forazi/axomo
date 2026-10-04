#!/usr/bin/env python3
"""Evaluate the AXOTEST security scanner model."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    from transformers import pipeline
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} test samples")

    print(f"Loading model: {args.model}...")
    classifier = pipeline("text-classification", model=args.model, device=-1)

    LABEL2ID = {"secure": 0, "vulnerable": 1}
    predictions, labels, times = [], [], []

    for item in data:
        start = time.time()
        result = classifier(item["text"][:2000])
        times.append(time.time() - start)
        pred = LABEL2ID.get(result[0]["label"], 0) if result else 0
        predictions.append(pred)
        labels.append(LABEL2ID.get(item["label"], 0))

    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")

    print(f"\n── AXOTEST Security Scanner Evaluation ──────────")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  F1:        {f1*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  Avg inference: {sum(times)/len(times)*1000:.1f}ms")
    print(f"────────────────────────────────────────────────\n")

    if acc >= 0.90:
        print(f"  ✓ PASS — accuracy {acc*100:.1f}% ≥ 90% target")
    else:
        print(f"  ✗ FAIL — accuracy {acc*100:.1f}% < 90% target")

if __name__ == "__main__":
    main()
