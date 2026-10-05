#!/usr/bin/env python3
"""Evaluate the AXOTEST edge case detector model."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    from transformers import pipeline
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} test samples")

    print(f"Loading model: {args.model}...")
    ner = pipeline("token-classification", model=args.model, device=-1)

    LABEL2ID = {"O": 0, "B-BOUNDARY": 1, "I-BOUNDARY": 2}
    all_preds, all_labels, times = [], [], []

    for item in data:
        text = " ".join(item["tokens"])[:2000]
        start = time.time()
        result = ner(text)
        times.append(time.time() - start)

        pred_labels = [LABEL2ID.get(r.get("entity", "O"), 0) for r in result]
        true_labels = [LABEL2ID.get(l, 0) for l in item["labels"][:len(pred_labels)]]
        all_preds.extend(pred_labels)
        all_labels.extend(true_labels)

    f1 = f1_score(all_labels, all_preds, average="weighted")
    p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

    print(f"\n── AXOTEST Edge Case Detector Evaluation ───────")
    print(f"  F1 (weighted): {f1*100:.1f}%")
    print(f"  Precision:      {p*100:.1f}%")
    print(f"  Recall:         {r*100:.1f}%")
    print(f"  Avg inference:  {sum(times)/len(times)*1000:.1f}ms")
    print(f"────────────────────────────────────────────────\n")

    if f1 >= 0.80:
        print(f"  ✓ PASS — F1 {f1*100:.1f}% ≥ 80% target")
    else:
        print(f"  ✗ FAIL — F1 {f1*100:.1f}% < 80% target")

if __name__ == "__main__":
    main()
