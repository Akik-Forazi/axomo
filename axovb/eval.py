#!/usr/bin/env python3
"""
AXOVB eval — evaluate the trained PEAK MLP on the held-out test set.

Replaces the old transformers-pipeline-based eval (which didn't apply to
the custom MLP). Loads bump_mlp.onnx + metadata.json, runs inference on
test.jsonl, and reports accuracy / precision / recall / F1 / confusion
matrix / per-class confidence / inference speed.

Usage:
    python -m axovb eval
    python -m axovb eval --model-dir path/to/models --data path/to/test.jsonl
    python axovb/eval.py
"""
import argparse, os, sys, json, time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axovb.predictor import BumpPredictor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl


def extract_features_for_test_item(item: Dict, feature_names: List[str]) -> List[float]:
    """Extract the 20-feature vector from a test.jsonl item, applying the same
    derived-feature logic train.py uses (so test data matches training data)."""
    f = item.get("features", item)
    row = [f.get(name, 0) for name in feature_names]
    # Derive the v3 PEAK features if missing (same logic as train.py)
    if "avg_callers_per_symbol" not in f:
        row[16] = round(f.get("num_callers_affected", 0) / max(f.get("num_changed_symbols", 1), 1), 2)
    if "export_change_ratio" not in f:
        exported = f.get("has_removed_exports", 0) + f.get("has_new_exports", 0) + f.get("has_exported_changes", 0)
        row[17] = round(exported / max(f.get("num_changed_symbols", 1), 1), 2)
    if "breaking_score" not in f:
        row[18] = f.get("has_removed_exports", 0) * 3 + f.get("has_signature_changes", 0) * 2 + min(f.get("num_callers_affected", 0), 50) / 50
    if "complexity_score" not in f:
        row[19] = (f.get("num_source_files", 0) + f.get("num_changed_symbols", 0) + min(f.get("num_callers_affected", 0), 50)) / 100
    return row


def evaluate_on_test_set(model_dir: str, data_path: str):
    """Load the ONNX model, run on the test set, print metrics."""
    print(f"  Loading ONNX model from {model_dir}...")
    predictor = BumpPredictor(model_dir)
    info = predictor.info()
    print(f"    feature count: {info['feature_count']}")
    print(f"    labels:        {info['labels']}")
    print(f"    onnx size:     {info['onnx_size_kb']} KB")

    data = load_jsonl(data_path)
    print(f"\n  Loaded {len(data)} test samples from {data_path}")

    # Build feature matrix + true labels
    import numpy as np
    from collections import Counter
    X_list = []
    y_true = []
    for item in data:
        feats = extract_features_for_test_item(item, predictor.feature_names)
        X_list.append(feats)
        # true label
        lbl = item.get("label", "none")
        y_true.append(predictor.label2id.get(lbl, 0))

    X = np.array(X_list, dtype=np.float32)
    y_true_np = np.array(y_true, dtype=np.int64)
    label_dist = Counter(y_true)
    print(f"  Label distribution: {dict((predictor.id2label[k], v) for k, v in label_dist.items())}")

    # Run inference in batch (single ONNX call)
    t0 = time.time()
    probs = predictor.sess.run([predictor.output_name], {predictor.input_name: X})[0]
    elapsed = time.time() - t0
    preds = probs.argmax(axis=1)

    # Metrics
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
    )
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="weighted", zero_division=0,
    )
    cm = confusion_matrix(y_true, preds, labels=list(range(len(predictor.id2label))))

    print(f"\n  {'─' * 60}")
    print(f"  {__import__('axovb').__version__ if False else 'AXOVB'} EVAL — PEAK MLP (ONNX)")
    print(f"  {'─' * 60}")
    print(f"  Accuracy:    {acc*100:.1f}%")
    print(f"  F1 (wtd):    {f1*100:.1f}%")
    print(f"  Precision:   {precision*100:.1f}%")
    print(f"  Recall:      {recall*100:.1f}%")
    print(f"  Params:      2,806,020 (2.8M)")
    print(f"  Model size:  ~10.7 MB")
    print(f"  Inference:   {elapsed*1000:.1f}ms total ({elapsed/len(data)*1000:.3f}ms/sample)")
    print(f"  Throughput:  {len(data)/elapsed:.0f} samples/sec")
    print(f"\n  Confusion matrix:")
    label_names = [predictor.id2label[i] for i in range(len(predictor.id2label))]
    print(f"    {'':>8} " + " ".join(f"{n:>8}" for n in label_names))
    for i, row in enumerate(cm):
        print(f"    {label_names[i]:>8} " + " ".join(f"{v:>8}" for v in row))

    # Per-class confidence
    print(f"\n  Per-class confidence (avg max prob):")
    for i, name in enumerate(label_names):
        mask = preds == i
        if mask.sum() > 0:
            avg_conf = probs[mask, i].mean()
            print(f"    {name:<8} {avg_conf*100:6.2f}%  (n={mask.sum()})")

    # Sample predictions
    print(f"\n  Sample predictions (first 5):")
    for i in range(min(5, len(data))):
        true_l = predictor.id2label[y_true[i]]
        pred_l = predictor.id2label[int(preds[i])]
        conf = float(probs[i][preds[i]])
        marker = "✓" if true_l == pred_l else "✗"
        print(f"    {marker} sample {i}: pred={pred_l:<8} true={true_l:<8} conf={conf*100:5.1f}%")

    print(f"\n  {'─' * 60}")
    target = 0.90
    if acc >= target:
        print(f"  ✓ PASS — accuracy {acc*100:.1f}% ≥ {target*100:.0f}% target")
    else:
        print(f"  ✗ FAIL — accuracy {acc*100:.1f}% < {target*100:.0f}% target")
    print()


def main():
    parser = argparse.ArgumentParser(description="AXOVB eval — evaluate the trained ONNX model on the test set")
    parser.add_argument("--model-dir", default=os.path.join(os.path.dirname(__file__), "models"))
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "test.jsonl"))
    args = parser.parse_args()
    evaluate_on_test_set(args.model_dir, args.data)


if __name__ == "__main__":
    main()
