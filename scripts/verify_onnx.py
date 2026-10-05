#!/usr/bin/env python3
"""Verify ONNX inference for the trained bumper MLP."""
import os, sys, json, time
sys.path.insert(0, '/home/z/my-project/axomo/shared')
from utils import load_jsonl
import numpy as np

def main():
    model_dir = '/home/z/my-project/axomo/axovb/models'
    data_path = '/home/z/my-project/axomo/axovb/data/test.jsonl'

    import onnxruntime as ort

    with open(os.path.join(model_dir, 'metadata.json')) as f:
        meta = json.load(f)
    feature_names = meta['feature_names']
    X_min = np.array(meta['feature_min'], dtype=np.float32)
    X_max = np.array(meta['feature_max'], dtype=np.float32)
    id2label = {int(k): v for k, v in meta['id2label'].items()}

    print(f"Loading ONNX: {os.path.join(model_dir, 'bump_mlp.onnx')}")
    sess = ort.InferenceSession(os.path.join(model_dir, 'bump_mlp.onnx'))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"  Input:  {input_name} {sess.get_inputs()[0].shape}")
    print(f"  Output: {output_name} {sess.get_outputs()[0].shape}")

    data = load_jsonl(data_path)
    print(f"\nLoaded {len(data)} test samples")

    FEATURE_NAMES = feature_names
    X_list = []
    y_true = []
    LABEL2ID = {"none": 0, "patch": 1, "minor": 2, "major": 3}
    for item in data:
        f = item.get('features', item)
        row = [f.get(name, 0) for name in FEATURE_NAMES]
        total_files = f.get('total_files', f.get('num_source_files', 0) + f.get('num_test_files', 0) + f.get('num_doc_files', 0) + f.get('num_config_files', 0))
        if 'avg_callers_per_symbol' not in f:
            row[16] = round(f.get('num_callers_affected', 0) / max(f.get('num_changed_symbols', 1), 1), 2)
        if 'export_change_ratio' not in f:
            exported = f.get('has_removed_exports', 0) + f.get('has_new_exports', 0) + f.get('has_exported_changes', 0)
            row[17] = round(exported / max(f.get('num_changed_symbols', 1), 1), 2)
        if 'breaking_score' not in f:
            row[18] = f.get('has_removed_exports', 0) * 3 + f.get('has_signature_changes', 0) * 2 + min(f.get('num_callers_affected', 0), 50) / 50
        if 'complexity_score' not in f:
            row[19] = (f.get('num_source_files', 0) + f.get('num_changed_symbols', 0) + min(f.get('num_callers_affected', 0), 50)) / 100
        X_list.append(row)
        y_true.append(LABEL2ID.get(item['label'], 0))

    X = np.array(X_list, dtype=np.float32)
    # Pass RAW features to ONNX — the BumpMLPInference wrapper normalizes internally
    # (double-normalizing would produce garbage predictions)

    t0 = time.time()
    probs = sess.run([output_name], {input_name: X})[0]
    elapsed = time.time() - t0
    preds = probs.argmax(axis=1)

    acc = (preds == y_true).mean()
    print(f"\n=== ONNX Inference Results ===")
    print(f"  Samples: {len(X)}")
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  Total time: {elapsed*1000:.1f}ms ({elapsed/len(X)*1000:.2f}ms/sample)")
    print(f"  Throughput: {len(X)/elapsed:.0f} samples/sec")

    print(f"\n=== Sample predictions ===")
    for i in range(min(5, len(data))):
        conf = float(probs[i][preds[i]])
        true_l = id2label[y_true[i]]
        pred_l = id2label[int(preds[i])]
        print(f"  sample {i}: pred={pred_l}  true={true_l}  conf={conf*100:.1f}%")

    print(f"\n=== ONNX model summary ===")
    print(f"  File: {os.path.join(model_dir, 'bump_mlp.onnx')}")
    print(f"  Size: {os.path.getsize(os.path.join(model_dir, 'bump_mlp.onnx'))//1024} KB")
    print(f"  Input: {len(feature_names)} features")
    print(f"  Output: 4 probabilities (none, patch, minor, major)")

if __name__ == '__main__':
    main()
