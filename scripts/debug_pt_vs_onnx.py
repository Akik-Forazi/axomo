#!/usr/bin/env python3
"""Debug: load .pt model directly, run inference, compare with ONNX output."""
import os, sys, json
sys.path.insert(0, '/home/z/my-project/axomo/shared')
from utils import load_jsonl
import numpy as np
import torch

# Load .pt checkpoint
ckpt = torch.load('/home/z/my-project/axomo/axovb/models/bump_mlp.pt', map_location='cpu', weights_only=False)
feature_names = ckpt['feature_names']
X_min = np.array(ckpt['feature_min'], dtype=np.float32)
X_max = np.array(ckpt['feature_max'], dtype=np.float32)
temperature = ckpt['temperature']
print(f"Saved temperature: {temperature}")
print(f"Feature count: {len(feature_names)}")
print(f"feature_min[:5]: {X_min[:5]}")
print(f"feature_max[:5]: {X_max[:5]}")

# Rebuild model architecture (matches train.py BumpMLP)
import torch.nn as nn
class BumpMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_sizes=[2048, 1024, 512, 256], num_classes=4, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = BumpMLP()
model.load_state_dict(ckpt['model_state'])
model.eval()

# Load test data + extract features (same logic as verify_onnx.py)
data = load_jsonl('/home/z/my-project/axomo/axovb/data/test.jsonl')
LABEL2ID = {"none": 0, "patch": 1, "minor": 2, "major": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

X_list = []
y_true = []
for item in data:
    f = item.get('features', item)
    row = [f.get(name, 0) for name in feature_names]
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
print(f"\nTest X shape: {X.shape}")
print(f"First test row (raw features): {X[0]}")
print(f"True label of first sample: {ID2LABEL[y_true[0]]}")

# Normalize (matches train.py normalization)
X_norm_pt = (X - X_min) / np.maximum(X_max - X_min, 1.0)
print(f"\nFirst test row (normalized): {X_norm_pt[0]}")

# Run through PyTorch model
with torch.no_grad():
    x_t = torch.FloatTensor(X_norm_pt)
    logits = model(x_t)
    print(f"\nRaw logits shape: {logits.shape}")
    print(f"First sample raw logits: {logits[0]}")
    scaled_logits = logits / temperature
    print(f"First sample scaled logits (T={temperature}): {scaled_logits[0]}")
    probs_pt = torch.softmax(scaled_logits, dim=1).numpy()

preds_pt = probs_pt.argmax(axis=1)
acc_pt = (preds_pt == y_true).mean()
print(f"\nPyTorch direct inference:")
print(f"  Accuracy: {acc_pt*100:.1f}%")
print(f"  Sample 0: pred={ID2LABEL[preds_pt[0]]} true={ID2LABEL[y_true[0]]} conf={probs_pt[0][preds_pt[0]]*100:.1f}%")
print(f"  Sample 0 probs: {probs_pt[0]}")

# Now run ONNX with RAW features (not normalized)
print(f"\n=== ONNX with RAW features ===")
import onnxruntime as ort
sess = ort.InferenceSession('/home/z/my-project/axomo/axovb/models/bump_mlp.onnx')
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
probs_onnx_raw = sess.run([output_name], {input_name: X})[0]
preds_onnx_raw = probs_onnx_raw.argmax(axis=1)
acc_onnx_raw = (preds_onnx_raw == y_true).mean()
print(f"  Accuracy: {acc_onnx_raw*100:.1f}%")
print(f"  Sample 0: pred={ID2LABEL[preds_onnx_raw[0]]} true={ID2LABEL[y_true[0]]} conf={probs_onnx_raw[0][preds_onnx_raw[0]]*100:.1f}%")
print(f"  Sample 0 probs: {probs_onnx_raw[0]}")

# Now run ONNX with NORMALIZED features (double-normalize — what verify_onnx was doing)
print(f"\n=== ONNX with NORMALIZED features (double-normalize) ===")
probs_onnx_norm = sess.run([output_name], {input_name: X_norm_pt.astype(np.float32)})[0]
preds_onnx_norm = probs_onnx_norm.argmax(axis=1)
acc_onnx_norm = (preds_onnx_norm == y_true).mean()
print(f"  Accuracy: {acc_onnx_norm*100:.1f}%")
print(f"  Sample 0: pred={ID2LABEL[preds_onnx_norm[0]]} true={ID2LABEL[y_true[0]]} conf={probs_onnx_norm[0][preds_onnx_norm[0]]*100:.1f}%")
print(f"  Sample 0 probs: {probs_onnx_norm[0]}")
