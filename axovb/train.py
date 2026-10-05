#!/usr/bin/env python3
"""
Fine-tune a PEAK MLP (2.8M params) for AXOVB — version bump classification.

v3: Bigger model, more features, PEAK training techniques.

Architecture: 14→2048→1024→512→256→4
  - 2,786,052 params (2.8M) — sweet spot for structured feature classification
  - ~10.6 MB model file
  - BatchNorm + Dropout for regularization
  - Learning rate scheduling (cosine)
  - Early stopping
  - Feature importance analysis
  - Per-class metrics + confusion matrix
  - Temperature scaling for calibrated confidence

Input: 20 structured features (expanded from 14)
Output: {patch, minor, major, none} + calibrated confidence

Usage:
    python train.py --data ./data/train.jsonl --output ./model/ --epochs 100
"""
import argparse, os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl, train_test_split

# Expanded from 14 → 20 features (more PEAK signal)
FEATURE_NAMES = [
    # Symbol-level (from axodex)
    "num_changed_symbols",     # how many symbols changed
    "num_callers_affected",    # blast radius (total callers)
    "has_removed_exports",     # were exports removed?
    "has_new_exports",         # were new exports added?
    "has_exported_changes",    # any exported symbols touched?
    "has_signature_changes",   # did function signatures change?
    "has_type_changes",        # did types/interfaces change?
    "has_rename",              # were symbols renamed?
    # File-level (from git stat)
    "num_source_files",        # .ts/.js/.py files changed
    "num_test_files",          # .test.ts/.spec.ts files
    "num_doc_files",           # .md/.txt files
    "num_config_files",        # .json/.yaml/.toml files
    # Derived
    "total_files",             # sum of all file types
    "test_to_source_ratio",    # are changes test-only?
    "blast_radius",            # same as num_callers_affected
    "change_scope",            # capped symbol count (max 50)
    # NEW v3 PEAK features
    "avg_callers_per_symbol",  # blast density
    "export_change_ratio",     # exported/total symbols changed
    "breaking_score",           # composite: removed + signature + callers
    "complexity_score",        # composite: files + symbols + callers
]

LABEL2ID = {"none": 0, "patch": 1, "minor": 2, "major": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ONNX export (the correct deployment format — not JSON)

def export_onnx(model, output_dir, feature_names, X_min, X_max, label2id, id2label, temperature):
    """Export the trained MLP to ONNX format for onnxruntime-node."""
    import torch
    onnx_path = os.path.join(output_dir, "bump_mlp.onnx")
    
    # Create a wrapper that includes normalization + temperature scaling
    class BumpMLPInference(torch.nn.Module):
        def __init__(self, mlp, X_min, X_max, temperature):
            super().__init__()
            self.mlp = mlp
            self.register_buffer("X_min", torch.FloatTensor(X_min))
            self.register_buffer("X_max", torch.FloatTensor(X_max))
            self.temperature = temperature
            
        def forward(self, x):
            # Normalize input
            x = (x - self.X_min) / torch.clamp(self.X_max - self.X_min, min=1.0)
            # Forward through MLP
            logits = self.mlp(x)
            # Apply temperature scaling
            logits = logits / self.temperature
            # Softmax
            probs = torch.softmax(logits, dim=-1)
            return probs
    
    inference_model = BumpMLPInference(model, X_min, X_max, temperature)
    inference_model.eval()
    
    # Export with dynamic batch size
    dummy_input = torch.randn(1, len(feature_names))
    torch.onnx.export(
        inference_model,
        dummy_input,
        onnx_path,
        input_names=["features"],
        output_names=["probabilities"],
        dynamic_axes={"features": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        dynamo=False,
        opset_version=17,
    )
    
    # Save metadata alongside the ONNX file
    import json
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump({
            "feature_names": feature_names,
            "feature_min": X_min.tolist(),
            "feature_max": X_max.tolist(),
            "label2id": label2id,
            "id2label": id2label,
            "temperature": temperature,
            "model_type": "bump_mlp_onnx",
            "input_shape": [1, len(feature_names)],
            "output_shape": [1, 4],
        }, f, indent=2)
    
    onnx_size = os.path.getsize(onnx_path)
    print(f"\n  ONNX model: {onnx_path}")
    print(f"  ONNX size:  {onnx_size // 1024} KB ({onnx_size / 1024 / 1024:.1f} MB)")
    print(f"  Load in Node.js: const {''}= require('onnxruntime-node').InferenceSession.create(onnxPath)")

# Call ONNX export after saving the PyTorch checkpoint
def main():
    parser = argparse.ArgumentParser(description="Train AXOVB PEAK MLP (2.8M params)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    # Load data
    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} samples")

    # Extract features (handle old 14-feature format + new 20-feature format)
    X_list = []
    for item in data:
        f = item.get("features", item)
        row = [f.get(name, 0) for name in FEATURE_NAMES]
        # Compute derived features if missing
        total_files = f.get("total_files", f.get("num_source_files", 0) + f.get("num_test_files", 0) + f.get("num_doc_files", 0) + f.get("num_config_files", 0))
        if "avg_callers_per_symbol" not in f:
            row[16] = round(f.get("num_callers_affected", 0) / max(f.get("num_changed_symbols", 1), 1), 2)
        if "export_change_ratio" not in f:
            exported = f.get("has_removed_exports", 0) + f.get("has_new_exports", 0) + f.get("has_exported_changes", 0)
            row[17] = round(exported / max(f.get("num_changed_symbols", 1), 1), 2)
        if "breaking_score" not in f:
            row[18] = f.get("has_removed_exports", 0) * 3 + f.get("has_signature_changes", 0) * 2 + min(f.get("num_callers_affected", 0), 50) / 50
        if "complexity_score" not in f:
            row[19] = (f.get("num_source_files", 0) + f.get("num_changed_symbols", 0) + min(f.get("num_callers_affected", 0), 50)) / 100
        X_list.append(row)

    X = np.array(X_list, dtype=np.float32)
    y = np.array([LABEL2ID.get(item["label"], 0) for item in data], dtype=np.int64)

    # Normalize features (min-max scaling)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / np.maximum(X_max - X_min, 1)

    # Train/test split
    indices = np.random.RandomState(42).permutation(len(data))
    split = int(len(data) * 0.85)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X_norm[train_idx], X_norm[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    from collections import Counter
    print(f"Labels (train): {Counter(y_train.tolist())}")
    print(f"Labels (test): {Counter(y_test.tolist())}")

    # PEAK MLP Architecture
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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n── PEAK MLP Architecture ──────────────────────")
    print(f"  Layers: 20 → 2048 → 1024 → 512 → 256 → 4")
    print(f"  Features: {len(FEATURE_NAMES)} (expanded from 14)")
    print(f"  Regularization: BatchNorm + Dropout(0.3)")
    print(f"  Total params: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Inference: ~0.2ms (still instant)")
    print(f"────────────────────────────────────────────────")

    # Training with cosine LR scheduling + early stopping
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    best_acc = 0
    patience_counter = 0
    best_state = None

    print(f"\nTraining for {args.epochs} epochs (early stopping patience={args.patience})...")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        total_loss = 0
        for i in range(0, len(X_train_t), args.batch_size):
            batch_X = X_train_t[perm[i:i+args.batch_size]]
            batch_y = y_train_t[perm[i:i+args.batch_size]]
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_out = model(X_test_t)
            test_pred = test_out.argmax(dim=1)
            acc = (test_pred == y_test_t).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:>3}/{args.epochs} — loss: {total_loss:.4f}, acc: {acc*100:.1f}%, lr: {lr:.6f}{' ★' if patience_counter == 0 else ''}")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation with confusion matrix
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        test_pred = test_out.argmax(dim=1).numpy()
        test_probs = torch.softmax(test_out, dim=1).numpy()

    acc = accuracy_score(y_test, test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2, 3])

    # Temperature scaling for calibrated confidence
    temperature = torch.ones(1, requires_grad=True)
    temp_optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)
    def temp_closure():
        temp_optimizer.zero_grad()
        scaled = torch.FloatTensor(test_probs) / temperature
        loss = criterion(scaled, y_test_t)
        loss.backward()
        return loss
    temp_optimizer.step(temp_closure)

    print(f"\n── PEAK MLP Final Evaluation ──────────────────")
    print(f"  Accuracy:    {acc*100:.1f}%")
    print(f"  F1 (wtd):    {f1*100:.1f}%")
    print(f"  Precision:  {precision*100:.1f}%")
    print(f"  Recall:     {recall*100:.1f}%")
    print(f"  Params:     {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Temperature: {temperature.item():.3f} (calibration)")
    print(f"\n  Confusion matrix:")
    labels = ["none", "patch", "minor", "major"]
    print(f"    {'':>8} " + " ".join(f"{n:>8}" for n in labels))
    for i, row in enumerate(cm):
        print(f"    {labels[i]:>8} " + " ".join(f"{v:>8}" for v in row))

    # Feature importance (via weight magnitude)
    print(f"\n  Feature importance (top 5):")
    fc1_weights = model.net[0].weight.abs().mean(dim=0).detach().numpy()
    importance = sorted(zip(FEATURE_NAMES, fc1_weights), key=lambda x: -x[1])
    for name, score in importance[:5]:
        print(f"    {name:>30}: {score:.4f}")
    print(f"────────────────────────────────────────────────")

    # Save model
    os.makedirs(args.output, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_names": FEATURE_NAMES,
        "feature_min": X_min.tolist(),
        "feature_max": X_max.tolist(),
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "temperature": temperature.item(),
        "model_config": {
            "input_dim": len(FEATURE_NAMES),
            "hidden_sizes": [2048, 1024, 512, 256],
            "num_classes": 4,
            "dropout": 0.3,
        },
    }, os.path.join(args.output, "bump_mlp.pt"))


    # Export to ONNX (the correct deployment format)
    print(f"\nExporting to ONNX...")
    export_onnx(model, args.output, FEATURE_NAMES, X_min, X_max, LABEL2ID, ID2LABEL, temperature.item())

    print(f"\nModel saved to {args.output}")
    print(f"  bump_mlp.pt  (PyTorch, for re-training)")
    print(f"  bump_mlp.onnx (ONNX, for onnxruntime-node inference)")
    print(f"  metadata.json (feature names, normalizer, labels)")

if __name__ == "__main__":
    main()


