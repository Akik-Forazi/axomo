"""
AXOVB predictor — ONNX inference wrapper for the trained PEAK MLP bumper.

Loads `bump_mlp.onnx` + `metadata.json`, accepts a 20-feature vector (or a
git diff string), and returns the predicted bump label + confidence +
per-class probabilities.

The 20-feature vector is the same one train.py extracts:
    num_changed_symbols, num_callers_affected, has_removed_exports, ...
"""
import os, sys, json
from typing import Dict, Tuple, Optional, List
import numpy as np

# Reuse the diff feature extractor from axovb.analyzer
sys.path.insert(0, os.path.dirname(__file__))
from analyzer import extract_diff_features


# Canonical feature order (must match train.py's FEATURE_NAMES)
FEATURE_NAMES = [
    "num_changed_symbols", "num_callers_affected", "has_removed_exports", "has_new_exports",
    "has_exported_changes", "has_signature_changes", "has_type_changes", "has_rename",
    "num_source_files", "num_test_files", "num_doc_files", "num_config_files",
    "total_files", "test_to_source_ratio", "blast_radius", "change_scope",
    "avg_callers_per_symbol", "export_change_ratio", "breaking_score", "complexity_score",
]


class BumpPredictor:
    """Load the trained ONNX bumper once, run inference many times."""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "models")
        self.model_dir = model_dir
        self.onnx_path = os.path.join(model_dir, "bump_mlp.onnx")
        self.meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found at {self.onnx_path}. "
                f"Run `python axovb/train.py --data axovb/data/train.jsonl --output axovb/models/` first."
            )
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"metadata.json not found at {self.meta_path}")

        with open(self.meta_path) as f:
            self.meta = json.load(f)
        self.feature_names = self.meta["feature_names"]
        self.X_min = np.array(self.meta["feature_min"], dtype=np.float32)
        self.X_max = np.array(self.meta["feature_max"], dtype=np.float32)
        self.id2label = {int(k): v for k, v in self.meta["id2label"].items()}
        self.label2id = {v: int(k) for k, v in self.meta["id2label"].items()}

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            ) from e
        self.sess = ort.InferenceSession(self.onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict_features(self, features: Dict) -> Tuple[str, float, List[float]]:
        """Predict from a 20-key feature dict (keys matching FEATURE_NAMES).
        Returns (label, confidence, [prob_none, prob_patch, prob_minor, prob_major]).
        """
        x = np.array(
            [[features.get(n, 0) for n in self.feature_names]],
            dtype=np.float32,
        )
        # IMPORTANT: pass RAW features — the BumpMLPInference ONNX wrapper
        # normalizes input internally. Pre-normalizing here would double-normalize.
        probs = self.sess.run([self.output_name], {self.input_name: x})[0]
        pred = int(probs.argmax(axis=1)[0])
        conf = float(probs[0][pred])
        return self.id2label[pred], conf, probs[0].tolist()

    def predict_diff(self, diff: str) -> Tuple[str, float, List[float], Dict]:
        """Predict from a raw git diff string. Extracts features, then predicts.
        Returns (label, confidence, probs, features_dict).
        """
        features, _, _ = extract_diff_features(diff)
        label, conf, probs = self.predict_features(features)
        return label, conf, probs, features

    def predict_batch(self, feature_dicts: List[Dict]) -> List[Tuple[str, float]]:
        """Batch predict for many feature dicts at once."""
        X = np.array(
            [[fd.get(n, 0) for n in self.feature_names] for fd in feature_dicts],
            dtype=np.float32,
        )
        probs = self.sess.run([self.output_name], {self.input_name: X})[0]
        preds = probs.argmax(axis=1)
        results = []
        for i, p in enumerate(preds):
            results.append((self.id2label[int(p)], float(probs[i][p])))
        return results

    def info(self) -> Dict:
        """Return model metadata as a dict (for `info` subcommand)."""
        return {
            "onnx_path": self.onnx_path,
            "onnx_size_kb": os.path.getsize(self.onnx_path) // 1024,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "labels": [self.id2label[i] for i in range(len(self.id2label))],
            "feature_min": self.X_min.tolist(),
            "feature_max": self.X_max.tolist(),
        }


# Convenience function for one-off predictions (loads model fresh each call —
# for batch use, instantiate BumpPredictor once and reuse).
def predict_diff(diff: str, model_dir: str = None) -> Tuple[str, float, Dict]:
    if not hasattr(predict_diff, "_predictor") or predict_diff._predictor is None:
        predict_diff._predictor = BumpPredictor(model_dir)
    label, conf, _, features = predict_diff._predictor.predict_diff(diff)
    return label, conf, features


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AXOVB predictor — quick prediction from a diff file")
    p.add_argument("--diff", required=True, help="Path to a .patch file")
    p.add_argument("--model-dir", default=None)
    args = p.parse_args()
    with open(args.diff) as f:
        diff = f.read()
    label, conf, features = predict_diff(diff, args.model_dir)
    print(f"Predicted bump: {label.upper()}  (confidence: {conf*100:.1f}%)")
    print(f"Features: {json.dumps(features, indent=2)}")
