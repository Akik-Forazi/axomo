"""
AXOVB explainer — distilgpt2 narrative generation wrapper.

Loads the locally-trained distilgpt2 explainer (or falls back to a
rule-based narrative if the model is missing). The model takes a 5-key
feature subset (the most informative ones) + the predicted bump type
and generates 1-3 sentences of natural-language explanation.
"""
import os, json
from typing import Dict, Tuple, Optional

# Import the rule-based narrative as fallback
from analyzer import rule_based_narrative


class Explainer:
    """Loads the trained distilgpt2 explainer; falls back to rule-based if missing."""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "explainer")
        self.model_dir = model_dir
        self.config_path = os.path.join(model_dir, "config.json")
        self._tok = None
        self._model = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if not os.path.exists(self.config_path):
            return  # stay None — fall back to rule-based
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            return
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._loaded = True

    def generate(self, features: Dict, removed_lines=None, sec_findings=None,
                edge_findings=None, blast=None, bump_label: str = "major",
                max_new_tokens: int = 80, temperature: float = 0.7) -> Tuple[str, float, bool]:
        """Generate a narrative. Returns (text, duration_ms, used_ml).
        Falls back to rule-based if the model isn't available.
        """
        removed_lines = removed_lines or []
        sec_findings = sec_findings or []
        edge_findings = edge_findings or []
        blast = blast or {}

        # Always compute the rule-based narrative as a guaranteed-coherent fallback
        rule_text = rule_based_narrative(features, removed_lines, sec_findings, edge_findings, blast)

        self._load()
        if not self._loaded:
            return rule_text, 0.0, False

        # Try ML generation
        import time
        prompt = (
            f'Features: {json.dumps({k: v for k, v in features.items() if k in ["num_changed_symbols","num_callers_affected","has_removed_exports","has_new_exports","has_signature_changes"]}, separators=(",", ": "))}\n'
            f'Bump: {bump_label}\nExplanation:'
        )
        import torch
        inputs = self._tok(prompt, return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tok.eos_token_id,
            )
        elapsed = (time.time() - t0) * 1000
        text = self._tok.decode(out[0], skip_special_tokens=True)
        response = text[len(prompt):].strip()
        return response, elapsed, True

    def info(self) -> Dict:
        """Return model metadata for the `info` subcommand."""
        info = {"model_dir": self.model_dir, "loaded": self._loaded}
        if os.path.exists(self.config_path):
            info["config"] = "exists"
            info["size_mb"] = sum(
                os.path.getsize(os.path.join(self.model_dir, f))
                for f in os.listdir(self.model_dir)
                if os.path.isfile(os.path.join(self.model_dir, f))
            ) / 1024 / 1024
            metrics_path = os.path.join(self.model_dir, "training_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    info["training_metrics"] = json.load(f)
        else:
            info["config"] = "missing (will use rule-based fallback)"
        return info


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AXOVB explainer — generate narrative from features")
    p.add_argument("--features", required=True, help="JSON file with feature dict")
    p.add_argument("--bump", default="major")
    args = p.parse_args()
    with open(args.features) as f:
        features = json.load(f)
    expl = Explainer()
    text, ms, used_ml = expl.generate(features, bump_label=args.bump)
    print(f"Used ML: {used_ml}")
    print(f"Duration: {ms:.0f}ms")
    print(f"\n{text}")
