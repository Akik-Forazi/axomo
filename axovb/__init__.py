"""
AXOVB — AI-powered semantic version bumper.

A trained PEAK MLP (2.8M params, exported as ONNX) predicts the correct
bump type (none/patch/minor/major) from a git diff. A locally-trained
distilgpt2 explainer (82M params) generates a natural-language narrative
of why the bump is needed. Project-wide version files are then edited
atomically across 8 ecosystems with git commit + tag.

Package layout:
    axovb/
    ├── __init__.py        — package metadata + version
    ├── __main__.py        — `python -m axovb` entry point
    ├── cli.py             — subcommand dispatcher (predict/apply/auto/eval/info/list)
    ├── bumper.py          — version-file detectors + atomic writer + git integration
    ├── analyzer.py        — diff feature extractor + static security/edge/blast analysis
    ├── predictor.py       — ONNX inference wrapper for the PEAK MLP
    ├── explainer.py       — distilgpt2 narrative generation
    ├── eval.py            — model evaluation on held-out test set (ONNX-based)
    ├── train.py           — train the PEAK MLP + export ONNX
    ├── train_explainer.py — train the distilgpt2 explainer (legacy, full-memory)
    ├── prepare_data.py    — parse bumper training data from cloned repos
    ├── prepare_explainer_data.py — parse explainer training data (synthetic)
    ├── model_config.json  — describes the deployed model artifacts
    ├── data/              — parsed training data (.jsonl, preserved)
    ├── models/            — trained ONNX + .pt + metadata.json
    └── explainer/         — trained distilgpt2 (safetensors gitignored — regenerable)

Quick start:
    # Show what version files exist in the current repo
    python -m axovb list

    # Predict the bump type from staged git diff
    python -m axovb predict

    # Apply a major bump across every version file + commit + tag
    python -m axovb apply --major --commit --tag

    # Predict + apply in one shot
    python -m axovb auto --commit --tag

    # Evaluate the trained model on the held-out test set
    python -m axovb eval

    # Show model artifacts + their sizes
    python -m axovb info
"""
__version__ = "0.3.0"
__all__ = ["__version__"]
