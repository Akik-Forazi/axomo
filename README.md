# AXOMO — Axo Models

**Fine-tuned sub-200M-parameter models for the Fraziym ecosystem — trained locally, exported to ONNX, run on CPU. No API calls.**

V00.01.000-beta-01 · FRAZIYM versioning

## What is AXOMO?

AXOMO contains the training scripts, evaluation tools, and exported models for the specialized small models used by AXOVB (semantic version bumping) and AXOTEST (test intelligence). Each model is purpose-built for ONE task — not a general-purpose chatbot.

## Model registry

A **class** (`c<N>`) is defined by the DATASET a checkpoint was trained on; one architecture can ship several classes side by side. Class naming: `<task>-v<semver>-c<N>`.

| Model class | For | Arch | Params | Task | Status |
|---|---|---|---|---|---|
| `semver-v1-c0` *(default)* | AXOVB | PEAK MLP | 2.8M | git diff → {none\|patch\|minor\|major} | Trained — 100% on its legacy synthetic set, **32.6% on real repos** (legacy class, kept for compatibility) |
| `semver-v1-c1` | AXOVB | PEAK MLP | 2.8M | git diff → {none\|patch\|minor\|major} | Trained — **93.2%** on v3 held-out, **78.3% on real repos** — recommended for real use |
| `semver-v1-c0-backup` | AXOVB | PEAK MLP | 2.8M | — | Archive of the original pre-rename c0 artifacts |
| `explainer-v1-c0` | AXOVB | distilgpt2 | 82M | features + bump → natural-language narrative | Trained 1 epoch (undertrained — rule-based narrative remains primary) |
| `axotest-security` | AXOTEST | DistilBERT | 66M | code → {secure\|vulnerable} | Dataset ready (504/89) — not trained yet |
| `axotest-edge` | AXOTEST | DistilBERT | 66M | code → boundary tokens | Dataset ready (1005/178) — not trained yet |
| `axotest-unit` | AXOTEST | CodeT5-small | 60M | source → test code | Blocked — data prep yielded 0 training pairs |

Full numbers, hashes and methodology: [`benchmark-axovb.md`](benchmark-axovb.md). Registry: [`axovb/model_config.json`](axovb/model_config.json).

## Structure

```
axomo/
├── axovb/                      # AXOVB — semantic version bumper
│   ├── analyzer.py             # diff feature extractor (regex) + security/edge/blast scans
│   ├── bumper.py               # version-file detectors (8 ecosystems) + atomic writer + git
│   ├── predictor.py            # ONNX inference wrapper (NaN-guarded)
│   ├── explainer.py            # distilgpt2 narrative generation + rule-based fallback
│   ├── cli.py, __main__.py     # `python -m axovb` (predict/apply/auto/eval/info/list)
│   ├── train.py                # train the PEAK MLP + export ONNX
│   ├── eval.py, benchmark.py   # evaluation + latency/calibration/robustness suite
│   ├── prepare_data.py         # build (diff → bump) datasets from real repos + synthetic
│   ├── data/                   # preserved training data (.jsonl — do not re-parse)
│   ├── semver-v1-c0/           # deployed default class (ONNX + .pt + metadata.json)
│   ├── semver-v1-c1/           # v3 dataset class — recommended for real repos
│   ├── semver-v1-c0-backup/    # archive of the original c0 bytes
│   └── explainer-v1-c0/        # fine-tuned distilgpt2 (weights gitignored — regenerable)
├── axotest-security/           # AXOTEST security scanner (dataset ready, untrained)
├── axotest-edge/               # AXOTEST edge-case detector (dataset ready, untrained)
├── axotest-unit/               # AXOTEST unit-test generator (blocked — re-prep needed)
├── shared/
│   ├── utils.py                # shared data utilities
│   └── export_onnx.py          # export any model to ONNX
├── scripts/                    # demo/eval/benchmark helpers
└── requirements.txt            # Python deps (torch, transformers, onnx, onnxruntime)
```

## Quick start

```bash
# Needs Python with torch + onnxruntime (this machine: py -3.14)
py -3.14 -m axovb info                    # model classes + sizes
py -3.14 -m axovb list                    # version files in the current repo
py -3.14 -m axovb predict                 # bump prediction from `git diff HEAD`
py -3.14 -m axovb predict --model-dir axovb/semver-v1-c1   # real-world-optimized class
py -3.14 -m axovb auto --commit --tag     # predict + apply + commit + tag
```

## Model architecture (PEAK)

### 1. Bumper — PEAK MLP (2.8M params, ONNX)

- Architecture: `20 → 2048 → 1024 → 512 → 256 → 4` (BatchNorm + Dropout(0.3) + ReLU), 2,806,020 params, ~10.7 MB
- Input: 20 structured features regex-extracted from the git diff (`num_changed_symbols`, `has_removed_exports`, `blast_radius`, …) — no tokenizer, no text model
- Output: `{none, patch, minor, major}` + confidence, **temperature-calibrated on validation logits** (`log_T` fit, sanity-guarded — the shipped v0.3.0 artifact baked in a T=-20.05 that inverted the softmax and scored ~0%)
- Training: fixed seeds, 3-way train/val/test split, early stopping on val; two dataset classes ship side by side:

| | `semver-v1-c0` (default) | `semver-v1-c1` (recommended) |
|---|---|---|
| Dataset | legacy 100% synthetic (425) | v3 real + jitter + synthetic (1,971) |
| Held-out v3 (296) | 79.4% | **93.2%** |
| Real repos axios/nanoid/zod (46) | 32.6% | **78.3%** |
| ECE in-distribution | 0.16% | **2.59%** |
| p50 latency / batch-128 | 0.79 ms / 6,102 samples/s | 1.14 ms / 3,714 samples/s |

- Robustness: finite outputs under all OOD probes, deterministic, batch-invariant, NaN/Inf-guarded at the API layer, PyTorch↔ONNX max diff < 5e-7 (measured 2026-10-18 — see [`benchmark-axovb.md`](benchmark-axovb.md))

### 2. Explainer — distilgpt2 (82M, locally fine-tuned)

- Fine-tuned locally on 1,700 synthetic samples (low-memory config); current checkpoint is **1 epoch / 180 samples — undertrained**, so the report shows the raw generation with an explicit caveat and uses the **rule-based principal-engineer narrative** as the primary text
- Input: structured features + predicted bump type → Output: 1–3 sentence explanation
- Weights (312 MB) exceed GitHub's limit and are gitignored — regenerate with `scripts/train_explainer_lomem.py` (5+ epochs on the full set recommended)

## Training workflow

```bash
pip install -r requirements.txt

# Bumper (class c1 shown; c0 uses axovb/data/legacy-train-synthetic.jsonl)
py -3.14 axovb/train.py --data axovb/data/train.jsonl \
  --output axovb/semver-v1-c1 --model-name semver-v1-c1 \
  --dataset-label "v3: real release deltas + jitter + synthetic (1971)"

# Explainer (regenerates explainer-v1-c0/model.safetensors)
py -3.14 scripts/train_explainer_lomem.py \
  --data axovb/data/explainer_train.jsonl --output axovb/explainer-v1-c0/ \
  --epochs 5 --batch-size 2 --grad-accum 8 --max-length 64

# Full benchmark (incl. comparison vs the broken v0.3.0 baseline pinned at 0291f55)
py -3.14 axovb/benchmark.py --compare-baseline
```

## Attribution

The fine-tuning/export stack uses [PyTorch](https://pytorch.org/), [HuggingFace Transformers](https://huggingface.co/docs/transformers) and [ONNX Runtime](https://onnxruntime.ai/). The explainer is fine-tuned from [distilgpt2](https://huggingface.co/distilgpt2); planned AXOTEST models use [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) and [Salesforce/codet5-small](https://huggingface.co/Salesforce/codet5-small).

## License

MIT © 2026 Akik Faraji — Fraziym Tech & AI
