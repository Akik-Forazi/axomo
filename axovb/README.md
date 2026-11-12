# axovb — AI-powered semantic version bumper

> PEAK MLP (2.8M params, ONNX) predicts the bump type from a git diff.
> distilgpt2 (82M params) generates a natural-language narrative of why.
> Project-wide version files are then edited atomically across 8 ecosystems
> with git commit + tag.

## Quick start

```bash
# Show what version files exist in the current repo
python -m axovb list

# Predict the bump type from staged git diff
python -m axovb predict

# Predict + also generate a narrative explanation
python -m axovb predict --explain

# Predict + apply project-wide + commit + tag
python -m axovb auto --commit --tag

# Apply an explicit major bump across every version file
python -m axovb apply --major --commit --tag

# Apply to a specific version
python -m axovb apply --to 3.0.0 --commit --tag --push

# Evaluate the trained model on the held-out test set
python -m axovb eval

# Show model artifacts + their sizes
python -m axovb info
```

## Subcommands

| Command | What it does |
|---|---|
| `predict` | Run the PEAK MLP on `git diff HEAD` (or `--from-diff path.patch`). Prints predicted bump type + per-class probabilities. |
| `apply` | Bump every detected version file by `--major` / `--minor` / `--patch` / `--to X.Y.Z`. Atomic writes with full rollback on any failure. |
| `auto` | Predict bump type from staged diff, then apply it. Single-shot. |
| `eval` | Load `bump_mlp.onnx` + `data/test.jsonl`, run inference, print accuracy / F1 / precision / recall / confusion matrix / per-class confidence / throughput. |
| `info` | Show all trained artifacts + their sizes + training metrics. |
| `list` | Walk the project root and list every detected version file + its current version + ecosystem. |

## Supported version files (auto-detected)

| Ecosystem | Files bumped |
|---|---|
| **node** | `package.json`, `lerna.json`, `packages/*/package.json`, `apps/*/package.json` |
| **python** | `pyproject.toml` (PEP 621), `setup.py` (legacy) |
| **rust** | `Cargo.toml` |
| **helm** | `Chart.yaml` (bumps both `version:` AND `appVersion:`) |
| **elixir** | `mix.exs` |
| **go** | `go.mod` `// Version:` comment |
| **dotnet** | `*.csproj` (bumps `Version` + `AssemblyVersion` + `FileVersion`) |
| **plain** | `VERSION` |
| **meta** | `CHANGELOG.md` (prepends new release section, doesn't replace) |

## Model classes

A class (`c<N>`) is defined by the DATASET a checkpoint was trained on; one architecture ships
as several classes side by side (`<task>-v<semver>-c<N>`).

| Class | Params | Size | Held-out v3 (296) | Real repos (46) | ECE (v3) | File |
|---|---|---|---|---|---|---|
| `semver-v1-c0` *(deployed default)* | 2.8M | 10.7 MB | 79.4% | **32.6%** | 19.5% | `semver-v1-c0/bump_mlp.onnx` |
| `semver-v1-c1` *(recommended for real repos)* | 2.8M | 10.7 MB | **93.2%** | **78.3%** | **2.6%** | `semver-v1-c1/bump_mlp.onnx` |
| `semver-v1-c0-backup` | — | — | archive of the pre-rename c0 bytes | | | `semver-v1-c0-backup/` |

Switch classes with `--model-dir axovb/semver-v1-c1`. Nothing is promoted automatically.

| Explainer | Params | Size | Quality | Inference | File |
|---|---|---|---|---|---|
| `explainer-v1-c0` (distilgpt2) | 82M | 312 MB (gitignored) | undertrained (1 epoch / 180 samples) — rule-based narrative stays primary | ~1.1 s/sample | `explainer-v1-c0/model.safetensors` |

The `model.safetensors` (312 MB) exceeds GitHub's 100 MB file limit, so it's gitignored.
It's regenerable from `data/explainer_train.jsonl` (1.5 MB, preserved) using:

```bash
python scripts/train_explainer_lomem.py \
  --data axovb/data/explainer_train.jsonl \
  --output axovb/explainer-v1-c0/ \
  --epochs 1 --batch-size 2 --grad-accum 8 --max-length 64 --limit 200
```

## Atomic write + rollback

When you run `apply`, the tool writes every file via its detector's writer.
Each writer:
1. Backs up the original file to a per-file tempdir
2. Writes the new version
3. Re-parses the file and asserts the parsed version equals the new version
4. If verification fails, raises → triggers full rollback of all already-written files

This means: if file #7 of 15 fails verification, files #1–6 are restored from
their backups. Zero state drift.

## Git integration

With `--commit`, the tool runs `git add <version files>` + `git commit -m "chore(release): <kind> bump to vX.Y.Z"` with a body listing the diff features
that drove the prediction. With `--tag`, it creates an annotated `vX.Y.Z` tag.
With `--push`, it pushes the commit + tag to `origin`.

## ML prediction details

The PEAK MLP takes a 20-feature vector extracted from the git diff:
- `num_changed_symbols`, `num_callers_affected`, `has_removed_exports`,
  `has_new_exports`, `has_exported_changes`, `has_signature_changes`,
  `has_type_changes`, `has_rename`
- `num_source_files`, `num_test_files`, `num_doc_files`, `num_config_files`
- `total_files`, `test_to_source_ratio`, `blast_radius`, `change_scope`
- `avg_callers_per_symbol`, `export_change_ratio`, `breaking_score`,
  `complexity_score`

The ONNX export wraps the MLP in a `BumpMLPInference` module that:
1. Normalizes input: `(x - X_min) / max(X_max - X_min, 1)`
2. Forwards through the MLP
3. Applies temperature scaling — `T = exp(log_T)` fitted on **validation logits**
   (Guo et al. 2017), guarded to `0.05 ≤ T ≤ 20` and rejected if it doesn't
   improve val NLL (fallback T=1.0). The shipped v0.3.0 artifact baked in a
   **T=-20.05** that inverted the softmax ranking and scored ~0% — never fit
   temperature in raw log space or on the test set.
4. Softmax

**Important**: callers must pass RAW features (not pre-normalized). The wrapper
handles normalization internally — pre-normalizing would double-normalize and
produce garbage predictions (0% accuracy). `BumpPredictor` also applies an
`np.nan_to_num` guard so one NaN/Inf feature can't poison the output.

## The distilgpt2 explainer

Trained on 180 samples (low-memory config: batch 2 + grad_accum 8 + gradient
checkpointing, fits in 4 GB RAM). The model is undertrained (1 epoch), so the
post-mortem report uses a rule-based principal-engineer narrative as the
primary text and shows the raw ML generation as a side panel with an explicit
"model is undertrained" caveat. To make the ML narrative carry the report,
train for 5+ epochs on the full 1700-sample dataset.

## File layout

```
axovb/
├── __init__.py        — package metadata + version
├── __main__.py        — `python -m axovb` entry point
├── cli.py             — subcommand dispatcher (predict/apply/auto/eval/info/list)
├── bumper.py          — version-file detectors + atomic writer + git integration
├── analyzer.py        — diff feature extractor + static security/edge/blast analysis
├── predictor.py       — ONNX inference wrapper for the PEAK MLP
├── explainer.py       — distilgpt2 narrative generation (with rule-based fallback)
├── eval.py            — model evaluation on held-out test set (ONNX-based)
├── train.py           — train the PEAK MLP + export ONNX
├── train_explainer.py — train the distilgpt2 explainer (legacy, full-memory)
├── prepare_data.py    — parse bumper training data from cloned repos
├── prepare_explainer_data.py — parse explainer training data (synthetic)
├── model_config.json  — describes the deployed model artifacts
├── data/              — parsed training data (.jsonl, preserved — DO NOT re-parse)
│   ├── train.jsonl                 (1971 — v3: real 172 + synthetic 1200 + jitter 514 + docs 85)
│   ├── test.jsonl                  (296 — 250 synthetic-v2 + 46 real)
│   ├── test-real.jsonl             (46 — axios 9 / nanoid 20 / zod 17)
│   ├── test-synthetic-v2.jsonl     (250)
│   ├── legacy-train-synthetic.jsonl (425 — the c0 class dataset)
│   ├── legacy-test-synthetic.jsonl  (75)
│   ├── explainer_train.jsonl       (1700 — explainer)
│   └── explainer_test.jsonl        (300 — explainer)
├── semver-v1-c0/      — deployed default class (ONNX + .pt + metadata.json)
├── semver-v1-c1/      — v3 dataset class — recommended for real repos
├── semver-v1-c0-backup/ — archive of the original pre-rename c0 bytes
└── explainer-v1-c0/   — trained distilgpt2 (safetensors gitignored — regenerable)
    ├── config.json
    ├── tokenizer.json
    ├── training_metrics.json
    ├── sample_generations.json
    └── model.safetensors         (312 MB — gitignored, regenerable)
```

Each model class directory contains `bump_mlp.onnx` (10.7 MB), `bump_mlp.pt` (PyTorch
checkpoint) and `metadata.json` (feature names, normalizer, label map, temperature, class name).

## Retraining from the preserved data

```bash
# Bumper class c1 (v3 dataset, recommended): retrain from data/train.jsonl
python axovb/train.py --data axovb/data/train.jsonl \
  --output axovb/semver-v1-c1 --model-name semver-v1-c1 \
  --dataset-label "v3: real release deltas + jitter + synthetic (1971)"

# Bumper class c0 (legacy dataset, deployed default)
python axovb/train.py --data axovb/data/legacy-train-synthetic.jsonl \
  --output axovb/semver-v1-c0 --model-name semver-v1-c0 \
  --dataset-label "legacy: 100% synthetic 425 (v2 era)"

# Explainer: retrain from data/explainer_train.jsonl (regenerates model.safetensors)
python scripts/train_explainer_lomem.py \
  --data axovb/data/explainer_train.jsonl \
  --output axovb/explainer-v1-c0/ \
  --epochs 5 --batch-size 2 --grad-accum 8 --max-length 64

# Full benchmark suite (latency, ECE, robustness, baseline comparison)
python axovb/benchmark.py --compare-baseline

# Verify everything still passes
python scripts/run_all_tests.py
```

## License

MIT
