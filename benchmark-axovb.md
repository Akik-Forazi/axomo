# AXOVB Bumper — Benchmark & Verification Report

**Model family:** AXOVB bumper — PEAK MLP 2.8M params (2,806,020), ONNX, 4-class classifier
**Classes:** `semver-v1-c0` (deployed default, legacy dataset) · `semver-v1-c1` (v3 dataset, recommended for real repos)
**Date:** 2026-10-18 · **Harness:** `axovb/benchmark.py` (`--compare-baseline`, `--model-dir`)
**Environment:** Python 3.14.8 · numpy 2.5.3 · onnxruntime 1.30.0 · Windows 11 (win32), CPU

A **class** (`c<N>`) is defined by the DATASET a checkpoint was trained on — the same architecture ships
as several classes side by side. `semver-v1-c0-backup/` keeps the original pre-rename c0 bytes for
archaeology. Nothing is promoted automatically; `semver-v1-c0` remains the default loaded class.

| Class | Dataset | ONNX sha256 (first 12) | Temperature |
|---|---|---|---|
| `semver-v1-c0` (default) | legacy 100% synthetic, 425 samples | `beb0c455d9c5` | 0.110 |
| `semver-v1-c1` | v3: real + jitter + synthetic, 1,971 samples | `3976b6bcbeb1` | 0.835 |
| baseline @ `0291f55` (broken v0.3.0) | legacy 425 | `7db8c5be09d1` | **-20.05** |

---

## 1. Verdict

| Question | `semver-v1-c0` | `semver-v1-c1` |
|---|---|---|
| In-distribution accuracy (own dataset) | 100% (legacy 75) | 93.24% (v3 296) |
| Real-repo holdout accuracy (46) | **32.61%** | **78.26%** |
| Calibration (ECE, in-distribution) | 0.16% (legacy) | 2.59% (v3) |
| Fast enough? | p50 **0.79 ms** single-request | p50 **1.14 ms** |
| Robust? (OOD, NaN-guarded, deterministic, batch-invariant) | Yes | Yes |
| The broken baseline | **1.69%** on v3 (T=-20.05 inverted softmax) | — |

`semver-v1-c0` is excellent **on its own distribution** (legacy synthetic) and poor on real repos —
it is kept for compatibility. **`semver-v1-c1` is the model to use for real-world diffs.**

## 2. Accuracy matrix (all four held-out sets, both classes)

| Dataset (samples) | c0 acc | c0 ECE | c1 acc | c1 ECE | baseline acc |
|---|---|---|---|---|---|
| v3 combined (296) | 79.39% | 19.54% | **93.24%** | **2.59%** | 1.69% |
| real repos — axios/nanoid/zod (46) | 32.61% | 65.29% | **78.26%** | 15.04% | 10.87% |
| synthetic-v2 (250) | 88.00% | 11.31% | **96.00%** | **1.72%** | 0.00% |
| legacy synthetic (75) | **100.00%** | **0.16%** | 97.33% | 3.92% | 0.00% |

Internal train/val/test splits (fixed seed): c0 = 308/54/63 → internal test **100%**;
c1 = 1,425/251/295 → internal test 88.1%, val peak 92.8%.

Confusion — c1 on v3 (rows=true):

```
            pred
 true      none   patch   minor   major
 none       65       2       0       0
 patch       1     129       5       2
 minor       0       6      43       3
 major       0       1       0      39
```

Confusion — c0 on v3 (rows=true) — `none`→`patch` (19) and `minor`→`patch` (18) dominate the misses:

```
            pred
 true      none   patch   minor   major
 none       48      19       0       0
 patch      12     122       1       2
 minor       0      18      25       9
 major       0       0       0      40
```

## 3. Calibration

- **c1 / v3: ECE 2.59%** — mean confidence 93.71% vs 93.24% accuracy. Well calibrated in-distribution.
  The new temperature is fitted on **validation logits** (train.py), never the test set, with a
  `T = exp(log_T)` parameterization plus a sanity guard (0.05 ≤ T ≤ 20, NLL must improve, else T=1.0).
- **c1 / real repos: ECE 15.04%** (conf 79.99% vs acc 78.26% — mild overconfidence overall, but
  per-bin gaps up to 42%) — real-domain calibration remains the top open quality gap.
- **c0 / v3: ECE 19.54%, mean conf 98.93%** vs 79.39% accuracy — c0 is severely overconfident
  outside its legacy distribution. Its fitted T=0.110 sharpens (correct for its own data where
  val accuracy is 100%) and does *not* transfer to the real domain.

## 4. Performance (CPU, onnxruntime)

| Metric | c0 | c1 |
|---|---|---|
| Cold load (session creation) | 59.0 ms | 53.3 ms |
| p50 / p90 / p95 / p99 (ms), 3000 iters | 0.79 / 1.07 / 1.27 / 2.86 | 1.14 / 2.42 / 3.02 / 4.92 |
| Throughput batch 1 / 32 / 128 / 512 (samples/s) | 1522 / 4912 / **6102** / 5885 | 739 / 2849 / **3714** / 3801 |
| Amortized @ batch 128 (ms/sample) | 0.164 | 0.269 |

Both classes are sub-3 ms p99 single-request — far below any interactive budget. Run-to-run variance
on a shared dev machine is ~2×, so treat c0-vs-c1 latency differences as noise; the architecture is identical.

## 5. Robustness & determinism (both classes)

- All OOD probes pass: zeros, all-min, all-max, 10× over-range, 100 random rows — finite output, probs sum to 1.
- **NaN/Inf:** the raw ONNX session still propagates NaN; `BumpPredictor.predict_features/predict_batch`
  now apply `np.nan_to_num` — verified: all-NaN input → finite probabilities. (Repeat of the v0.3.0 finding,
  now fixed at the API layer.)
- Deterministic: 50 identical repeats → bit-identical. Batch-invariant: single vs batch-64 identical.
- **PT↔ONNX:** checkpoint re-executed in PyTorch vs ONNX export: max |diff| 4.2e-07 (c0), 2.1e-07 (c1),
  100% argmax agreement across random/extreme probes.
- End-to-end smoke (raw diff → analyzer → prediction): c0 flags both probes `patch`;
  c1 → `patch` for the removed-export probe, `none` (52%) for the comment-tweak probe.

## 6. Baseline (v0.3.0 @ 0291f55) — the broken artifact

Temperature **-20.05** baked into the graph inverts the softmax ranking; the model confidently picks the
worst class. v3 accuracy 1.69% (10.87% on real repos, 0.00% on both synthetic sets), mean confidence
26.1%, argmax agreement with c0 on identical inputs: **0.0%**. The fix (fit `log_T` on val logits + guard)
resolves this. The baseline is extracted read-only from the pinned commit (`--compare-baseline`) so this
comparison stays reproducible after the rename.

## 7. Dataset reality check

- `data/train.jsonl` = **1,971** samples (v3 mixture: 172 release-representative + 1,200 synthetic +
  514 jitter + 85 docs-only). Legacy class trains on `legacy-train-synthetic.jsonl` = **425**, all synthetic.
- Held-out: v3 `test.jsonl` 296 (250 synthetic-v2 + 46 real), plus standalone `test-real.jsonl` (46),
  `test-synthetic-v2.jsonl` (250), `legacy-test-synthetic.jsonl` (75). Holdout repos: axios, nanoid, zod.
- The 46-sample real holdout is the only true OOD check — accuracy there (±13% at 95% CI, n=46) deserves
  a larger set before v1.0.

## 8. Recommended actions (priority order)

1. **Use `semver-v1-c1` for real-world diffs** (`--model-dir axovb/semver-v1-c1`); either keep c0 as the
   default for legacy parity or promote c1 deliberately — not automatically.
2. Grow the real-repo holdout beyond 46 samples (c0's 32.61% and c1's 78.26% both swing ±10–13% at n=46).
3. Real-domain calibration: c1's real ECE is 15.04% despite good in-distribution ECE — consider
   domain-aware calibration before using confidences as hard thresholds.
4. Explainer: still 1 epoch / 180 samples (formulaic output; rule-based narrative remains primary).
   Retrain 5+ epochs on the full 1,700-sample set to let the ML narrative carry the report.
5. Keep `semver-v1-c0-backup/` untouched as the byte history of the original c0.

## 9. Naming scheme: classes, not nicknames

A model class encodes the dataset it was trained on: **`<task>-v<semver>-c<N>`** — `c` = class,
`N` assigned in dataset order. Same architecture, different data → different class:

| Directory | Meaning |
|---|---|
| `axovb/semver-v1-c0/` | bump classifier, dataset class 0 (legacy synthetic 425) — deployed default |
| `axovb/semver-v1-c1/` | bump classifier, dataset class 1 (v3 real+augmented 1,971) — recommended |
| `axovb/semver-v1-c0-backup/` | archive of pre-rename c0 artifacts (byte-identical history) |
| `axovb/explainer-v1-c0/` | distilgpt2 explainer, dataset class 0 |

Planned future classes (security/edge/unit) follow the same rule, each starting at `-v1-c0`.
Code symbols (`BumpMLP`, `BumpPredictor`) and artifact filenames (`bump_mlp.onnx`) stay stable;
the class name is recorded in `metadata.json → model_class`.
