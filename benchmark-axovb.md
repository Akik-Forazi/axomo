# SemVerdict — Benchmark & Verification Report

**Model:** SemVerdict (architecture: BumpMLP, 2.8M params) — the AXOVB release-impact classifier
**Artifact:** `axovb/models/bump_mlp.onnx` · 11,258,770 bytes · sha256 `c2a0677a…47449`
**Date:** 2026-10-18 · **Harness:** `axovb/benchmark.py` (`python axovb\benchmark.py --compare-head`)
**Environment:** Python 3.13.15 · numpy 2.5.4 · onnxruntime 1.31.0 · Windows 11 (win32), CPU

---

## 1. Verdict

| Question | Answer |
|---|---|
| Is the model accurate? | **Yes — 98.67%** on the held-out test set (74/75) |
| Is it fast enough? | **Yes — p50 ≈ 0.96 ms single-request**, ~11.8K samples/s batched on CPU |
| Is it robust? | **Yes** — finite, normalized output under all OOD probes; deterministic; batch-invariant |
| Is the uncommitted fix safe to commit? | **Yes — it is critical.** HEAD's model scores **0.00%**; the fix restores 98.67% |
| Ready as a release gate? | **Yes**, with two follow-ups (temperature refit, bigger `major` test coverage) |

## 2. Accuracy & confusion (75 test samples: 33 patch / 18 minor / 16 none / 8 major)

```
              pred
 true      none   patch   minor   major
 none        15       1       0       0
 patch        0      33       0       0
 minor        0       0      18       0
 major        0       0       0       8
```

- **Accuracy 98.67% · weighted F1 98.66% · P 98.71% / R 98.67%** — PASS vs the 90% gate.
- Single miss: one `none` sample predicted `patch` (conservative direction — flags a release that could have been skipped; never the reverse).
- `major` (8/8) and `minor` (18/18) are perfect — breaking-change detection, the expensive-to-get-wrong classes, has zero misses.

## 3. Calibration (ECE, 10 bins)

- **ECE 19.76%** — the model is **systematically underconfident**: mean confidence 78.9% vs 98.67% accuracy. Every confidence bin above 0.5 scored 100% accuracy.
- Wrong predictions are still separable: mean confidence on the miss is 46.4% vs 79.3% on correct ones — a ~50% confidence threshold would filter the residual error.
- *Direction of error is safe*: underconfidence on a release gate means "ask a human", not "silently approve".
- **Recommendation:** refit temperature on train logits (expected T ≈ 0.4–0.6 would sharpen confidence ~2×) and re-export ONNX. Current T=1.0 is the sanity fallback from the recent fix.

## 4. Performance (CPU, single thread via onnxruntime)

| Metric | Value |
|---|---|
| Cold load (session creation) | ~49 ms |
| Single request p50 / p90 / p95 / p99 | **0.96 / 3.89 / 11.95 / 36.32 ms** (3000 iters) |
| Single request mean | 2.70 ms (p99+ tail is OS/GC scheduling noise, not model cost) |
| Throughput, batch 1 / 32 / 128 / 512 | 300 / 9,620 / **11,844** / 10,891 samples/s |
| Amortized per-sample @ batch 128 | 0.084 ms |
| README claim ("0.48 ms") | **Optimistic** — realistic p50 is ~1 ms single-request; 0.08–0.10 ms amortized batched. Either way, far below any interactive budget. |

## 5. Robustness & determinism

- **All OOD probes pass**: zeros, all-min, all-max, 10× over-range, 100 random rows — outputs always finite, probabilities sum to 1 (min/max deviation < 1e-6), no crashes.
- **NaN/Inf inputs propagate to NaN outputs** (softmax of NaN) — recommend a one-line `np.nan_to_num` guard in `BumpPredictor.predict_features` before production use with untrusted feature extraction.
- **Deterministic**: 50 identical repeats → bit-identical probabilities.
- **Batch-invariant**: single-sample vs batch-64 probabilities identical (dynamic-axis export is correct).
- **End-to-end smoke (raw diff → analyzer → prediction)**: pipeline runs; a synthetic "removed exported symbol" diff scored patch 69% (major 11%) — small synthetic diffs under-represent the structural features the model actually consumes; the 98.67% on real extracted-feature data is the authoritative number.

## 6. Critical finding: HEAD model is broken — the fix must be committed

Benchmarking the model currently at `git HEAD` (temperature **-20.05** baked into the graph, extracted read-only from git):

| Metric | HEAD (T = -20.05) | Working tree (T = 1.0) |
|---|---|---|
| Accuracy | **0.00%** | **98.67%** |
| Mean confidence | 26.0% (≈ flattened uniform) | 78.9% |
| ECE | 26.03% | 19.76% |
| Argmax agreement | **0.0% of samples** | — |

**Mechanism:** softmax(logits / T) with *negative* T **inverts the ranking** — the model confidently (~26%) selects the *worst* class for every input. The committed v0.3.0 artifact would fail 100% of release decisions if deployed from HEAD. The working-tree fix (fit `log_T`, clamp to [0.05, 20], fallback to 1.0) fully resolves it.

## 7. Dataset reality check

- `train.jsonl` holds **425 samples** (not the "67 real + 10K augmented" in README); test = 75. Update README or regenerate the augmented dataset.
- Only 8 `major` test samples — the most business-critical class has the thinnest coverage. Recommend expanding the test set before v1.0.

## 8. Recommended actions (priority order)

1. **Commit the temperature-scaling fix** (train.py + metadata + ONNX) — it converts a 0%-accuracy artifact into the working 98.67% one.
2. Refit temperature properly on training logits; re-export ONNX (fixes systematic underconfidence).
3. Add NaN/Inf input guard in `predictor.py`.
4. Grow the test set, especially `major` (n=8) and `none` (n=16).
5. Replace the README "0.48 ms" claim with measured numbers from §4.

## 9. Name rationale: SemVerdict

**SemVerdict** = **Sem**antic **Ver**dic**t** — the model renders the verdict a changeset deserves under semantic versioning: *none / patch / minor / major*.

- **Descriptive**: names exactly what it decides (a SemVer verdict), not how it's built.
- **Professional**: fits the AXOMO product line without cutesy filler; reads well in a CLI: `semverdict: major (91%) — 3 exports removed, 15 callers affected`.
- **Non-invasive adoption**: code symbols (`BumpMLP`, `BumpPredictor`) and artifact filenames (`bump_mlp.onnx`) stay unchanged for compatibility; the brand lives in `metadata.json → display_name` and docs. Suggested full styling: **SemVerdict** — *Release Impact Classifier (AXOVB)*.
