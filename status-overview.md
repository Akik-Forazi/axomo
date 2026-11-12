# AXOMO — Project Status Report

*Generated 2026-10-18 · updated after the axovb model-class remediation · branch `main`, HEAD `6fe2c47` + uncommitted remediation*

## TL;DR

- **axovb remediation done**: the deployed artifact was a legacy-dataset model; the bump classifier now ships as numbered dataset classes (`semver-v1-c0` default, `semver-v1-c1` recommended), retrained with temperature fitted on validation logits, NaN-guarded at the API layer, and benchmarked against the broken v0.3.0 baseline (T=-20.05, ~0% accuracy). Docs/registry updated with honest numbers.
- **Two of three axotest trainers have clean datasets and have never been trained** (security: ready; edge: ready).
- **axotest-unit is blocked**: its data prep scraped essentially nothing (clone timeouts).
- Repo hygiene from the first pass is committed (`.gitignore` incl. `.env` protection, temp script deleted).

## Pipeline status per module

| Module | Dataset | Trained | ONNX export | Verdict |
|---|---|---|---|---|
| **axovb** (bump classifier, BumpMLP 2.8M) | ✅ 1,971 v3 (real+synthetic) / 425 legacy | ✅ c0+c1 (v0.4.0) | ✅ `semver-v1-c0/c1 bump_mlp.onnx` | **Working** — c1: 93.2% v3 / 78.3% real; c0: 79.4% v3 / 32.6% real |
| **axotest-security** (DistilBERT 66M) | ✅ 504 train / 89 test (67/33 secure–vulnerable) | ❌ | ❌ | **Ready to train** |
| **axotest-edge** (DistilBERT 66M, token classification) | ✅ 1005 train / 178 test | ❌ | ❌ | **Ready to train** (prep "crash" was cosmetic — data intact) |
| **axotest-unit** (CodeT5-small 60M) | ❌ 0 train / 1 test | ❌ | ❌ | **Blocked** — re-prep needed |

## 1. axovb — model-class remediation (was: uncommitted bugfix)

What the audit found and what changed:

1. **Deployed c0 was trained on 100% synthetic legacy data** (425 samples) — 98.7% on legacy test, only **32.6% on real repos**. Kept as `semver-v1-c0` (default), documented honestly.
2. **v3-trained model promoted to its own class** `semver-v1-c1` (1,971 samples) — **93.2% on v3 held-out, 78.3% on real repos** (axios/nanoid/zod), ECE 2.59% in-distribution.
3. **Temperature scaling fixed properly** — fitted on **validation logits** with `log_T` parameterization + sanity guard (0.05 ≤ T ≤ 20, reject if val NLL worsens); never touches the test split. New T: c0 = 0.110, c1 = 0.835. The old artifact's T=-20.05 (which inverts the softmax, ~0% accuracy) is pinned for comparison at `0291f55` via `benchmark.py --compare-baseline`.
4. **NaN/Inf guard** added to `BumpPredictor.predict_features/predict_batch` (`np.nan_to_num`).
5. **Renames** (git mv, history preserved): `models/ → semver-v1-c0/`, `models-v2-staging/ → semver-v1-c1/`, `models-v0.3.1-backup/ → semver-v1-c0-backup/`, `explainer/ → explainer-v1-c0/`.
6. **Docs/registry rewritten with honest numbers** (`README.md`, `axovb/README.md`, `benchmark-axovb.md`, `model_config.json`), stale DistilBERT/66M claims and the "from axodex" wording removed, Windows UTF-8 console guard added, dead `total_files` code fixed, docstring param count corrected.

Open quality gaps (documented in `benchmark-axovb.md`): real-repo holdout is only 46 samples (±10–13% CI); c1's real-domain ECE is 15.0%; explainer still 1 epoch / 180 samples.

## 2. axotest-security — clean prep, ready to train

- Prep log: no errors. 504 train / 89 test; labels: secure 338 (67.1%) / vulnerable 166 (32.9%). One django clone timeout (warn only).
- Data on disk: `data/train.jsonl` (831 KB / 504 lines), `data/test.jsonl` (146 KB / 89 lines), fields `text/label/file`.
- No `model/`, `onnx/`, or checkpoints → training never started.

**Next:** `python train.py --data ./data/train.jsonl --output ./model/` (base `distilbert-base-uncased`).

## 3. axotest-edge — data intact, ready to train

Details in `status-axotest-edge.md` (prior investigation, 2026-10-18):

- Prep wrote **1005 train / 178 test** successfully, then crashed printing stats (`shared/utils.py:48`, `TypeError: unhashable type: 'list'` — token-classification labels are BIO lists). Cosmetic; "Data saved" line never printed but files are complete.
- No model/onnx artifacts — training never started.

**Next:** `python axotest-edge/train.py --data axotest-edge/data/train.jsonl --output axotest-edge/model` → `eval.py` → `shared/export_onnx.py --task token-classification`. Optional 1-line fix to `shared/utils.py:48` so token-classification stats don't crash.

## 4. axotest-unit — blocked, prep yielded no data

- Prep log: no crash, but **0 train / 1 test sample** — scrape produced 1 pair (zod), 0 for swr/yup/pm2; react-query clone timed out.
- `data/train.jsonl` is 0 bytes.

**Next:** fix/re-run `prepare_data.py` first (clone strategy/timeouts), then train `Salesforce/codet5-small`.

## 5. Repo hygiene — status

1. ~~Commit the `.gitignore` change (`+.env`)~~ — **done** (`6fe2c47`), plus axodex artifacts, `.zwork/`, `*.log`, and the explainer weights glob.
2. ~~Delete `scripts/_patch_train_temp.py`~~ — **done** (deleted).
3. ~~Fix README `package.json` claim~~ — **done** (README rewritten with the real structure).
4. `.zwork/` — **ignored** now.
5. Remaining cosmetics: `scripts/test_results.json` (stale run log) and `status-axotest-edge.md` parked at repo root; axodex context files are gitignored.

## Recommended order of work

1. Commit the axovb remediation (rename + retrain + docs).
2. Decide whether `semver-v1-c1` becomes the loaded default (nothing auto-promotes today; `--model-dir` switches per run).
3. Kick off training for **axotest-security** and **axotest-edge** (both datasets ready; longest lead time).
4. Re-prep **axotest-unit** data (fix clone timeouts).
5. Grow the real-repo holdout (46 samples is thin) and consider an explainer retrain at 5+ epochs.
