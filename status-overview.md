# AXOMO — Project Status Report

*Generated 2026-10-18 · branch `main`, up to date with `origin/main` (HEAD `0291f55` "Some bugs on windows")*

## TL;DR

- **axovb is done and working** (98.7% accuracy, ONNX inference 0.25 ms) — it carries a coherent, finished-looking uncommitted bugfix that should be committed.
- **Two of three axotest trainers have clean datasets and have never been trained** (security: ready; edge: ready).
- **axotest-unit is blocked**: its data prep scraped essentially nothing (clone timeouts).
- Repo hygiene: an uncommitted `.gitignore` line is the only thing ignoring a `.env` with live API keys — commit it first.

## Pipeline status per module

| Module | Dataset | Trained | ONNX export | Verdict |
|---|---|---|---|---|
| **axovb** (bump classifier, BumpMLP 2.8M) | ✅ 67 real + 10K augmented | ✅ v0.3.0 | ✅ `bump_mlp.onnx` | **Working** — 98.7% acc / 75 samples, 0.251 ms/sample, distilgpt2 explainer (312 MB) loads |
| **axotest-security** (DistilBERT 66M) | ✅ 504 train / 89 test (67/33 secure–vulnerable) | ❌ | ❌ | **Ready to train** |
| **axotest-edge** (DistilBERT 66M, token classification) | ✅ 1005 train / 178 test | ❌ | ❌ | **Ready to train** (prep "crash" was cosmetic — data intact) |
| **axotest-unit** (CodeT5-small 60M) | ❌ 0 train / 1 test | ❌ | ❌ | **Blocked** — re-prep needed |

## 1. axovb — uncommitted work is a finished bugfix

`git diff` on `axovb/` is coherent, not work-in-progress:

- `axovb/train.py:275-296` — temperature scaling now fits `log_T` (T = exp(log_T), always > 0) with a sanity fallback to 1.0; comment cites *"bug v0.3.0: T=-20 flattened all probs"*.
- `axovb/models/metadata.json` — temperature `-20.05 → 1.0`.
- `bump_mlp.onnx` / `bump_mlp.pt` — re-exported (same size; whether retrained or only re-exported is undetermined — T is baked in as a graph constant, `train.py:80`).

Verified working: `python axovb\cli.py info` → v0.3.0 artifact list; ONNX loads and normalizes (random input → probs sum 1.0); `python axovb\eval.py` → **98.7% accuracy, PASS**. `python -m axovb` fails only due to the vendored Python's `safe_path=True` (environmental, not a code bug).

Gaps: no `pyproject.toml`/`setup.py`/console-script (not pip-installable; CLI is `python -m axovb` only) and **no tests** — `eval.py`'s 90% gate is the only check.

**Next:** commit the fix; then add packaging metadata and a minimal test suite.

## 2. axotest-security — clean prep, ready to train

- Prep log: no errors. 504 train / 89 test; labels: secure 338 (67.1%) / vulnerable 166 (32.9%). One django clone timeout (warn only).
- Data on disk: `data/train.jsonl` (831 KB / 504 lines), `data/test.jsonl` (146 KB / 89 lines), fields `text/label/file`.
- No `model/`, `onnx/`, or checkpoints → training never started.
- Not affected by the `shared/utils.py:48` list-labels bug (labels are int-mapped via `LABEL2ID` before `print_stats`).

**Next:** `python train.py --data ./data/train.jsonl --output ./model/` (base `distilbert-base-uncased`).

## 3. axotest-edge — data intact, ready to train

Details in `status-axotest-edge.md` (prior investigation, 2026-10-18):

- Prep wrote **1005 train / 178 test** successfully, then crashed printing stats (`shared/utils.py:48`, `TypeError: unhashable type: 'list'` — token-classification labels are BIO lists). Cosmetic; "Data saved" line never printed but files are complete.
- No model/onnx artifacts — training never started.

**Next:** `python axotest-edge/train.py --data axotest-edge/data/train.jsonl --output axotest-edge/model` → `eval.py` → `shared/export_onnx.py --task token-classification`. Optional 1-line fix to `shared/utils.py:48` so token-classification stats don't crash.

## 4. axotest-unit — blocked, prep yielded no data

- Prep log: no crash, but **0 train / 1 test sample** — scrape produced 1 pair (zod), 0 for swr/yup/pm2; react-query clone timed out.
- `data/train.jsonl` is 0 bytes.
- Not affected by the print_stats bug (its label field is code text, hashable).

**Next:** fix/re-run `prepare_data.py` first (clone strategy/timeouts), then train `Salesforce/codet5-small`.

## 5. Repo hygiene (worth fixing, in order)

1. **Commit the `.gitignore` change (`+.env`)** — it is currently the *only* thing ignoring `.env`, which holds `OPENROUTER_API_KEY` and `ORCA_ROUTER_API_KEY_for_autoclaw` (names verified; values not read).
2. Delete or archive `scripts/_patch_train_temp.py` — one-off rewriter that produced the axovb train.py fix; 0 references repo-wide; untracked leftover.
3. Delete or archive `status-axotest-edge.md` once its content lives in the docs (root placement was a fallback; `.zwork/` refuses writes).
4. Add `.zwork/` to `.gitignore` (runtime dir, untracked).
5. README:47 claims a root `package.json` (`@fraziym/axomo`) — **none exists**. Fix README or add the file.
6. Minor: stray `scripts/test_results.json` and `scripts/__pycache__/`.

## 6. Other folders (context)

- `demo-monorepo/` — generated fixture `axomo-monorepo-demo` v1.4.2 (private, workspaces `pkg/apps`, 8 ecosystems: Cargo, go, mix, Helm chart, dotnet…); no nested `.git`; produced by `scripts/build_demo_monorepo.py`. Scratch/demo, not a runnable app.
- `download/` — 3 tracked files (~2.2 KB): `generated_tests/auth.{boundary,regression,security}.test.ts`, output of `scripts/write_generated_tests.py`.

## Recommended order of work

1. `git add .gitignore && git commit` (protect the secrets ignore) → then commit the axovb temperature-scaling fix.
2. Kick off training for **axotest-security** and **axotest-edge** (both datasets ready; longest lead time).
3. Re-prep **axotest-unit** data (fix clone timeouts).
4. Cleanup: remove temp script + stray report, fix README, gitignore `.zwork/`.
