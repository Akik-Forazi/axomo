#!/usr/bin/env python3
"""
Prepare training data for the AXOVB bump classifier (PEAK MLP) — v4.

v4 CHANGES vs v3 (the "depth-120 shallow clone" era):
  1. PRERELEASE TAG BUG FIXED: parse_semver() matched "1.2.3" inside
     "v1.2.3-rc.1", so RC/alpha/beta/next tags leaked into release pairs and
     mislabeled diffs. Tags now must be exact stable semver
     (`v?X.Y.Z` — no prerelease/build suffix) to be considered.
  2. FULL TAG HISTORY: depth-120 shallow clones only saw the newest ~120
     commits, hiding most release tags (14 repos yielded just 303 samples).
     Clones are now BLOBLESS partial clones (`git clone --filter=blob:none
     --no-checkout`): full commit + tag history at a fraction of the size,
     blobs fetched on demand per diff. A `.axomo-blobless` marker identifies
     the new layout; legacy shallow clones are re-cloned automatically.
  3. PER-REPO PAIR CAP (`--max-pairs`, default 60, most recent first) bounds
     scrape time on huge histories (babel/eslint/react); per-pair errors no
     longer abort the whole repo — the pair is skipped and the run continues.
  4. BUILD-NOISE EXCLUDES: coverage/*, *.map, *.tsbuildinfo added to the diff
     pathspec (dist/*, *.min.js, lockfiles, CHANGELOG.md were already out).
  5. uid ON EVERY SAMPLE (repo + version pair / docs commit / synthetic id)
     so train.py can do group-aware splits — jitter copies of one real sample
     can no longer leak across train/val (which previously made early
     stopping + temperature calibration optimistic).
  6. BIGGER HOLDOUT: zod/axios/nanoid kept for continuity, dayjs + date-fns
     added (target: >=150 real holdout samples instead of 46).
  7. Expanded REPOS list (14 -> 37 actively-tagged JS/TS libraries; JS/TS
     only because the analyzer's export/signature regexes are JS-shaped).

v3 CHANGES vs v2 (the "100% synthetic" era):
  1. REAL release-delta data: for each npm library, diff consecutive release
     tags (tagA..tagB) and extract features with axovb.analyzer.
     extract_diff_features — the EXACT code path used at inference time
     (predict_diff). This eliminates the train/serve skew of v2, which
     parsed axodex/file-stat output instead.
  2. Label = the actual version bump between the two released versions
     (classify_bump), taken from tag names — ground truth from real repos.
  3. REAL "none" samples: docs/test/config-only commits (no release needed).
  4. Label-preserving jitter augmentation of real samples (numeric ±20%,
     derived features recomputed with the same formulas train.py uses).
  5. Harder synthetic generator (v2): class overlap (majors without removed
     exports, minors with signature changes, etc.) instead of cleanly
     separable patterns.
  6. Repo-grouped holdout: entire repos (--test-repos) are excluded from
     training so the test set measures generalization to unseen projects.

Usage:
    python prepare_data.py --output ./data/
    python prepare_data.py --output ./data/ --force          # recompute cached repos
    python prepare_data.py --output ./data/ --test-repos colinhacks/zod,axios/axios
"""
import argparse, json, os, random, re, shutil, subprocess, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axovb.analyzer import extract_diff_features
from utils import save_jsonl

REPOS = [
    # — original 14 (v3) ————————————————————————————————————————————————
    "https://github.com/chalk/chalk.git",
    "https://github.com/colinhacks/zod.git",
    "https://github.com/jquense/yup.git",
    "https://github.com/ramda/ramda.git",
    "https://github.com/tj/commander.js.git",
    "https://github.com/debug-js/debug.git",
    "https://github.com/npm/node-semver.git",
    "https://github.com/isaacs/rimraf.git",
    "https://github.com/isaacs/node-glob.git",
    "https://github.com/axios/axios.git",
    "https://github.com/expressjs/express.git",
    "https://github.com/uuidjs/uuid.git",
    "https://github.com/yargs/yargs.git",
    "https://github.com/ai/nanoid.git",
    # — v4 additions (actively tagged, JS/TS) ————————————————————————————
    "https://github.com/iamkun/dayjs.git",
    "https://github.com/date-fns/date-fns.git",
    "https://github.com/moment/moment.git",
    "https://github.com/lodash/lodash.git",
    "https://github.com/jashkenas/underscore.git",
    "https://github.com/jquery/jquery.git",
    "https://github.com/vuejs/core.git",
    "https://github.com/facebook/react.git",
    "https://github.com/rollup/rollup.git",
    "https://github.com/webpack/webpack.git",
    "https://github.com/vitejs/vite.git",
    "https://github.com/postcss/postcss.git",
    "https://github.com/micromatch/micromatch.git",
    "https://github.com/eslint/eslint.git",
    "https://github.com/prettier/prettier.git",
    "https://github.com/babel/babel.git",
    "https://github.com/mochajs/mocha.git",
    "https://github.com/sindresorhus/got.git",
    "https://github.com/sindresorhus/p-limit.git",
    "https://github.com/caolan/async.git",
    "https://github.com/petkaantonov/bluebird.git",
    "https://github.com/socketio/socket.io.git",
    "https://github.com/SBoudrias/Inquirer.js.git",
]

# Only file types the analyzer counts — also keeps lockfiles/build noise out.
DIFF_PATHS = [
    "*.ts", "*.tsx", "*.js", "*.jsx", "*.mjs", "*.cjs", "*.py",
    "*.md", "*.json", "*.yaml", "*.yml", "*.toml",
    ":(exclude)*.lock", ":(exclude)package-lock.json", ":(exclude)yarn.lock",
    ":(exclude)pnpm-lock.yaml", ":(exclude)*.min.js", ":(exclude)dist/*",
    ":(exclude)coverage/*", ":(exclude)*.map", ":(exclude)*.tsbuildinfo",
    ":(exclude)CHANGELOG.md",
]

# v4: exact stable-semver tags only. Rejects v1.2.3-rc.1 / 4.0.0-next.3 /
# 2.0.0-beta (the v3 parser matched "1.2.3" inside these and mislabeled
# release pairs whose diff was actually an RC delta, not a release delta).
STABLE_TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

BLOBLESS_MARKER = ".axomo-blobless"


def _force_rmtree(path: str):
    """rmtree that survives Windows read-only git objects (.git/objects/*
    are read-only on Windows; plain shutil.rmtree(ignore_errors=True) leaves
    them behind and the subsequent git clone then fails with 'destination
    path already exists and is not an empty directory')."""
    import stat

    def _onexc(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:            # noqa: BLE001 — best effort, logged by caller
            pass

    try:
        shutil.rmtree(path, onexc=_onexc)
    except TypeError:                # Python < 3.12 — onexc not yet available
        def _onerror(func, p, _exc):
            try:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            except Exception:        # noqa: BLE001
                pass
        shutil.rmtree(path, onerror=_onerror)


def run_git(args, cwd, timeout=60):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


def classify_bump(old: str, new: str) -> str:
    o = [int(x) for x in old.split(".")[:3]]
    n = [int(x) for x in new.split(".")[:3]]
    if len(o) != 3 or len(n) != 3:
        return "none"
    if n[0] > o[0]:
        return "major"
    if n[1] > o[1]:
        return "minor"
    if n[2] > o[2]:
        return "patch"
    return "none"


def parse_semver(text: str):
    """Loose 3-part semver finder (used to SKIP release-ish commit subjects)."""
    m = re.search(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", text)
    return m.group(1) if m else None


def repo_tags(repo_dir: str):
    """[(version, commit_hash)] sorted by tag creation date ascending.
    v4: only exact stable-semver tags; re-tagged versions keep the newest."""
    r = run_git(["for-each-ref", "refs/tags", "--sort=creatordate",
                 "--format=%(refname:short)%09%(*objectname)%09%(objectname)"],
                repo_dir, timeout=120)
    by_ver = {}
    order = 0
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, peeled, direct = parts[0], parts[1], parts[2]
        m = STABLE_TAG_RE.match(name.strip())
        if not m:
            continue                      # prerelease / non-semver tag → skip
        ver = ".".join(m.groups())
        commit = peeled or direct
        if not commit:
            continue
        by_ver[ver] = (ver, commit, order)   # later tag with same version wins
        order += 1
    tags = sorted(by_ver.values(), key=lambda t: t[2])
    return [(ver, commit) for ver, commit, _ in tags]


def diff_features(repo_dir: str, ref_a: str, ref_b: str, cap=6_000_000):
    r = run_git(["diff", "--no-color", f"{ref_a}..{ref_b}", "--"] + DIFF_PATHS,
                repo_dir, timeout=180)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    diff = r.stdout[:cap]
    features, _, _ = extract_diff_features(diff)
    return features


def ensure_blobless_clone(repo_url: str, clone_dir: str) -> bool:
    """Blobless partial clone with full tag history. Replaces legacy
    depth-120 shallow clones (marker-based). Returns True if usable."""
    if os.path.exists(os.path.join(clone_dir, BLOBLESS_MARKER)):
        r = run_git(["fetch", "--prune", "--tags", "--quiet"], clone_dir, timeout=180)
        return r.returncode == 0
    if os.path.exists(clone_dir):
        _force_rmtree(clone_dir)          # legacy shallow clone / partial clone
    r = subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "--quiet", repo_url, clone_dir],
                       capture_output=True, timeout=420)
    if r.returncode != 0:
        print(f"    clone failed: {r.stderr.decode(errors='replace')[:120]}")
        return False
    with open(os.path.join(clone_dir, BLOBLESS_MARKER), "w") as f:
        f.write("blobless partial clone — full commit/tag history\n")
    return True


def collect_from_repo(repo_url: str, clones_dir: str, real_dir: str,
                      force: bool, max_pairs: int, none_cap: int) -> int:
    """Scrape release-pair + docs-only samples from one repo. Returns count.
    v4: per-pair error isolation — one bad pair never kills the repo."""
    name = repo_url.split("/")[-1].removesuffix(".git")
    out_path = os.path.join(real_dir, f"{name}.jsonl")
    if os.path.exists(out_path) and not force:
        return len(open(out_path, encoding="utf-8").read().splitlines())
    if force and os.path.exists(out_path):
        os.remove(out_path)         # stale cache from a previous layout — no mixing

    clone_dir = os.path.join(clones_dir, name)
    try:
        if not ensure_blobless_clone(repo_url, clone_dir):
            return 0
    except subprocess.TimeoutExpired:
        print(f"    clone timeout — skipped ({name})")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"    clone error: {e}")
        return 0

    samples = []
    skipped = 0

    # --- release pairs (real patch/minor/major) -----------------------
    try:
        tags = repo_tags(clone_dir)
    except Exception as e:  # noqa: BLE001
        print(f"    tag walk failed: {e}")
        tags = []
    pairs = list(zip(tags, tags[1:]))[-max_pairs:]     # most recent pairs
    for (v_old, c_old), (v_new, c_new) in pairs:
        if c_old == c_new:
            continue
        bump = classify_bump(v_old, v_new)
        if bump == "none":          # equal or decreasing version → skip
            continue
        try:
            feats = diff_features(clone_dir, c_old, c_new)
        except Exception:           # noqa: BLE001 — timeout etc: skip pair
            skipped += 1
            continue
        if feats is None or feats["total_files"] == 0:
            continue
        samples.append({"features": feats, "label": bump, "repo": name,
                        "old_version": v_old, "new_version": v_new,
                        "kind": "release",
                        "uid": f"{name}:{v_old}->{v_new}"})

    # --- docs/test/config-only commits (real "none") -------------------
    try:
        log = run_git(["log", "--format=%H%x09%s"], clone_dir, timeout=120)
        log_lines = log.stdout.splitlines()
    except Exception:               # noqa: BLE001
        log_lines = []
    n_none = 0
    for line in log_lines:
        if n_none >= none_cap:
            break
        parts = line.split("\t", 1)
        if len(parts) < 2 or parse_semver(parts[1]):
            continue
        try:
            feats = diff_features(clone_dir, f"{parts[0]}~1", parts[0])
        except Exception:           # noqa: BLE001
            continue
        if feats is None:
            continue
        if (feats["total_files"] > 0 and feats["num_source_files"] == 0
                and feats["num_changed_symbols"] == 0):
            samples.append({"features": feats, "label": "none", "repo": name,
                            "old_version": None, "new_version": None,
                            "kind": "docs_only",
                            "uid": f"{name}:docs:{parts[0][:10]}"})
            n_none += 1

    if samples:
        save_jsonl(samples, out_path)
    elif os.path.exists(out_path):
        os.remove(out_path)          # stale cache from a previous layout
    if skipped:
        print(f"    ({skipped} pairs skipped on errors)")
    return len(samples)


# ─── derived features (must mirror train.py exactly) ──────────────────────

def derive(features14: dict) -> dict:
    f = dict(features14)
    f["total_files"] = (f["num_source_files"] + f["num_test_files"]
                        + f["num_doc_files"] + f["num_config_files"])
    f["test_to_source_ratio"] = round(f["num_test_files"] / max(f["num_source_files"], 1), 2)
    f["blast_radius"] = f["num_callers_affected"]
    f["change_scope"] = min(f["num_changed_symbols"], 50)
    f["avg_callers_per_symbol"] = round(
        f["num_callers_affected"] / max(f["num_changed_symbols"], 1), 2)
    f["export_change_ratio"] = round(
        (f["has_removed_exports"] + f["has_new_exports"] + f["has_exported_changes"])
        / max(f["num_changed_symbols"], 1), 2)
    f["breaking_score"] = (f["has_removed_exports"] * 3
                           + f["has_signature_changes"] * 2
                           + min(f["num_callers_affected"], 50) / 50)
    f["complexity_score"] = (f["num_source_files"] + f["num_changed_symbols"]
                             + min(f["num_callers_affected"], 50)) / 100
    return f


NUMERIC = ["num_changed_symbols", "num_callers_affected", "num_source_files",
           "num_test_files", "num_doc_files", "num_config_files"]


def jitter_sample(sample: dict, rng: random.Random) -> dict:
    f = dict(sample["features"])
    for k in NUMERIC:
        f[k] = max(0, round(f[k] * rng.uniform(0.8, 1.2)))
    return {"features": derive(f), "label": sample["label"],
            "repo": sample["repo"], "old_version": sample.get("old_version"),
            "new_version": sample.get("new_version"), "kind": "jitter",
            "uid": sample.get("uid")}      # SAME uid as parent → grouped split


def gen_synthetic_v2(n: int, seed: int) -> list:
    """Harder synthetic data: overlapping class-conditional distributions."""
    rng = random.Random(seed)
    data = []
    for i in range(n):
        bump = rng.choices(["patch", "minor", "major", "none"],
                           weights=[40, 25, 15, 20])[0]
        if bump == "major":
            f = {
                "num_changed_symbols": rng.randint(3, 30),
                "num_callers_affected": rng.randint(5, 100),
                "has_removed_exports": 1 if rng.random() < 0.75 else 0,
                "has_new_exports": rng.choice([0, 1]),
                "has_exported_changes": 1,
                "has_signature_changes": rng.randint(1, 4),
                "has_type_changes": 1 if rng.random() < 0.7 else 0,
                "has_rename": 1 if rng.random() < 0.3 else 0,
                "num_source_files": rng.randint(2, 15),
                "num_test_files": rng.randint(0, 5),
                "num_doc_files": rng.randint(0, 2),
                "num_config_files": rng.randint(0, 1),
            }
        elif bump == "minor":
            f = {
                "num_changed_symbols": rng.randint(2, 14),
                "num_callers_affected": rng.randint(0, 25),
                "has_removed_exports": 1 if rng.random() < 0.05 else 0,
                "has_new_exports": 1 if rng.random() < 0.8 else 0,
                "has_exported_changes": 1 if rng.random() < 0.85 else 0,
                "has_signature_changes": rng.randint(0, 2),
                "has_type_changes": rng.choice([0, 0, 1]),
                "has_rename": 1 if rng.random() < 0.15 else 0,
                "num_source_files": rng.randint(1, 10),
                "num_test_files": rng.randint(0, 5),
                "num_doc_files": rng.randint(0, 2),
                "num_config_files": rng.randint(0, 1),
            }
        elif bump == "patch":
            f = {
                "num_changed_symbols": rng.randint(0, 6),
                "num_callers_affected": rng.randint(0, 8),
                "has_removed_exports": 1 if rng.random() < 0.02 else 0,
                "has_new_exports": 0,
                "has_exported_changes": 1 if rng.random() < 0.25 else 0,
                "has_signature_changes": rng.randint(0, 1),
                "has_type_changes": 1 if rng.random() < 0.1 else 0,
                "has_rename": 0,
                "num_source_files": rng.randint(1, 6),
                "num_test_files": rng.randint(0, 4),
                "num_doc_files": rng.randint(0, 1),
                "num_config_files": rng.randint(0, 2),
            }
        else:  # none — docs / tests / config only (no shipped code change)
            f = {
                "num_changed_symbols": 0,
                "num_callers_affected": 0,
                "has_removed_exports": 0,
                "has_new_exports": 0,
                "has_exported_changes": 0,
                "has_signature_changes": 0,
                "has_type_changes": 0,
                "has_rename": 0,
                "num_source_files": 1 if rng.random() < 0.12 else 0,
                "num_test_files": rng.randint(0, 3),
                "num_doc_files": rng.randint(0, 6),
                "num_config_files": rng.randint(0, 3),
            }
        data.append({"features": derive(f), "label": bump,
                     "repo": "synthetic-v2", "kind": "synthetic",
                     "uid": f"syn-v4:{seed}:{i}"})
    return data


def stats(title: str, rows: list):
    print(f"\n  {title}: {len(rows)} samples")
    if not rows:
        return
    for k, c in sorted(Counter(r["label"] for r in rows).items()):
        print(f"    {k:<6} {c:>5} ({c/len(rows)*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Prepare the bump classifier dataset v4 (real + synthetic)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--synthetic", type=int, default=1200)
    ap.add_argument("--synthetic-test", type=int, default=250)
    ap.add_argument("--jitter-copies", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=60,
                    help="release pairs per repo (most recent first)")
    ap.add_argument("--none-per-repo", type=int, default=12)
    ap.add_argument("--test-repos",
                    default="colinhacks/zod,axios/axios,ai/nanoid,iamkun/dayjs,date-fns/date-fns",
                    help="comma-separated repo slugs held out from training")
    ap.add_argument("--force", action="store_true", help="recompute cached repos")
    args = ap.parse_args()

    out = args.output
    clones_dir = os.path.join(out, "_clones")
    real_dir = os.path.join(out, "_real")
    os.makedirs(clones_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    holdout = {s.strip().split("/")[-1].lower() for s in args.test_repos.split(",") if s.strip()}

    # Preserve the legacy 100%-synthetic dataset for before/after comparison.
    legacy_train = os.path.join(out, "legacy-train-synthetic.jsonl")
    legacy_test = os.path.join(out, "legacy-test-synthetic.jsonl")
    if os.path.exists(os.path.join(out, "train.jsonl")) and not os.path.exists(legacy_train):
        shutil.copyfile(os.path.join(out, "train.jsonl"), legacy_train)
        shutil.copyfile(os.path.join(out, "test.jsonl"), legacy_test)
        print("Legacy synthetic dataset preserved as legacy-*.jsonl")

    # v4: preserve the v3 dataset (semver-v1-c1's training data) the same way
    # before overwriting train/test/test-real — every model class must stay
    # retrainable from its own preserved .jsonl.
    v3_train = os.path.join(out, "v3-train.jsonl")
    if os.path.exists(os.path.join(out, "train.jsonl")) and not os.path.exists(v3_train):
        shutil.copyfile(os.path.join(out, "train.jsonl"), v3_train)
        shutil.copyfile(os.path.join(out, "test.jsonl"), os.path.join(out, "v3-test.jsonl"))
        if os.path.exists(os.path.join(out, "test-real.jsonl")):
            shutil.copyfile(os.path.join(out, "test-real.jsonl"),
                            os.path.join(out, "v3-test-real.jsonl"))
        print("v3 dataset preserved as v3-*.jsonl")

    # ── scrape real data (incremental, cached) ───────────────────────────
    print(f"Scraping {len(REPOS)} repos (holdout: {sorted(holdout)})...")
    real_all = []
    for url in REPOS:
        name = url.split("/")[-1].removesuffix(".git")
        n = collect_from_repo(url, clones_dir, real_dir, args.force,
                              args.max_pairs, args.none_per_repo)
        print(f"  {name:<14} {n:>3} samples")
    for fn in os.listdir(real_dir):
        if fn.endswith(".jsonl"):
            real_all.extend(json.loads(l) for l in open(os.path.join(real_dir, fn), encoding="utf-8") if l.strip())

    test_real = [r for r in real_all if r["repo"].lower() in holdout]
    real_train = [r for r in real_all if r["repo"].lower() not in holdout]

    rng = random.Random(13)
    jittered = [jitter_sample(r, rng) for r in real_train for _ in range(args.jitter_copies)]

    syn_train = gen_synthetic_v2(args.synthetic, seed=4242)
    syn_test = gen_synthetic_v2(args.synthetic_test, seed=777)

    train_pool = real_train + jittered + syn_train
    test_pool = test_real + syn_test
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)

    save_jsonl(train_pool, os.path.join(out, "train.jsonl"))
    save_jsonl(test_pool, os.path.join(out, "test.jsonl"))
    save_jsonl(test_real, os.path.join(out, "test-real.jsonl"))
    save_jsonl(syn_test, os.path.join(out, "test-synthetic-v2.jsonl"))

    stats("REAL scraped (all repos)", real_all)
    stats("REAL train pool (holdout repos excluded)", real_train)
    stats("REAL holdout (unseen repos)", test_real)
    stats("Jitter-augmented", jittered)
    stats("Synthetic v2 train", syn_train)
    stats("Synthetic v2 test", syn_test)
    stats("FINAL train.jsonl", train_pool)
    stats("FINAL test.jsonl", test_pool)
    by_repo = Counter(r["repo"] for r in real_all)
    print("\n  Real samples per repo: " + ", ".join(f"{k}={v}" for k, v in sorted(by_repo.items())))
    print(f"\nData saved to {out}")


if __name__ == "__main__":
    main()
