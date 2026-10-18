#!/usr/bin/env python3
"""
Prepare training data for SemVerdict (AXOVB bump classifier) — v3.

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
    python prepare_data.py --output ./data/ --force          # rescrape all
    python prepare_data.py --output ./data/ --test-repos colinhacks/zod,axios/axios
"""
import argparse, json, os, random, re, shutil, subprocess, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from axovb.analyzer import extract_diff_features
from utils import save_jsonl

REPOS = [
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
]

# Only file types the analyzer counts — also keeps lockfiles/build noise out.
DIFF_PATHS = [
    "*.ts", "*.tsx", "*.js", "*.jsx", "*.mjs", "*.cjs", "*.py",
    "*.md", "*.json", "*.yaml", "*.yml", "*.toml",
    ":(exclude)*.lock", ":(exclude)package-lock.json", ":(exclude)yarn.lock",
    ":(exclude)pnpm-lock.yaml", ":(exclude)*.min.js", ":(exclude)dist/*",
    ":(exclude)CHANGELOG.md",
]

SEMVER_RE = re.compile(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)")


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
    m = SEMVER_RE.search(text)
    return m.group(1) if m else None


def repo_tags(repo_dir: str):
    """[(version, commit_hash)] sorted by tag creation date ascending."""
    r = run_git(["for-each-ref", "refs/tags", "--sort=creatordate",
                 "--format=%(refname:short)%09%(*objectname)%09%(objectname)"],
                repo_dir)
    tags = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, peeled, direct = parts[0], parts[1], parts[2]
        ver = parse_semver(name)
        if not ver:
            continue
        tags.append((ver, peeled or direct))
    return tags


def diff_features(repo_dir: str, ref_a: str, ref_b: str, cap=6_000_000):
    r = run_git(["diff", "--no-color", f"{ref_a}..{ref_b}", "--"] + DIFF_PATHS,
                repo_dir, timeout=120)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    diff = r.stdout[:cap]
    features, _, _ = extract_diff_features(diff)
    return features


def collect_from_repo(repo_url: str, clones_dir: str, real_dir: str,
                      force: bool) -> int:
    """Scrape release-pair + docs-only samples from one repo. Returns count."""
    name = repo_url.split("/")[-1].removesuffix(".git")
    out_path = os.path.join(real_dir, f"{name}.jsonl")
    if os.path.exists(out_path) and not force:
        return len(open(out_path, encoding="utf-8").read().splitlines())

    clone_dir = os.path.join(clones_dir, name)
    try:
        if not os.path.exists(clone_dir):
            print(f"    cloning (depth 120)...")
            r = subprocess.run(["git", "clone", "--depth=120", "--quiet",
                                repo_url, clone_dir],
                               capture_output=True, timeout=150)
            if r.returncode != 0:
                print(f"    clone failed: {r.stderr.decode(errors='replace')[:120]}")
                return 0
        elif not force:
            r = run_git(["fetch", "--depth=120", "--quiet", "origin", "+refs/heads/*:refs/heads/*"],
                        clone_dir, timeout=90)

        samples = []

        # --- release pairs (real patch/minor/major) -----------------------
        tags = repo_tags(clone_dir)
        for (v_old, c_old), (v_new, c_new) in zip(tags, tags[1:]):
            bump = classify_bump(v_old, v_new)
            if bump == "none":      # equal or decreasing version → skip
                continue
            feats = diff_features(clone_dir, c_old, c_new)
            if feats is None or feats["total_files"] == 0:
                continue
            samples.append({"features": feats, "label": bump, "repo": name,
                            "old_version": v_old, "new_version": v_new,
                            "kind": "release"})

        # --- docs/test/config-only commits (real "none") -------------------
        log = run_git(["log", "--format=%H%x09%s"], clone_dir)
        n_none = 0
        for line in log.stdout.splitlines():
            if n_none >= 8:
                break
            parts = line.split("\t", 1)
            if len(parts) < 2 or parse_semver(parts[1]):
                continue
            feats = diff_features(clone_dir, f"{parts[0]}~1", parts[0])
            if feats is None:
                continue
            if (feats["total_files"] > 0 and feats["num_source_files"] == 0
                    and feats["num_changed_symbols"] == 0):
                samples.append({"features": feats, "label": "none", "repo": name,
                                "old_version": None, "new_version": None,
                                "kind": "docs_only"})
                n_none += 1

        save_jsonl(samples, out_path)
        return len(samples)
    except subprocess.TimeoutExpired:
        print(f"    timeout — skipped ({name})")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"    error: {e}")
        return 0


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
            "new_version": sample.get("new_version"), "kind": "jitter"}


def gen_synthetic_v2(n: int, seed: int) -> list:
    """Harder synthetic data: overlapping class-conditional distributions."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
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
                     "repo": "synthetic-v2", "kind": "synthetic"})
    return data


def stats(title: str, rows: list):
    print(f"\n  {title}: {len(rows)} samples")
    for k, c in sorted(Counter(r["label"] for r in rows).items()):
        print(f"    {k:<6} {c:>5} ({c/len(rows)*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Prepare SemVerdict dataset v3 (real + synthetic)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--synthetic", type=int, default=1200)
    ap.add_argument("--synthetic-test", type=int, default=250)
    ap.add_argument("--jitter-copies", type=int, default=2)
    ap.add_argument("--test-repos", default="colinhacks/zod,axios/axios,ai/nanoid",
                    help="comma-separated repo slugs held out from training")
    ap.add_argument("--force", action="store_true", help="re-scrape cached repos")
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

    # ── scrape real data (incremental, cached) ───────────────────────────
    print(f"Scraping {len(REPOS)} repos (holdout: {sorted(holdout)})...")
    real_all = []
    for url in REPOS:
        name = url.split("/")[-1].removesuffix(".git")
        n = collect_from_repo(url, clones_dir, real_dir, args.force)
        print(f"  {name:<12} {n:>3} samples")
    for fn in os.listdir(real_dir):
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
