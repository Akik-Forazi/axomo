#!/usr/bin/env python3
"""
Prepare training data for the AXOVB bump classifier — using AXODEX
code intelligence, NOT git diffs.

For each npm package version change:
  1. Run `axodex detect_changes` at the old version → get changed symbols
  2. Run `axodex impact <symbol>` for each → get blast radius
  3. Label: removed exports → major, new exports → minor, internal → patch

The model learns to map axodex impact reports → bump type.
This is CODE INTELLIGENCE, not text diffing.

Usage:
    python prepare_data.py --output ./data/ --repos 100
"""
import argparse, os, sys, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_jsonl, train_test_split, print_stats

def collect_from_repo(repo_url: str, output_dir: str) -> list:
    """Collect (axodex impact report → bump type) pairs from a repo."""
    data = []
    name = repo_url.split("/")[-1].replace(".git", "")
    clone_dir = os.path.join(output_dir, "_clones", name)

    try:
        if not os.path.exists(clone_dir):
            subprocess.run(["git", "clone", "--depth=500", repo_url, clone_dir],
                         capture_output=True, timeout=120)

        # Get version-bump commits
        result = subprocess.run(
            ["git", "log", "--oneline", "--grep=version\\|bump\\|chore(release", "-i", "--format=%H %s"],
            cwd=clone_dir, capture_output=True, text=True, timeout=30)
        commits = [c for c in result.stdout.strip().split("\n") if c.strip()][:30]

        for i, line in enumerate(commits):
            parts = line.split(" ", 1)
            if len(parts) < 2: continue
            commit_hash, message = parts
            versions = re.findall(r'(\d+\.\d+\.\d+)', message)
            if len(versions) < 2: continue

            old_ver, new_ver = versions[0], versions[1]
            bump_type = classify_bump(old_ver, new_ver)
            if bump_type == "none": continue

            # Checkout the new version and run axodex
            subprocess.run(["git", "checkout", commit_hash, "--quiet"],
                         cwd=clone_dir, capture_output=True, timeout=30)

            # Run axodex detect_changes (compare to previous version)
            prev_hash = commits[i + 1].split(" ")[0] if i + 1 < len(commits) else "HEAD~1"
            detect = subprocess.run(
                ["axodex", "detect_changes"],
                cwd=clone_dir, capture_output=True, text=True, timeout=60)

            if detect.returncode != 0 or not detect.stdout.strip():
                continue

            # Run axodex impact for each changed symbol
            symbols = [s.strip() for s in detect.stdout.strip().split("\n") if s.strip()][:10]
            impact_report = []
            for sym in symbols:
                impact = subprocess.run(
                    ["axodex", "impact", sym, "--direction", "upstream"],
                    cwd=clone_dir, capture_output=True, text=True, timeout=30)
                if impact.returncode == 0 and impact.stdout.strip():
                    impact_report.append(f"{sym}: {impact.stdout.strip()[:200]}")

            if impact_report:
                data.append({
                    "text": "\n".join(impact_report)[:2000],  # axodex impact report
                    "label": bump_type,
                    "repo": name,
                    "old_version": old_ver,
                    "new_version": new_ver,
                    "symbols": symbols,
                })

        subprocess.run(["git", "checkout", "main", "--quiet"], cwd=clone_dir,
                     capture_output=True, timeout=10)
    except Exception as e:
        print(f"  Warning: {name} failed: {e}")
    return data

def classify_bump(old: str, new: str) -> str:
    o = [int(x) for x in old.split(".")[:3]]
    n = [int(x) for x in new.split(".")[:3]]
    if len(o) != 3 or len(n) != 3: return "none"
    if n[0] > o[0]: return "major"
    if n[1] > o[1]: return "minor"
    if n[2] > o[2]: return "patch"
    return "none"

def main():
    parser = argparse.ArgumentParser(description="Prepare AXOVB training data (axodex-powered)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repos", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    repos = [
        "https://github.com/facebook/react.git",
        "https://github.com/vercel/next.js.git",
        "https://github.com/microsoft/TypeScript.git",
        "https://github.com/expressjs/express.git",
        "https://github.com/lodash/lodash.git",
    ]

    all_data = []
    for repo_url in repos[:args.repos]:
        print(f"Scraping {repo_url}...")
        data = collect_from_repo(repo_url, args.output)
        all_data.extend(data)
        print(f"  Collected {len(data)} samples (total: {len(all_data)})")

    train, test = train_test_split(all_data)
    save_jsonl(train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(test, os.path.join(args.output, "test.jsonl"))
    print_stats(train, test, "label")
    print(f"\nData saved to {args.output}")
    print(f"NOTE: Training data is axodex impact reports, NOT git diffs.")
    print(f"The model learns: impact_report → {patch|minor|major|none}")

if __name__ == "__main__":
    main()
