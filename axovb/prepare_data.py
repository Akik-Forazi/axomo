#!/usr/bin/env python3
"""
Prepare training data for the AXOVB bump classifier.

Scrapes npm publish history to build (git diff → bump type) pairs.
For each npm package that uses semver, we:
  1. Find all version publish events
  2. For each publish, get the git diff since the previous version
  3. Label the diff with the bump type (patch/minor/major)

Usage:
    python prepare_data.py --output ./data/ --repos 1000
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add shared utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_jsonl, print_stats

def classify_bump(old_version: str, new_version: str) -> str:
    """Determine bump type from version change."""
    old_parts = old_version.split(".")
    new_parts = new_version.split(".")
    # Remove prerelease suffixes
    old_parts = [p.split("-")[0] for p in old_parts]
    new_parts = [p.split("-")[0] for p in new_parts]
    if len(old_parts) != 3 or len(new_parts) != 3:
        return "none"
    try:
        old_major, old_minor, old_patch = int(old_parts)
        new_major, new_minor, new_patch = int(new_parts)
    except ValueError:
        return "none"
    if new_major > old_major:
        return "major"
    if new_minor > old_minor:
        return "minor"
    if new_patch > old_patch:
        return "patch"
    return "none"

def collect_from_repo(repo_url: str, output_dir: str) -> list:
    """Collect (diff → label) pairs from a single repo."""
    data = []
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_dir = os.path.join(output_dir, "_clones", repo_name)

    try:
        if not os.path.exists(clone_dir):
            subprocess.run(
                ["git", "clone", "--depth=500", repo_url, clone_dir],
                capture_output=True, timeout=60
            )

        # Get all version-bump commits
        result = subprocess.run(
            ["git", "log", "--oneline", "--grep=version\\|bump\\|chore(release",
             "-i", "--format=%H %s"],
            cwd=clone_dir, capture_output=True, text=True, timeout=30
        )

        commits = result.stdout.strip().split("\n") if result.stdout.strip() else []
        commits = [c for c in commits if c.strip()][:50]

        for i, commit_line in enumerate(commits):
            parts = commit_line.split(" ", 1)
            if len(parts) < 2:
                continue
            commit_hash, message = parts

            # Try to extract old and new version from the commit message
            import re
            versions = re.findall(r'(\d+\.\d+\.\d+)', message)
            if len(versions) < 2:
                continue

            old_ver, new_ver = versions[0], versions[1]
            bump_type = classify_bump(old_ver, new_ver)
            if bump_type == "none":
                continue

            # Get the diff since the previous version commit
            if i + 1 < len(commits):
                prev_hash = commits[i + 1].split(" ")[0]
                diff_result = subprocess.run(
                    ["git", "diff", f"{prev_hash}..{commit_hash}", "--stat"],
                    cwd=clone_dir, capture_output=True, text=True, timeout=30
                )
                diff = diff_result.stdout.strip()
                if diff and len(diff) > 50:
                    data.append({
                        "text": diff[:2000],  # Truncate for model input
                        "label": bump_type,
                        "repo": repo_name,
                        "old_version": old_ver,
                        "new_version": new_ver,
                    })
    except Exception as e:
        print(f"  Warning: {repo_name} failed: {e}")
    finally:
        pass  # Keep clones for reuse

    return data

def main():
    parser = argparse.ArgumentParser(description="Prepare AXOVB training data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--repos", type=int, default=500, help="Number of repos to scrape")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Popular npm packages to scrape (seeds)
    seed_repos = [
        "https://github.com/facebook/react.git",
        "https://github.com/vuejs/vue.git",
        "https://github.com/expressjs/express.git",
        "https://github.com/lodash/lodash.git",
        "https://github.com/axios/axios.git",
        "https://github.com/chalk/chalk.git",
        "https://github.com/microsoft/TypeScript.git",
        "https://github.com/nodejs/node.git",
        "https://github.com/vercel/next.js.git",
        "https://github.com/tailwindlabs/tailwindcss.git",
    ]

    all_data = []
    for repo_url in seed_repos[:args.repos]:
        print(f"Scraping {repo_url}...")
        data = collect_from_repo(repo_url, args.output)
        all_data.extend(data)
        print(f"  Collected {len(data)} samples (total: {len(all_data)})")

    # Save
    save_jsonl(all_data, os.path.join(args.output, "train.jsonl"))

    # Split
    from utils import train_test_split
    train, test = train_test_split(all_data)
    save_jsonl(train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(test, os.path.join(args.output, "test.jsonl"))

    print_stats(train, test, "label")
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()
