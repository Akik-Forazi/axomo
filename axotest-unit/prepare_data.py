#!/usr/bin/env python3
"""
Prepare training data for the AXOTEST unit test generator.

Collects (source code, test code) pairs from GitHub repos that have
test directories. For each source file, finds the corresponding test
file and creates a training pair.

Usage:
    python prepare_data.py --output ./data/ --repos 200
"""
import argparse, os, sys, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_jsonl, train_test_split, print_stats

def find_test_pairs(repo_dir: str) -> list:
    """Find (source, test) pairs in a repo."""
    pairs = []
    for root, dirs, files in os.walk(repo_dir):
        if 'node_modules' in root or '.git' in root:
            continue
        for f in files:
            if f.endswith('.test.ts') or f.endswith('.spec.ts'):
                test_path = os.path.join(root, f)
                # Derive source file name
                base = re.sub(r'\.(test|spec)\.ts$', '.ts', f)
                src_path = os.path.join(root, base)
                if not os.path.exists(src_path):
                    # Try src/ directory
                    src_path = os.path.join(root, 'src', base)
                if os.path.exists(src_path):
                    try:
                        src = open(src_path, encoding='utf-8').read()[:2000]
                        test = open(test_path, encoding='utf-8').read()[:2000]
                        if len(src) > 50 and len(test) > 50:
                            pairs.append({
                                "input": src,
                                "output": test,
                                "source_file": os.path.basename(src_path),
                            })
                    except: pass
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repos", type=int, default=200)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    repos = [
        "https://github.com/vercel/swr.git",
        "https://github.com/tannerlinsley/react-query.git",
        "https://github.com/jquense/yup.git",
        "https://github.com/colinhacks/zod.git",
        "https://github.com/Unitech/pm2.git",
    ]

    all_data = []
    for repo_url in repos:
        name = repo_url.split("/")[-1].replace(".git", "")
        clone_dir = os.path.join(args.output, "_clones", name)
        print(f"Scraping {name}...")
        try:
            if not os.path.exists(clone_dir):
                subprocess.run(["git", "clone", "--depth=50", repo_url, clone_dir],
                             capture_output=True, timeout=60)
            pairs = find_test_pairs(clone_dir)
            all_data.extend(pairs)
            print(f"  Found {len(pairs)} pairs (total: {len(all_data)})")
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    train, test = train_test_split(all_data)
    save_jsonl(train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(test, os.path.join(args.output, "test.jsonl"))
    print_stats(train, test, "output")
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()
