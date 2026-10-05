#!/usr/bin/env python3
"""
Prepare training data for the AXOTEST edge case detector.

Collects code with labeled boundary tokens (null, undefined, 0, -1, max, min, empty, etc.).
"""
import argparse, os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_jsonl, train_test_split, print_stats

BOUNDARY_TOKENS = [
    "null", "undefined", "None", "0", "-1", "1", "max", "min", "MAX_VALUE",
    "MIN_VALUE", "empty", "[]", "{}", "NaN", "Infinity", "-Infinity",
    "true", "false", "0.0", "-0", "Integer.MAX_VALUE", "Integer.MIN_VALUE",
    "Float.MAX_VALUE", "None", "nil", "void 0",
]

def label_tokens(code: str) -> list:
    """Label boundary tokens in code using BIO scheme."""
    tokens = code.split()  # Simple whitespace tokenization
    labels = []
    for token in tokens:
        is_boundary = any(bt in token for bt in BOUNDARY_TOKENS)
        labels.append("B-BOUNDARY" if is_boundary else "O")
    return list(zip(tokens, labels))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repos", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    import subprocess
    repos = [
        "https://github.com/lodash/lodash.git",
        "https://github.com/ramda/ramda.git",
        "https://github.com/jquense/yup.git",
        "https://github.com/colinhacks/zod.git",
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
            for root, dirs, files in os.walk(clone_dir):
                if '.git' in root or 'node_modules' in root:
                    continue
                for f in files:
                    if not f.endswith(('.ts', '.js', '.py')):
                        continue
                    try:
                        code = open(os.path.join(root, f), encoding='utf-8').read()[:2000]
                        if len(code) < 50:
                            continue
                        tokens_labels = label_tokens(code)
                        if any(l != "O" for _, l in tokens_labels):
                            all_data.append({
                                "tokens": [t for t, _ in tokens_labels],
                                "labels": [l for _, l in tokens_labels],
                                "file": f,
                            })
                    except: pass
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    train, test = train_test_split(all_data)
    save_jsonl(train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(test, os.path.join(args.output, "test.jsonl"))
    print_stats(train, test, "labels")
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()
