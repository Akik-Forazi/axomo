#!/usr/bin/env python3
"""
Prepare training data for the AXOTEST security scanner.

Collects (code, vulnerable?) pairs from CVE databases + SAST tool outputs.
"""
import argparse, os, sys, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_jsonl, train_test_split, print_stats

# Patterns that indicate vulnerable code
VULN_PATTERNS = [
    (r"eval\s*\(", "injection"),
    (r"innerHTML\s*=", "xss"),
    (r"document\.write\s*\(", "xss"),
    (r"exec\s*\(\s*['\"]", "command-injection"),
    (r"child_process\.exec", "command-injection"),
    (r"sql.*\+.*['\"]", "sql-injection"),
    (r"password\s*=\s*['\"]", "hardcoded-credentials"),
    (r"api[_-]?key\s*=\s*['\"]", "hardcoded-credentials"),
    (r"http://(?!localhost)", "insecure-transport"),
    (r"fs\.readFile.*req\.", "path-traversal"),
    (r"\.\.\/\.\.\/", "path-traversal"),
    (r"crypto\.createDecipher\s*\(", "weak-crypto"),
    (r"Math\.random\s*\(\s*\).*token", "weak-random"),
    (r"setTimeout\s*\(\s*['\"]", "code-injection"),
    (r"new\s+Function\s*\(", "code-injection"),
]

def label_code(code: str) -> str:
    for pattern, _ in VULN_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return "vulnerable"
    return "secure"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repos", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    import subprocess
    repos = [
        "https://github.com/expressjs/express.git",
        "https://github.com/koajs/koa.git",
        "https://github.com/fastify/fastify.git",
        "https://github.com/pallets/flask.git",
        "https://github.com/django/django.git",
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
                    fpath = os.path.join(root, f)
                    try:
                        code = open(fpath, encoding='utf-8').read()[:2000]
                        if len(code) < 50:
                            continue
                        label = label_code(code)
                        all_data.append({"text": code, "label": label, "file": f})
                    except: pass
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    # Balance the dataset (undersample secure)
    secure = [d for d in all_data if d["label"] == "secure"]
    vulnerable = [d for d in all_data if d["label"] == "vulnerable"]
    import random
    random.seed(42)
    random.shuffle(secure)
    secure = secure[:max(len(vulnerable) * 3, 500)]
    all_data = secure + vulnerable

    train, test = train_test_split(all_data)
    save_jsonl(train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(test, os.path.join(args.output, "test.jsonl"))
    print_stats(train, test, "label")
    print(f"Data saved to {args.output}")

if __name__ == "__main__":
    main()
