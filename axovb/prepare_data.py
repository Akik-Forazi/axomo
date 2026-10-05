#!/usr/bin/env python3
"""
Prepare training data for the AXOVB bump classifier.

v2: Extracts STRUCTURED FEATURES from axodex output, not raw text.
The model learns from features like:
  - num_changed_symbols: 5
  - num_callers_affected: 12
  - has_removed_exports: true
  - has_new_exports: false
  - num_test_files_changed: 0
  - num_source_files_changed: 3

NOT from raw text like DistilBERT would need.

This means the model can be a tiny MLP (1-5M params) instead of
DistilBERT (66M). Inference drops from 32ms to ~1ms. Model size
drops from 256MB to ~1-5MB.

Usage:
    python prepare_data.py --output ./data/ --repos 100
"""
import argparse, os, sys, subprocess, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_jsonl, train_test_split

# Feature extraction from axodex output
def extract_features(detect_output: str, impact_outputs: list, git_stat: str) -> dict:
    """Extract structured features from axodex + git output."""
    symbols = [s.strip() for s in detect_output.split("\n") if s.strip() and not s.startswith("[")]

    # Count callers across all impact reports
    total_callers = 0
    has_removed = False
    has_new = False
    has_exported = False
    has_signature_change = False

    for impact in impact_outputs:
        lines = [l for l in impact.split("\n") if l.strip() and not l.startswith("[")]
        total_callers += len(lines)
        lower = impact.lower()
        if "removed" in lower or "deleted" in lower or "breaking" in lower:
            has_removed = True
        if "new" in lower or "added" in lower or "created" in lower:
            has_new = True
        if "export" in lower or "public" in lower or "external" in lower:
            has_exported = True
        if "signature" in lower or "parameter" in lower or "return type" in lower:
            has_signature_change = True

    # File type analysis from git stat
    source_files = 0
    test_files = 0
    doc_files = 0
    config_files = 0
    for line in git_stat.split("\n"):
        if "|" not in line and "=>" not in line:
            continue
        filename = line.split("|")[0].strip().split("=>")[-1].strip()
        if filename.endswith((".test.ts", ".spec.ts", ".test.js", "_test.py", "test_*.py")):
            test_files += 1
        elif filename.endswith((".ts", ".js", ".py", ".go", ".rs", ".java")):
            source_files += 1
        elif filename.endswith((".md", ".txt")):
            doc_files += 1
        elif filename.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".env")):
            config_files += 1

    return {
        # Symbol-level features
        "num_changed_symbols": len(symbols),
        "num_callers_affected": total_callers,
        "has_removed_exports": int(has_removed),
        "has_new_exports": int(has_new),
        "has_exported_changes": int(has_exported),
        "has_signature_changes": int(has_signature_change),
        # File-level features
        "num_source_files": source_files,
        "num_test_files": test_files,
        "num_doc_files": doc_files,
        "num_config_files": config_files,
        # Derived features
        "total_files": source_files + test_files + doc_files + config_files,
        "test_to_source_ratio": round(test_files / max(source_files, 1), 2),
        "blast_radius": total_callers,
        "change_scope": min(len(symbols), 50),  # Capped
    }

def collect_from_repo(repo_url: str, output_dir: str) -> list:
    """Collect (features → bump type) pairs from a repo."""
    data = []
    name = repo_url.split("/")[-1].replace(".git", "")
    clone_dir = os.path.join(output_dir, "_clones", name)

    try:
        if not os.path.exists(clone_dir):
            subprocess.run(["git", "clone", "--depth=500", repo_url, clone_dir],
                         capture_output=True, timeout=120)

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

            prev_hash = commits[i + 1].split(" ")[0] if i + 1 < len(commits) else f"{commit_hash}~1"

            # Get git stat
            git_stat = subprocess.run(
                ["git", "diff", "--stat", f"{prev_hash}..{commit_hash}"],
                cwd=clone_dir, capture_output=True, text=True, timeout=30).stdout

            # Run axodex (if available)
            subprocess.run(["git", "checkout", commit_hash, "--quiet"],
                         cwd=clone_dir, capture_output=True, timeout=30)

            detect = subprocess.run(["axodex", "detect_changes"],
                cwd=clone_dir, capture_output=True, text=True, timeout=60)

            detect_output = detect.stdout if detect.returncode == 0 else ""

            # Run impact for top symbols
            symbols = [s.strip() for s in detect_output.split("\n") if s.strip() and not s.startswith("[")][:10]
            impact_outputs = []
            for sym in symbols:
                impact = subprocess.run(["axodex", "impact", sym, "--direction", "upstream"],
                    cwd=clone_dir, capture_output=True, text=True, timeout=30)
                if impact.returncode == 0 and impact.stdout.strip():
                    impact_outputs.append(impact.stdout)

            # If axodex isn't available, still extract features from git stat alone
            if not detect_output:
                # Synthetic detect output from git stat
                detect_output = "\n".join(
                    line.split("|")[0].strip() for line in git_stat.split("\n")
                    if "|" in line and line.split("|")[0].strip().endswith((".ts", ".js", ".py"))
                )[:500]

            features = extract_features(detect_output, impact_outputs, git_stat)

            data.append({
                "features": features,
                "label": bump_type,
                "repo": name,
                "old_version": old_ver,
                "new_version": new_ver,
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
    parser = argparse.ArgumentParser(description="Prepare AXOVB training data (structured features)")
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
        "https://github.com/axios/axios.git",
        "https://github.com/chalk/chalk.git",
        "https://github.com/tailwindlabs/tailwindcss.git",
    ]

    all_data = []
    for repo_url in repos[:args.repos]:
        print(f"Scraping {repo_url}...")
        data = collect_from_repo(repo_url, args.output)
        all_data.extend(data)
        print(f"  Collected {len(data)} samples (total: {len(all_data)})")

    # Also generate synthetic data to boost dataset size
    print(f"\nGenerating synthetic training data...")
    synthetic = generate_synthetic_data(500)
    all_data.extend(synthetic)
    print(f"  Added {len(synthetic)} synthetic samples (total: {len(all_data)})")

    train, test = train_test_split(all_data)
    save_jsonl(train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(test, os.path.join(args.output, "test.jsonl"))

    print(f"\nDataset statistics:")
    print(f"  Train: {len(train)} samples")
    print(f"  Test:  {len(test)} samples")
    from collections import Counter
    labels = Counter(item["label"] for item in train)
    print(f"  Label distribution (train):")
    for label, count in labels.most_common():
        print(f"    {label}: {count} ({count/len(train)*100:.1f}%)")
    print(f"\nFeature count: {len(all_data[0]['features']) if all_data else 0}")
    print(f"Features: {list(all_data[0]['features'].keys()) if all_data else []}")
    print(f"\nData saved to {args.output}")
    print(f"Model type: tiny MLP (not DistilBERT) — ~1-5M params, ~1ms inference")

def generate_synthetic_data(n: int) -> list:
    """Generate synthetic training data with realistic feature patterns."""
    import random
    random.seed(42)
    data = []
    for _ in range(n):
        bump = random.choices(["patch", "minor", "major", "none"], weights=[40, 25, 15, 20])[0]
        if bump == "major":
            f = {
                "num_changed_symbols": random.randint(5, 30),
                "num_callers_affected": random.randint(10, 100),
                "has_removed_exports": 1,
                "has_new_exports": random.choice([0, 1]),
                "has_exported_changes": 1,
                "has_signature_changes": 1,
                "num_source_files": random.randint(3, 15),
                "num_test_files": random.randint(0, 5),
                "num_doc_files": random.randint(0, 2),
                "num_config_files": random.randint(0, 1),
                "total_files": 0, "test_to_source_ratio": 0, "blast_radius": 0, "change_scope": 0,
            }
        elif bump == "minor":
            f = {
                "num_changed_symbols": random.randint(3, 15),
                "num_callers_affected": random.randint(0, 20),
                "has_removed_exports": 0,
                "has_new_exports": 1,
                "has_exported_changes": 1,
                "has_signature_changes": 0,
                "num_source_files": random.randint(2, 8),
                "num_test_files": random.randint(1, 5),
                "num_doc_files": random.randint(0, 2),
                "num_config_files": random.randint(0, 1),
                "total_files": 0, "test_to_source_ratio": 0, "blast_radius": 0, "change_scope": 0,
            }
        elif bump == "patch":
            f = {
                "num_changed_symbols": random.randint(1, 8),
                "num_callers_affected": random.randint(0, 10),
                "has_removed_exports": 0,
                "has_new_exports": 0,
                "has_exported_changes": random.choice([0, 0, 0, 1]),
                "has_signature_changes": random.choice([0, 0, 1]),
                "num_source_files": random.randint(1, 5),
                "num_test_files": random.randint(0, 3),
                "num_doc_files": random.randint(0, 1),
                "num_config_files": random.randint(0, 1),
                "total_files": 0, "test_to_source_ratio": 0, "blast_radius": 0, "change_scope": 0,
            }
        else:  # none
            f = {
                "num_changed_symbols": random.randint(0, 2),
                "num_callers_affected": 0,
                "has_removed_exports": 0,
                "has_new_exports": 0,
                "has_exported_changes": 0,
                "has_signature_changes": 0,
                "num_source_files": 0,
                "num_test_files": random.randint(0, 2),
                "num_doc_files": random.randint(1, 5),
                "num_config_files": random.randint(1, 3),
                "total_files": 0, "test_to_source_ratio": 0, "blast_radius": 0, "change_scope": 0,
            }
        f["total_files"] = f["num_source_files"] + f["num_test_files"] + f["num_doc_files"] + f["num_config_files"]
        f["test_to_source_ratio"] = round(f["num_test_files"] / max(f["num_source_files"], 1), 2)
        f["blast_radius"] = f["num_callers_affected"]
        f["change_scope"] = min(f["num_changed_symbols"], 50)
        data.append({"features": f, "label": bump, "repo": "synthetic"})
    return data

if __name__ == "__main__":
    main()
