#!/usr/bin/env python3
"""
Prepare training data for the AXOVB explainer — a small language model
that generates natural language explanations instead of hardcoded strings.

Instead of:
  "Exported symbol(s) removed/changed — 15 callers affected. Breaking change."

The model generates:
  "I checked the code graph and found that `authMiddleware` was removed from
  the public API. This breaks 15 callers across 3 modules — the auth flow,
  rate limiter, and session manager. This is a breaking change, so I'm
  recommending a major version bump."

Task: conditional text generation
Input: structured features + axodex impact summary
Output: 1-3 sentence natural language explanation

Usage:
    python prepare_explainer_data.py --output ./data/
"""
import argparse, os, sys, random, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_jsonl, train_test_split

random.seed(42)

# Templates for generating natural language explanations from features
# These are the "ground truth" — the model learns to generate like these
EXPLANATION_TEMPLATES = {
    "major": [
        "Removed export `{symbol}` from the public API. This breaks {callers} callers across {modules} modules — the {module_list}. This is a breaking change, so I'm recommending a major version bump.",
        "Breaking change detected: the signature of `{symbol}` changed, affecting {callers} callers. The callers in {module_list} will need to be updated. Recommending major bump.",
        "The export `{symbol}` was removed and {callers} callers depend on it across {modules} modules. This will break existing code that imports it. Major version bump required.",
        "I found {symbols} changed symbols including removed exports. The blast radius is {callers} callers in {modules} modules. This is a breaking change — major bump.",
        "Breaking: `{symbol}` was modified in a way that changes its public contract. {callers} callers across {modules} modules will be affected. Major bump needed.",
    ],
    "minor": [
        "New export `{symbol}` was added to the public API. This is backwards compatible — existing callers aren't affected. {callers} existing callers found, none broken. Recommending minor bump.",
        "Added new functionality: `{symbol}` is now exported. No existing code breaks, but this is a new feature. Minor version bump.",
        "I detected {symbols} new exported symbols. These are additions, not changes to existing exports. {callers} callers exist but none are affected. Minor bump.",
        "New feature: `{symbol}` exported with 0 callers yet. Backwards compatible addition. Recommending minor version bump.",
        "Found {symbols} new exports and 0 removed exports. All changes are additions. Minor bump — backwards compatible.",
    ],
    "patch": [
        "Modified {symbols} internal symbol(s) with {callers} callers. No exports were changed — this is an internal implementation detail. Patch bump.",
        "Internal change: `{symbol}` was modified but it's not part of the public API. {callers} callers exist but they're all internal. Patch bump.",
        "I found {symbols} changed symbols but none are exported. The blast radius is {callers} internal callers. This is a non-breaking implementation change. Patch bump.",
        "Only internal symbols changed — {symbols} symbol(s), {callers} callers, 0 exports affected. Patch version bump.",
        "Implementation detail: `{symbol}` was refactored internally. No public API impact. {callers} callers updated. Patch bump.",
    ],
    "none": [
        "Only {doc_files} documentation file(s) and {config_files} config file(s) changed. No code symbols were modified. No version bump needed.",
        "No source files changed — only docs and config. The code graph shows 0 changed symbols. No bump required.",
        "I checked the code graph: 0 symbols changed. Only {doc_files} doc files and {config_files} config files were modified. No version bump.",
        "Changes are documentation-only ({doc_files} files). No code impact detected. No bump needed.",
        "All changes are in test files ({test_files} files) or docs ({doc_files} files). No production code changed. No version bump.",
    ],
}

def generate_explanation(features: dict, label: str) -> str:
    """Generate a natural language explanation from structured features."""
    templates = EXPLANATION_TEMPLATES.get(label, ["Unknown change type."])
    template = random.choice(templates)

    # Pick a representative symbol name (synthetic for now)
    symbols = ["authMiddleware", "validateToken", "createUser", "fetchData",
               "parseConfig", "renderComponent", "handleError", "dbQuery",
               "routeHandler", "serializeData"]

    return template.format(
        symbol=random.choice(symbols),
        callers=features.get("num_callers_affected", 0),
        modules=random.randint(1, 5),
        module_list=", ".join(random.sample(
            ["auth", "rate limiter", "session manager", "API router", "database layer"],
            k=min(3, random.randint(1, 4)))),
        symbols=features.get("num_changed_symbols", 0),
        doc_files=features.get("num_doc_files", 0),
        config_files=features.get("num_config_files", 0),
        test_files=features.get("num_test_files", 0),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=2000, help="Number of samples to generate")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    data = []
    for _ in range(args.n):
        bump = random.choices(["patch", "minor", "major", "none"], weights=[35, 25, 15, 25])[0]

        # Generate realistic features for this bump type
        if bump == "major":
            features = {
                "num_changed_symbols": random.randint(5, 30),
                "num_callers_affected": random.randint(10, 100),
                "has_removed_exports": 1, "has_new_exports": random.choice([0, 1]),
                "has_exported_changes": 1, "has_signature_changes": random.choice([0, 1]),
                "num_source_files": random.randint(3, 15), "num_test_files": random.randint(0, 5),
                "num_doc_files": random.randint(0, 2), "num_config_files": random.randint(0, 1),
            }
        elif bump == "minor":
            features = {
                "num_changed_symbols": random.randint(3, 15),
                "num_callers_affected": random.randint(0, 20),
                "has_removed_exports": 0, "has_new_exports": 1,
                "has_exported_changes": 1, "has_signature_changes": 0,
                "num_source_files": random.randint(2, 8), "num_test_files": random.randint(1, 5),
                "num_doc_files": random.randint(0, 2), "num_config_files": random.randint(0, 1),
            }
        elif bump == "patch":
            features = {
                "num_changed_symbols": random.randint(1, 8),
                "num_callers_affected": random.randint(0, 10),
                "has_removed_exports": 0, "has_new_exports": 0,
                "has_exported_changes": random.choice([0, 0, 0, 1]),
                "has_signature_changes": random.choice([0, 0, 1]),
                "num_source_files": random.randint(1, 5), "num_test_files": random.randint(0, 3),
                "num_doc_files": random.randint(0, 1), "num_config_files": random.randint(0, 1),
            }
        else:
            features = {
                "num_changed_symbols": random.randint(0, 2),
                "num_callers_affected": 0,
                "has_removed_exports": 0, "has_new_exports": 0,
                "has_exported_changes": 0, "has_signature_changes": 0,
                "num_source_files": 0, "num_test_files": random.randint(0, 2),
                "num_doc_files": random.randint(1, 5), "num_config_files": random.randint(1, 3),
            }

        features["total_files"] = features["num_source_files"] + features["num_test_files"] + features["num_doc_files"] + features["num_config_files"]
        features["test_to_source_ratio"] = round(features["num_test_files"] / max(features["num_source_files"], 1), 2)
        features["blast_radius"] = features["num_callers_affected"]
        features["change_scope"] = min(features["num_changed_symbols"], 50)

        # Generate the explanation
        explanation = generate_explanation(features, bump)

        # Format as prompt → response for language model training
        prompt = f"Features: {json.dumps(features)}\nBump: {bump}\nExplanation:"
        data.append({
            "prompt": prompt,
            "response": f" {explanation}",
            "label": bump,
            "features": features,
        })

    train, test = train_test_split(data)
    save_jsonl(train, os.path.join(args.output, "explainer_train.jsonl"))
    save_jsonl(test, os.path.join(args.output, "explainer_test.jsonl"))

    print(f"Generated {len(train)} train + {len(test)} test samples")
    print(f"\nSample prompt:\n  {data[0]['prompt'][:100]}...")
    print(f"\nSample response:\n  {data[0]['response'][:150]}...")
    print(f"\nThe model learns: structured features + bump type → natural language explanation")
    print(f"No more hardcoded strings — the model generates contextual explanations.")

if __name__ == "__main__":
    main()
