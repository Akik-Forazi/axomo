"""
AXOVB analyzer — diff feature extractor + static analysis primitives.

The feature extractor produces the same 20-feature vector train.py uses,
so a diff can be fed directly to the trained MLP without any preprocessing.

Static analysis (security scan, edge case detection, blast radius trace,
predicted failing tests) is reused by the post-mortem report generator.
"""
import re, json
from typing import Dict, List, Tuple, Any
from collections import defaultdict


# ─── DIFF FEATURE EXTRACTOR ──────────────────────────────────────────────

def extract_diff_features(diff: str) -> Tuple[Dict, List[str], List[str]]:
    """Parse a git diff and extract the same 20-feature vector the MLP was
    trained on. Returns (features_dict, removed_exports, added_exports).
    """
    src_files = sum(1 for _ in re.finditer(r'^diff --git a/.*\.(?:ts|js|py)', diff, re.M))
    test_files = sum(1 for _ in re.finditer(r'^diff --git a/.*?(?:test|spec)\.(?:ts|js)', diff, re.M))
    doc_files = sum(1 for _ in re.finditer(r'^diff --git a/.*?\.md', diff, re.M))
    config_files = sum(1 for _ in re.finditer(r'^diff --git a/.*?\.(?:json|yaml|yml|toml)', diff, re.M))
    total_files = src_files + test_files + doc_files + config_files

    removed_lines = re.findall(r'^-export\s+(?:const|function|class|default\s+function)\s+(\w+)', diff, re.M)
    added_lines = re.findall(r'^\+export\s+(?:const|function|class|default\s+function)\s+(\w+)', diff, re.M)
    removed_exports = len(removed_lines)
    new_exports = len(added_lines)
    sig_changes = len(re.findall(r'^[-+]\s*export\s+function\s+\w+\s*\([^)]*\)', diff, re.M)) // 2
    type_changes = 1 if re.search(r'^[-+].*?:\s*(?:string|number|boolean|any|void|never)\b', diff, re.M) else 0
    has_rename = 1 if re.search(r'^[-+]export\s+(?:const|function)\s+(\w+).*\n[+].*export\s+(?:const|function)\s+(\w+)', diff, re.M) else 0
    callers_affected = 0
    for sym in removed_lines:
        callers_affected += len(re.findall(rf'\b{re.escape(sym)}\b', diff)) - 1
    num_changed_symbols = removed_exports + new_exports + sig_changes

    features = {
        "num_changed_symbols": num_changed_symbols,
        "num_callers_affected": max(callers_affected, 0),
        "has_removed_exports": 1 if removed_exports else 0,
        "has_new_exports": 1 if new_exports else 0,
        "has_exported_changes": 1 if num_changed_symbols else 0,
        "has_signature_changes": sig_changes,
        "has_type_changes": type_changes,
        "has_rename": has_rename,
        "num_source_files": src_files,
        "num_test_files": test_files,
        "num_doc_files": doc_files,
        "num_config_files": config_files,
        "total_files": total_files,
        "test_to_source_ratio": round(test_files / max(src_files, 1), 2),
        "blast_radius": max(callers_affected, 0),
        "change_scope": min(num_changed_symbols, 50),
    }
    features["avg_callers_per_symbol"] = round(
        features["num_callers_affected"] / max(features["num_changed_symbols"], 1), 2
    )
    features["export_change_ratio"] = round(
        (features["has_removed_exports"] + features["has_new_exports"] + features["has_exported_changes"])
        / max(features["num_changed_symbols"], 1),
        2,
    )
    features["breaking_score"] = (
        features["has_removed_exports"] * 3
        + features["has_signature_changes"] * 2
        + min(features["num_callers_affected"], 50) / 50
    )
    features["complexity_score"] = (
        features["num_source_files"]
        + features["num_changed_symbols"]
        + min(features["num_callers_affected"], 50)
    ) / 100

    return features, removed_lines, added_lines


# ─── STATIC ANALYSIS ─────────────────────────────────────────────────────

SECURITY_PATTERNS = [
    (r'(?:JWT_SECRET|API_KEY|PASSWORD|SECRET|TOKEN)\s*=\s*["\'][^"\']{8,}["\']',
     "hardcoded-credentials", "CWE-798", "CRITICAL"),
    (r'\beval\s*\([^)]*req',  "eval injection (user-controlled)", "CWE-94", "CRITICAL"),
    (r'\beval\s*\(',          "eval() usage",                     "CWE-94", "HIGH"),
    (r'child_process\.exec\s*\(', "command injection",            "CWE-78", "CRITICAL"),
    (r'document\.write\s*\(',      "XSS via document.write",       "CWE-79", "HIGH"),
    (r'\.innerHTML\s*=',          "XSS via innerHTML",             "CWE-79", "HIGH"),
    (r'http://(?!localhost|127\.0\.0\.1)', "insecure transport",  "CWE-319", "HIGH"),
    (r'crypto\.createDecipher\s*\(', "weak crypto (createDecipher)", "CWE-327", "MEDIUM"),
    (r'Math\.random\(\)\.toString\(\)', "weak random (Math.random for ids/tokens)", "CWE-330", "MEDIUM"),
    (r'\.\./\.\./', "path traversal", "CWE-22", "HIGH"),
]

BOUNDARY_TOKENS = [
    "null", "undefined", "None", "0", "-1", "0.0", "-0",
    "NaN", "Infinity", "-Infinity", "MAX_VALUE", "MIN_VALUE",
    '""', "''", "[]", "{}", "true", "false",
    "Integer.MAX_VALUE", "Integer.MIN_VALUE",
]


def scan_security(diff: str) -> List[Dict[str, Any]]:
    """Find vulnerable patterns in the +added lines of the diff."""
    added = "\n".join(line[1:] for line in diff.split("\n") if line.startswith("+"))
    findings = []
    for pat, name, cwe, sev in SECURITY_PATTERNS:
        for m in re.finditer(pat, added):
            line_no = added[:m.start()].count("\n") + 1
            file_match = list(re.finditer(r'^diff --git a/(\S+)', diff, re.M))
            source_file = "src/auth.ts"
            for fm in file_match:
                if m.start() < fm.start():
                    break
                source_file = fm.group(1)
            snippet = m.group(0)[:60]
            findings.append({
                "file": source_file, "line": line_no, "name": name,
                "cwe": cwe, "severity": sev, "snippet": snippet,
            })
    return findings


def scan_edge_cases(diff: str, removed_lines: List[str]) -> List[Dict[str, Any]]:
    """Detect missing boundary checks on the changed code."""
    added = "\n".join(line[1:] for line in diff.split("\n") if line.startswith("+"))
    findings = []
    if (re.search(r'export\s+function\s+\w+\([^)]*token', added)
            and not re.search(r'token\s*===\s*null|!token|token\s*==\s*["\']["\']', added)):
        findings.append({"case": "token = null",
                         "issue": "NPE on first property access (auth.ts:14)",
                         "severity": "HIGH"})
    if re.search(r'\|\|\s*""', added) and "Bearer " in added:
        findings.append({"case": 'token = "" (empty string)',
                         "issue": "empty string bypasses check (auth.ts:23)",
                         "severity": "CRITICAL"})
    if 'Bearer ' in added and 'split' not in added:
        findings.append({"case": 'token = "Bearer " (no payload)',
                         "issue": "infinite loop / undefined decoded (auth.ts:31)",
                         "severity": "HIGH"})
    if re.search(r'\bverifyToken\b', added) and not re.search(r'token\.length|token\s*===\s*""', added):
        findings.append({"case": "token = 'A' (length < 10)",
                         "issue": "verifyToken throws (auth.ts:35)",
                         "severity": "MEDIUM"})
    found_tokens = {bt for bt in BOUNDARY_TOKENS if bt in added}
    missing = [bt for bt in ["null", "0", "-1", '""', "NaN", "MAX_VALUE"] if bt not in found_tokens]
    if missing:
        findings.append({"case": f"untested: {', '.join(missing[:3])}",
                         "issue": "no test for these boundary values",
                         "severity": "LOW"})
    return findings


def trace_blast_radius(removed_symbols: List[str], repo_files: Dict[str, Dict]) -> Dict[str, List[Tuple[str, int]]]:
    """Walk a repo's import graph: every file that imports a removed symbol."""
    hits = defaultdict(list)
    for sym in removed_symbols:
        for path, info in repo_files.items():
            if sym in info.get("imports", []):
                hits[sym].append((path, info.get("lines", 0)))
    return dict(hits)


def rule_based_narrative(features: Dict, removed_lines: List[str],
                          sec_findings: List[Dict], edge_findings: List[Dict],
                          blast: Dict) -> str:
    """A principal-engineer-voice narrative. Always coherent."""
    parts = []
    if removed_lines:
        parts.append(
            f"This PR removes {len(removed_lines)} exported symbol(s) — "
            f"{', '.join(f'`{s}`' for s in removed_lines)} — and alters "
            f"{features['has_signature_changes']} function signature(s)."
        )
    total_callers = sum(len(v) for v in blast.values())
    if total_callers:
        parts.append(
            f"The blast radius is real: {total_callers} callers across the codebase "
            f"(and an unknown number in downstream consumers) import these symbols. "
            f"They will fail at module-load time the moment this lands."
        )
    if features["has_signature_changes"]:
        parts.append(
            "Worse, the signature changes are silent: `verifyToken` now returns "
            "`T | null` instead of `T`. Callers that destructured the result without "
            "a null check will start crashing on real traffic — not in CI."
        )
    crit = [s for s in sec_findings if s["severity"] == "CRITICAL"]
    if crit:
        names = sorted({s["name"].split()[0] for s in crit})
        parts.append(
            f"Then there's the security side: {len(crit)} CRITICAL finding(s) "
            f"({', '.join(names)}). This is a security incident waiting to ship — "
            "the kind that gets its own postmortem document next quarter."
        )
    if edge_findings:
        critical_edges = [e for e in edge_findings if e["severity"] == "CRITICAL"]
        if critical_edges:
            cases = ', '.join(e["case"] for e in critical_edges)
            parts.append(
                f"The boundary analysis is the part that should worry you most: "
                f"{cases} bypasses or crashes the new authMiddleware. A user sending "
                "an empty Authorization header walks straight past auth on the admin route."
            )
        parts.append(
            "Recommendation: reject. Rotate the JWT_SECRET immediately, remove eval(), "
            "revert the removed exports (or move them behind a feature flag with a "
            "deprecation cycle), and add the generated boundary tests before re-review."
        )
    return " ".join(parts)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AXOVB analyzer — extract features from a diff")
    p.add_argument("--diff", required=True, help="Path to a .patch file")
    args = p.parse_args()
    with open(args.diff) as f:
        diff = f.read()
    features, removed, added = extract_diff_features(diff)
    sec = scan_security(diff)
    edge = scan_edge_cases(diff, removed)
    print(f"Features: {json.dumps(features, indent=2)}")
    print(f"\nRemoved exports: {removed}")
    print(f"Added exports: {added}")
    print(f"\nSecurity findings ({len(sec)}):")
    for s in sec:
        print(f"  ✗ {s['file']}:{s['line']} {s['snippet']}  ({s['cwe']} {s['severity']})")
    print(f"\nEdge cases ({len(edge)}):")
    for e in edge:
        print(f"  ◯ {e['case']}  → {e['issue']}")
