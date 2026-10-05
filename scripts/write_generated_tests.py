#!/usr/bin/env python3
"""Write the generated test files to disk so users can paste them."""
import os, sys
sys.path.insert(0, '/home/z/my-project/axomo/scripts')
from axomo_wtf_demo import NASTY_PR, SIMULATED_REPO, extract_diff_features, scan_security, scan_edge_cases, generate_tests

features, removed_lines, added_lines = extract_diff_features(NASTY_PR)
sec_findings = scan_security(NASTY_PR)
edge_findings = scan_edge_cases(NASTY_PR, removed_lines)
generated = generate_tests(removed_lines, edge_findings, sec_findings)

out_dir = '/home/z/my-project/axomo/download/generated_tests'
os.makedirs(out_dir, exist_ok=True)
for g in generated:
    p = os.path.join(out_dir, os.path.basename(g['path']))
    with open(p, 'w') as f:
        f.write(g['code'])
    print(f"Wrote {p}  ({len(g['code'])} bytes)")
print(f"\nTotal: {len(generated)} test files written to {out_dir}")
