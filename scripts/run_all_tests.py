#!/usr/bin/env python3
"""
AXOMO unified test runner — exercises every trained model surface and writes
a structured results JSON to scripts/test_results.json.

Tests run:
  1. PEAK MLP bumper (ONNX) — load + inference on test.jsonl
  2. distilgpt2 explainer (PyTorch) — load + generate on 4 prompts
  3. axomo_bump — end-to-end on a fresh demo monorepo (preview-only, no writes)
  4. axomo_wtf_demo — full pipeline on the crafted nasty PR
"""
import json, os, sys, time, subprocess, tempfile, shutil
from pathlib import Path

ROOT = Path('/home/z/my-project/axomo')
PYTHON = '/home/z/.venv/bin/python3'

results = {
    "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "python": PYTHON,
    "tests": []
}

def record(name, status, duration_ms, summary, error=None, details=None):
    results["tests"].append({
        "name": name, "status": status, "duration_ms": round(duration_ms, 2),
        "summary": summary, "error": error, "details": details or {},
    })

def run_test(name, fn):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        summary, details = fn()
        elapsed = (time.time() - t0) * 1000
        print(f"  ✓ PASS  ({elapsed:.0f}ms) — {summary}")
        record(name, "pass", elapsed, summary, None, details)
        return True
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        err = f"{type(e).__name__}: {e}"
        print(f"  ✗ FAIL  ({elapsed:.0f}ms) — {err}")
        record(name, "fail", elapsed, "", err)
        return False


def test_bumper_onnx():
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT / 'shared'))
    from verify_onnx import main as verify_main
    import io, contextlib, re
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        verify_main()
    out = buf.getvalue()
    m = re.search(r'Accuracy:\s+([\d.]+)%', out)
    acc = float(m.group(1)) if m else 0
    m2 = re.search(r'Total time:\s+([\d.]+)ms', out)
    infer_ms = float(m2.group(1)) if m2 else 0
    summary = f"ONNX inference on 75 test samples: {acc:.1f}% accuracy in {infer_ms:.1f}ms total"
    return summary, {"accuracy_pct": acc, "total_inference_ms": infer_ms, "samples": 75}


def test_explainer():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    model_dir = str(ROOT / 'axovb' / 'explainer')
    assert os.path.exists(os.path.join(model_dir, 'config.json')), "explainer model not found"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    prompts = [
        'Features: {"num_changed_symbols": 5, "num_callers_affected": 15, "has_removed_exports": 1, "has_new_exports": 0}\nBump: major\nExplanation:',
        'Features: {"num_changed_symbols": 3, "num_callers_affected": 0, "has_removed_exports": 0, "has_new_exports": 1}\nBump: minor\nExplanation:',
        'Features: {"num_changed_symbols": 2, "num_callers_affected": 5, "has_removed_exports": 0, "has_new_exports": 0}\nBump: patch\nExplanation:',
        'Features: {"num_changed_symbols": 0, "num_callers_affected": 0, "has_removed_exports": 0, "has_new_exports": 0, "num_doc_files": 3}\nBump: none\nExplanation:',
    ]
    generations = []
    t0 = time.time()
    for p in prompts:
        inputs = tok(p, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50, temperature=0.7, do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        resp = text[len(p):].strip()[:200]
        generations.append({"prompt": p[:80], "response": resp})
    elapsed = (time.time() - t0) * 1000
    summary = f"Generated {len(generations)} explanations in {elapsed:.0f}ms (avg {elapsed/len(generations):.0f}ms each)"
    return summary, {"generations": generations, "total_ms": elapsed}


def test_axomo_bump():
    demo = ROOT / 'demo-monorepo'
    assert demo.exists(), "demo-monorepo not found"
    r = subprocess.run(
        [PYTHON, str(ROOT / 'scripts' / 'axomo_bump.py'),
         '--root', str(demo), '--major', '--preview', '--yes'],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"axomo_bump exit {r.returncode}: {r.stderr[:500]}")
    import re
    clean = re.sub(r'\033\[\d+m', '', r.stdout)
    m = re.search(r'files affected:\s+(\d+)', clean)
    n_files = int(m.group(1)) if m else 0
    m2 = re.search(r'target version:\s+(\S+)', clean)
    target = m2.group(1) if m2 else "?"
    summary = f"axomo_bump preview: detected {n_files} files to bump to {target}"
    return summary, {"files_affected": n_files, "target_version": target, "preview_lines": len(clean.splitlines())}


def test_wtf_demo():
    r = subprocess.run(
        [PYTHON, str(ROOT / 'scripts' / 'axomo_wtf_demo.py')],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"wtf_demo exit {r.returncode}: {r.stderr[:500]}")
    out = r.stdout
    import re
    clean = re.sub(r'\033\[\d+m', '', out)
    sections = re.findall(r'▶\s+(.+?)(?:\s*\()', clean)
    sec_findings = re.search(r'SECURITY SMOKES\s+\((\d+) found\)', clean)
    edge_findings = re.search(r'FORGOTTEN EDGE CASES\s+\((\d+) found\)', clean)
    blast = re.search(r'BLAST RADIUS\s+\((\d+) affected', clean)
    pred_tests = re.search(r'PREDICTED FAILING TESTS\s+\((\d+)\)', clean)
    gen_tests = re.search(r'GENERATED TESTS\s+\((\d+) new', clean)
    bump_match = re.search(r'prediction:\s+(\w+)\s+confidence:\s+([\d.]+)%', clean)
    total = re.search(r'Total:\s+([\d.]+)s', clean)
    summary = (
        f"bump={bump_match.group(1) if bump_match else '?'} "
        f"@ {bump_match.group(2) if bump_match else '?'}%, "
        f"{sec_findings.group(1) if sec_findings else '?'} sec, "
        f"{edge_findings.group(1) if edge_findings else '?'} edge, "
        f"{blast.group(1) if blast else '?'} callers, "
        f"{pred_tests.group(1) if pred_tests else '?'} failing tests, "
        f"{gen_tests.group(1) if gen_tests else '?'} gen tests, "
        f"total={total.group(1) if total else '?'}s"
    )
    return summary, {
        "sections": sections,
        "bump_predicted": bump_match.group(1) if bump_match else None,
        "confidence_pct": float(bump_match.group(2)) if bump_match else None,
        "security_findings": int(sec_findings.group(1)) if sec_findings else 0,
        "edge_findings": int(edge_findings.group(1)) if edge_findings else 0,
        "blast_radius": int(blast.group(1)) if blast else 0,
        "predicted_failing_tests": int(pred_tests.group(1)) if pred_tests else 0,
        "generated_tests": int(gen_tests.group(1)) if gen_tests else 0,
        "total_seconds": float(total.group(1)) if total else None,
    }


def test_axovb_cli():
    """Test the new `python -m axovb` CLI surface — info, list, eval."""
    import re
    # Test 1: axovb info
    r = subprocess.run(
        [PYTHON, "-m", "axovb", "info"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"axovb info exit {r.returncode}: {r.stderr[:500]}")
    info_clean = re.sub(r'\033\[\d+m', '', r.stdout)
    has_bumper = "BUMPER MLP" in info_clean
    has_explainer = "EXPLAINER" in info_clean

    # Test 2: axovb list on demo-monorepo
    r2 = subprocess.run(
        [PYTHON, "-m", "axovb", "list", "--root", str(ROOT / "demo-monorepo")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30,
    )
    if r2.returncode != 0:
        raise RuntimeError(f"axovb list exit {r2.returncode}: {r2.stderr[:500]}")
    list_clean = re.sub(r'\033\[\d+m', '', r2.stdout)
    m = re.search(r'(\d+)\s+file\(s\)\s+detected', list_clean)
    n_files = int(m.group(1)) if m else 0
    # Count ecosystem headers
    ecosystems = set(re.findall(r'^\s*([A-Z]+)$', list_clean, re.M))

    # Test 3: axovb eval
    r3 = subprocess.run(
        [PYTHON, "-m", "axovb", "eval"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    if r3.returncode != 0:
        raise RuntimeError(f"axovb eval exit {r3.returncode}: {r3.stderr[:500]}")
    eval_clean = re.sub(r'\033\[\d+m', '', r3.stdout)
    m = re.search(r'Accuracy:\s+([\d.]+)%', eval_clean)
    acc = float(m.group(1)) if m else 0
    summary = (
        f"axovb CLI: info={'OK' if has_bumper and has_explainer else 'FAIL'}, "
        f"list={n_files} files in {len(ecosystems)} ecosystems, "
        f"eval={acc:.1f}% accuracy"
    )
    return summary, {
        "info_has_bumper": has_bumper,
        "info_has_explainer": has_explainer,
        "list_files_detected": n_files,
        "list_ecosystems": sorted(ecosystems),
        "eval_accuracy_pct": acc,
    }


if __name__ == "__main__":
    print(f"\n  AXOMO UNIFIED TEST RUNNER")
    print(f"  python: {PYTHON}")
    print(f"  root:   {ROOT}")

    passes = []
    passes.append(run_test("PEAK MLP bumper (ONNX inference)", test_bumper_onnx))
    passes.append(run_test("distilgpt2 explainer (text generation)", test_explainer))
    passes.append(run_test("axomo_bump end-to-end (preview)", test_axomo_bump))
    passes.append(run_test("axomo_wtf_demo (full pipeline)", test_wtf_demo))
    passes.append(run_test("axovb CLI (info/list/eval)", test_axovb_cli))

    n_pass = sum(passes)
    n_total = len(passes)
    results["summary"] = f"{n_pass}/{n_total} tests passed"

    out_path = ROOT / 'scripts' / 'test_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to {out_path}")
    print(f"\n  {n_pass}/{n_total} tests passed\n")

    sys.exit(0 if n_pass == n_total else 1)
