#!/usr/bin/env python3
"""
Evaluate the AXOTEST unit test generator model.
Generates tests from source code and measures pass rate.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import load_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--workspace", default=".", help="Workspace to write+run tests")
    args = parser.parse_args()

    from transformers import pipeline

    data = load_jsonl(args.data)
    print(f"Loaded {len(data)} test samples")

    print(f"Loading model: {args.model}...")
    generator = pipeline("text2text-generation", model=args.model, device=-1)

    generated = 0
    passed = 0
    times = []

    for i, item in enumerate(data[:50]):
        source = item["input"][:1500]
        start = time.time()
        result = generator(f"generate test: {source}", max_length=512)
        elapsed = time.time() - start
        times.append(elapsed)

        test_code = result[0]["generated_text"] if result else ""
        if len(test_code) < 20:
            continue

        generated += 1
        test_file = os.path.join(args.workspace, f"test/eval/eval-{i}.test.ts")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w') as f:
            f.write(test_code)

        # Try running the test
        import subprocess
        r = subprocess.run(["npx", "vitest", "run", test_file],
                          capture_output=True, text=True, timeout=30,
                          cwd=args.workspace)
        if r.returncode == 0:
            passed += 1

    print(f"\n── AXOTEST Unit Generator Evaluation ──────────")
    print(f"  Generated:  {generated}")
    print(f"  Passed:     {passed}")
    print(f"  Pass rate:  {passed/max(generated,1)*100:.1f}%")
    print(f"  Avg inference: {sum(times)/max(len(times),1)*1000:.0f}ms")
    print(f"  P95 inference: {sorted(times)[int(len(times)*0.95)]*1000:.0f}ms" if len(times) > 1 else "")
    print(f"──────────────────────────────────────────────\n")

    target = 0.70
    rate = passed / max(generated, 1)
    if rate >= target:
        print(f"  ✓ PASS — pass rate {rate*100:.1f}% ≥ {target*100:.0f}% target")
    else:
        print(f"  ✗ FAIL — pass rate {rate*100:.1f}% < {target*100:.0f}% target")

if __name__ == "__main__":
    main()
