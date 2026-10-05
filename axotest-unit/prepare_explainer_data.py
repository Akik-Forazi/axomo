#!/usr/bin/env python3
"""
Prepare training data for the AXOTEST explainer — generates natural
language test result summaries.

Instead of:
  "3/5 tests passed, 2 failed"

The model generates:
  "I ran 5 test agents on the changed code. The unit test agent generated
   2 tests — both passed. The security agent flagged a potential SQL
   injection in db.ts. The edge case agent found 3 boundary conditions
   that need handling. Overall: 3/5 passed, 2 need attention."

Usage:
    python prepare_explainer_data.py --output ./data/
"""
import argparse, os, sys, random, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import save_jsonl, train_test_split

random.seed(42)

SUMMARIES = {
    "all_pass": [
        "All {total} tests passed across {agents} agents. The unit test agent generated {unit_gen} tests for changed code — all green. Security scan found no vulnerabilities. Edge case coverage looks solid. No issues to report.",
        "Clean run: {total}/{total} tests passed. Unit tests for `{symbol}` passed on first try. No security issues detected. The code changes are safe to merge.",
        "Everything passed — {total} tests across {agents} agents. The changes to `{symbol}` are well-tested. No regressions, no security concerns. Good to go.",
    ],
    "some_fail": [
        "{passed}/{total} tests passed. The unit test agent generated a test for `{symbol}` but it failed — the function returns void instead of Promise. Security scan found {sec_issues} potential issue(s) in the changed code. The edge case agent passed. Recommend fixing the unit test before merging.",
        "Mixed results: {passed}/{total} passed. Unit test for `{symbol}` failed because the mock wasn't set up correctly. Security agent found {sec_issues} SQL injection risk in the new query builder. Edge cases all passed. Fix the test + the SQL issue before merge.",
        "{passed}/{total} tests passed. The security agent flagged {sec_issues} concern(s) — possible XSS in the template renderer. Unit tests passed. Edge case agent found 2 unhandled null inputs. Address the security issue first.",
    ],
    "security_fail": [
        "Security scan failed: {sec_issues} vulnerability(ies) found. The security agent detected what looks like an eval() call with user input in `{file}`. This is a critical injection risk. Unit tests passed ({unit_gen} generated). Fix the security issue before anything else.",
        "BLOCKED: {sec_issues} security issue(s) detected. Found hardcoded API key in `{file}`. This should never be in source code. All other tests passed but this must be fixed first. Do not merge.",
    ],
    "edge_fail": [
        "{passed}/{total} tests passed. The edge case agent found that `{symbol}` doesn't handle null input — it throws instead of returning a default. Unit and security tests passed. Add a null check before merging.",
        "Edge case failure: `{symbol}` crashes on empty array input. The function assumes the array is never empty. Unit tests pass for normal cases. Security clean. Add an empty-array guard.",
    ],
}

def generate_summary(scenario: str, context: dict) -> str:
    templates = SUMMARIES.get(scenario, ["Unknown test result."])
    template = random.choice(templates)
    symbols = ["authMiddleware", "validateToken", "createUser", "fetchData",
               "parseConfig", "renderComponent", "handleError"]
    files = ["src/api.ts", "src/db.ts", "src/auth.ts", "src/utils.ts", "src/server.ts"]
    return template.format(
        total=context.get("total", 5),
        passed=context.get("passed", 3),
        agents=context.get("agents", 5),
        unit_gen=context.get("unit_gen", 2),
        sec_issues=context.get("sec_issues", 1),
        symbol=random.choice(symbols),
        file=random.choice(files),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    data = []
    for _ in range(args.n):
        scenario = random.choices(
            ["all_pass", "some_fail", "security_fail", "edge_fail"],
            weights=[40, 30, 15, 15])[0]

        if scenario == "all_pass":
            ctx = {"total": random.randint(3, 10), "passed": 0, "agents": 5,
                   "unit_gen": random.randint(1, 5), "sec_issues": 0}
            ctx["passed"] = ctx["total"]
        elif scenario == "some_fail":
            ctx = {"total": random.randint(4, 10), "passed": 0, "agents": 5,
                   "unit_gen": random.randint(1, 4), "sec_issues": random.randint(1, 3)}
            ctx["passed"] = ctx["total"] - random.randint(1, 3)
        elif scenario == "security_fail":
            ctx = {"total": random.randint(3, 8), "passed": 0, "agents": 5,
                   "unit_gen": random.randint(1, 3), "sec_issues": random.randint(1, 2)}
            ctx["passed"] = ctx["total"] - 1
        else:
            ctx = {"total": random.randint(3, 8), "passed": 0, "agents": 5,
                   "unit_gen": random.randint(1, 3), "sec_issues": 0}
            ctx["passed"] = ctx["total"] - 1

        summary = generate_summary(scenario, ctx)
        prompt = f"Test results: {json.dumps(ctx)}\nScenario: {scenario}\nSummary:"
        data.append({"prompt": prompt, "response": f" {summary}"})

    train, test = train_test_split(data)
    save_jsonl(train, os.path.join(args.output, "explainer_train.jsonl"))
    save_jsonl(test, os.path.join(args.output, "explainer_test.jsonl"))

    print(f"Generated {len(train)} train + {len(test)} test samples")
    print(f"\nSample:\n  Prompt: {data[0]['prompt'][:80]}...")
    print(f"  Response: {data[0]['response'][:150]}...")
    print(f"\nThe model learns: test results → natural language summary")
    print(f"No more '3/5 passed' — the model explains WHAT failed and WHY.")

if __name__ == "__main__":
    main()
