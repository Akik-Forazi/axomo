"""
AXOVB CLI — unified command-line interface for the axovb package.

Subcommands:
    predict   — predict bump type from a git diff (using the trained ONNX MLP)
    apply     — apply a bump (major/minor/patch/--to) project-wide + optional commit + tag
    auto      — predict + apply in one shot
    eval      — evaluate the trained model on the held-out test set
    info      — show model artifacts + their sizes
    list      — list every version file detected in the project root

Usage:
    python -m axovb predict
    python -m axovb predict --diff path/to/pr.patch
    python -m axovb apply --major --commit --tag
    python -m axovb apply --to 3.0.0 --commit --tag --push
    python -m axovb auto --commit --tag
    python -m axovb eval
    python -m axovb info
    python -m axovb list
"""
import argparse, os, sys, json, subprocess, time
from pathlib import Path
from typing import Optional, Dict, List

# Make axovb package importable when invoked via `python -m axovb`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axovb.predictor import BumpPredictor
from axovb.explainer import Explainer
from axovb.bumper import find_version_files, bump_version, apply_plan, git_commit_and_tag
from axovb.analyzer import (
    extract_diff_features, scan_security, scan_edge_cases, trace_blast_radius,
    rule_based_narrative,
)


# ANSI colors
RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
CYAN = "\033[36m"; BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"


def bar(sep="═", n=72):
    return sep * n


def get_diff(root: Path, from_diff: Optional[str] = None) -> str:
    """Get the diff either from a .patch file or `git diff`."""
    if from_diff:
        if not os.path.exists(from_diff):
            print(f"  {RED}diff file not found: {from_diff}{RESET}")
            sys.exit(1)
        with open(from_diff) as f:
            return f.read()
    # Try staged first, then unstaged
    r = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True, text=True)
    diff = r.stdout if r.returncode == 0 else ""
    if not diff.strip():
        r = subprocess.run(["git", "diff", "--cached"], cwd=root, capture_output=True, text=True)
        diff = r.stdout
    return diff


# ─── PREDICT ─────────────────────────────────────────────────────────────

def cmd_predict(args):
    root = Path(args.root).resolve()
    diff = get_diff(root, args.from_diff)
    if not diff.strip():
        print(f"  {YELLOW}no git diff found — run with --diff <path> or stage some changes{RESET}")
        sys.exit(1)

    print(f"  {DIM}running PEAK MLP on diff ({len(diff)} bytes)...{RESET}")
    predictor = BumpPredictor(args.model_dir)
    label, conf, probs, features = predictor.predict_diff(diff)

    removed_exports = []
    if features.get("has_removed_exports"):
        # Re-extract to get the symbol names (predict_diff only returns features)
        _, removed_exports, _ = extract_diff_features(diff)

    print(f"\n  {BOLD}══ AXOVB PREDICTION ══{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")
    print(f"  bump kind:    {GREEN}{BOLD}{label.upper()}{RESET}    confidence: {conf*100:.1f}%")
    print(f"  inference:    onnxruntime (PEAK MLP, 2.8M params)")
    if removed_exports:
        print(f"  removed:      {', '.join(f'`{s}`' for s in removed_exports)}")
    print(f"  features:     {json.dumps({k: v for k, v in features.items() if k in ['num_changed_symbols','num_callers_affected','has_removed_exports','has_new_exports','has_signature_changes','total_files']})}")
    print(f"\n  {DIM}Per-class probabilities:{RESET}")
    for i, p in enumerate(probs):
        lbl = predictor.id2label[i]
        marker = " ←" if lbl == label else ""
        print(f"    {lbl:<8} {p*100:6.2f}%{marker}")

    if args.explain:
        print(f"\n  {DIM}Generating narrative...{RESET}")
        expl = Explainer()
        text, ms, used_ml = expl.generate(
            features, removed_lines=removed_exports,
            bump_label=label,
        )
        print(f"\n  {BOLD}NARRATIVE{RESET} {'(distilgpt2)' if used_ml else '(rule-based fallback)'}:")
        print(f"  {text}")

    print()
    if args.apply:
        _apply_bump(root, label, features, args)
    else:
        print(f"  {DIM}next: run with --apply to bump project-wide{RESET}")


# ─── APPLY ──────────────────────────────────────────────────────────────

def _apply_bump(root: Path, bump_kind: str, features: Dict, args):
    """Apply a bump project-wide. Shared by `apply` and `auto`."""
    print(f"\n  {DIM}scanning {root} for version files...{RESET}")
    files = find_version_files(root)
    if not files:
        print(f"  {RED}no version files detected{RESET}")
        sys.exit(1)
    for f in files:
        f["_root"] = root

    from collections import Counter
    version_counts = Counter(f["current_version"] for f in files)
    current_target, _ = version_counts.most_common(1)[0]

    if args.to:
        new_target = args.to
        m1 = re.match(r'^(\d+)\.(\d+)\.(\d+)', new_target) if False else None  # placeholder
    else:
        new_target = bump_version(current_target, bump_kind)

    plan = []
    inconsistent = []
    for f in files:
        f["new_version"] = bump_version(f["current_version"], bump_kind)
        if f["current_version"] != current_target:
            inconsistent.append(f)
        plan.append(f)

    # Render preview
    print(f"\n  {BOLD}══ AXOVB BUMP — PROJECT-WIDE VERSION EDIT ══{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")
    print(f"  bump source:    {CYAN}{args.source if hasattr(args, 'source') else 'manual --' + bump_kind}{RESET}"
          + (f"  (MLP confidence: {args.confidence*100:.1f}%)" if hasattr(args, 'confidence') and args.confidence else ""))
    print(f"  bump kind:      {BOLD}{bump_kind.upper()}{RESET}")
    print(f"  target version: {GREEN}{new_target}{RESET}  (from {current_target})")
    print(f"  files affected: {BOLD}{len(plan)}{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")

    by_eco = {}
    for p in plan:
        by_eco.setdefault(p["ecosystem"], []).append(p)
    for eco, items in by_eco.items():
        print(f"\n  {CYAN}{BOLD}{eco.upper()}{RESET}")
        for p in items:
            old = p["current_version"]; new = p["new_version"]
            rel = p["path"].relative_to(p["_root"]) if p.get("_root") else p["path"]
            print(f"    {rel}")
            if p["ecosystem"] == "meta":
                print(f"      {GREEN}+ ## [{new}] - {time.strftime('%Y-%m-%d')}{RESET}   {DIM}({p['notes']}){RESET}")
            else:
                print(f"      {RED}- {old}{RESET}  {GREEN}+ {new}{RESET}   {DIM}({p['notes']}){RESET}")
    print(f"\n  {DIM}{'─' * 60}{RESET}")

    if inconsistent:
        print(f"  {YELLOW}⚠ {len(inconsistent)} file(s) had a different current version{RESET}")
        print(f"  {DIM}each file is bumped from its OWN current version (handles monorepo drift){RESET}")

    if args.preview:
        print(f"\n  {DIM}(preview only — no files written){RESET}")
        return

    if not args.yes:
        r = input(f"\n  {YELLOW}apply bump to {len(plan)} file(s)? [y/N]{RESET} ").strip().lower()
        if r != "y":
            print(f"  {DIM}aborted{RESET}")
            return

    ok, backups = apply_plan(plan)
    if not ok:
        sys.exit(2)
    for _, _, bd in backups:
        shutil.rmtree(bd, ignore_errors=True)
    print(f"\n  {GREEN}✓ applied {bump_kind} bump to {len(plan)} file(s) atomically{RESET}")
    print(f"  {GREEN}✓ target version: {new_target}{RESET}")

    if args.commit or args.tag:
        git_commit_and_tag(root, new_target, bump_kind, features or {},
                           tag=args.tag, push=args.push)


def cmd_apply(args):
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"  {RED}root does not exist: {root}{RESET}")
        sys.exit(1)
    bump_kind = args.bump
    if bump_kind is None and not args.to:
        print(f"  {RED}must specify --major | --minor | --patch | --to X.Y.Z{RESET}")
        sys.exit(1)
    if args.to:
        # Explicit target version — derive kind for commit message
        import re
        if not re.match(r'^\d+\.\d+\.\d+', args.to):
            print(f"  {RED}--to must be semver, got: {args.to}{RESET}")
            sys.exit(1)
        # Compute kind by comparing to current
        files = find_version_files(root)
        if not files:
            print(f"  {RED}no version files detected{RESET}")
            sys.exit(1)
        from collections import Counter
        cur, _ = Counter(f["current_version"] for f in files).most_common(1)[0]
        m1 = re.match(r'^(\d+)\.(\d+)\.(\d+)', args.to)
        m2 = re.match(r'^(\d+)\.(\d+)\.(\d+)', cur)
        if m1 and m2:
            a, b, c = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
            x, y, z = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            if a > x: bump_kind = "major"
            elif b > y: bump_kind = "minor"
            elif c > z: bump_kind = "patch"
            else: bump_kind = "none"
        else:
            bump_kind = "patch"
        args.source = f"explicit --to {args.to}"
    else:
        args.source = f"manual --{bump_kind}"
    _apply_bump(root, bump_kind, {}, args)


def cmd_auto(args):
    """Predict bump from staged diff + apply project-wide."""
    root = Path(args.root).resolve()
    diff = get_diff(root, args.from_diff)
    if not diff.strip():
        print(f"  {YELLOW}no git diff found — defaulting to patch bump{RESET}")
        bump_kind = "patch"
        features = {}
        conf = None
    else:
        print(f"  {DIM}running PEAK MLP on diff ({len(diff)} bytes)...{RESET}")
        predictor = BumpPredictor(args.model_dir)
        bump_kind, conf, _, features = predictor.predict_diff(diff)
        if bump_kind is None:
            print(f"  {RED}MLP model not found — falling back to patch{RESET}")
            bump_kind = "patch"
    args.source = "PEAK MLP (ONNX)" + (f"  (MLP confidence: {conf*100:.1f}%)" if conf else "")
    args.confidence = conf
    _apply_bump(root, bump_kind, features, args)


# ─── EVAL ───────────────────────────────────────────────────────────────

def cmd_eval(args):
    """Evaluate the trained ONNX model on the held-out test set."""
    from axovb.eval import evaluate_on_test_set
    evaluate_on_test_set(
        model_dir=args.model_dir or os.path.join(os.path.dirname(__file__), "models"),
        data_path=args.data or os.path.join(os.path.dirname(__file__), "data", "test.jsonl"),
    )


# ─── INFO ───────────────────────────────────────────────────────────────

def cmd_info(args):
    """Show model artifacts + their sizes."""
    axovb_dir = Path(__file__).parent
    print(f"\n  {BOLD}══ AXOVB INFO ══{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")
    print(f"  version:    0.3.0")
    print(f"  package:    {axovb_dir}")
    print()

    # Bumper MLP
    models_dir = axovb_dir / "models"
    print(f"  {CYAN}{BOLD}BUMPER MLP (PEAK, ONNX){RESET}")
    if models_dir.exists():
        for f in sorted(models_dir.iterdir()):
            if f.is_file():
                size_kb = f.stat().st_size // 1024
                print(f"    {f.name:<25} {size_kb:>8} KB")
        # Try to load + show accuracy
        try:
            pred = BumpPredictor(str(models_dir))
            info = pred.info()
            print(f"    feature count:           {info['feature_count']}")
            print(f"    labels:                  {info['labels']}")
        except Exception as e:
            print(f"    {RED}load failed: {e}{RESET}")
    else:
        print(f"    {YELLOW}not trained — run `python axovb/train.py` first{RESET}")

    # Explainer
    expl_dir = axovb_dir / "explainer"
    print(f"\n  {CYAN}{BOLD}EXPLAINER (distilgpt2, 82M){RESET}")
    if expl_dir.exists() and (expl_dir / "config.json").exists():
        for f in sorted(expl_dir.iterdir()):
            if f.is_file():
                size_kb = f.stat().st_size // 1024
                size_str = f"{size_kb} KB" if size_kb < 1024 else f"{size_kb//1024} MB"
                print(f"    {f.name:<25} {size_str:>8}")
        # Show training metrics
        metrics_path = expl_dir / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            print(f"    params:                  {m.get('params', '?'):,}")
            print(f"    train samples:           {m.get('train_samples', '?')}")
            print(f"    epochs:                  {m.get('epochs', '?')}")
            print(f"    train time:              {m.get('train_time_seconds', 0):.1f}s")
            print(f"    peak train loss:         {m.get('peak_train_loss', '?')}")
            print(f"    final train loss:       {m.get('final_train_loss', '?')}")
            print(f"    final eval perplexity:  {m.get('final_eval_loss', '?')}")
    else:
        print(f"    {YELLOW}not trained — run `python scripts/train_explainer_lomem.py` first{RESET}")
    print()


# ─── LIST ───────────────────────────────────────────────────────────────

def cmd_list(args):
    """List every version file detected in the project root."""
    root = Path(args.root).resolve()
    files = find_version_files(root)
    if not files:
        print(f"  {YELLOW}no version files detected in {root}{RESET}")
        return
    print(f"\n  {BOLD}══ AXOVB VERSION FILES ══{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")
    print(f"  {len(files)} file(s) detected in {root}\n")
    by_eco = {}
    for f in files:
        by_eco.setdefault(f["ecosystem"], []).append(f)
    for eco, items in sorted(by_eco.items()):
        print(f"  {CYAN}{BOLD}{eco.upper()}{RESET}")
        for f in items:
            rel = f["path"].relative_to(root)
            print(f"    {str(rel):<40} {f['current_version']:<10} {DIM}({f['notes']}){RESET}")
        print()


# ─── CLI DISPATCHER ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="axovb",
        description="AXOVB — AI-powered semantic version bumper (PEAK MLP + distilgpt2 explainer)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    p = sub.add_parser("predict", help="Predict bump type from a git diff")
    p.add_argument("--root", default=".", help="Project root (default: cwd)")
    p.add_argument("--model-dir", default=None, help="Path to ONNX model dir")
    p.add_argument("--from-diff", default=None, help="Path to a .patch file (else: git diff HEAD)")
    p.add_argument("--apply", action="store_true", help="Also apply the predicted bump project-wide")
    p.add_argument("--explain", action="store_true", help="Also generate a narrative explanation")
    p.add_argument("--commit", action="store_true", help="(with --apply) git commit the changes")
    p.add_argument("--tag", action="store_true", help="(with --apply) git tag vX.Y.Z")
    p.add_argument("--push", action="store_true", help="(with --apply) git push origin HEAD + tag")
    p.add_argument("--yes", action="store_true", help="(with --apply) skip confirmation prompt")
    p.add_argument("--preview", action="store_true", help="(with --apply) show changes without writing")
    p.set_defaults(func=cmd_predict)

    # apply
    p = sub.add_parser("apply", help="Apply a bump project-wide")
    p.add_argument("--root", default=".", help="Project root (default: cwd)")
    p.add_argument("--major", action="store_const", const="major", dest="bump")
    p.add_argument("--minor", action="store_const", const="minor", dest="bump")
    p.add_argument("--patch", action="store_const", const="patch", dest="bump")
    p.add_argument("--none",  action="store_const", const="none",  dest="bump")
    p.add_argument("--to", default=None, help="Set explicit target version (overrides --major etc.)")
    p.add_argument("--commit", action="store_true", help="git commit with bump message")
    p.add_argument("--tag", action="store_true", help="git tag vX.Y.Z")
    p.add_argument("--push", action="store_true", help="git push origin HEAD + tag")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--preview", action="store_true", help="Show changes without writing")
    p.set_defaults(func=cmd_apply)

    # auto
    p = sub.add_parser("auto", help="Predict bump from staged diff + apply project-wide")
    p.add_argument("--root", default=".", help="Project root (default: cwd)")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--from-diff", default=None, help="Path to a .patch file (else: git diff HEAD)")
    p.add_argument("--commit", action="store_true", help="git commit with bump message")
    p.add_argument("--tag", action="store_true", help="git tag vX.Y.Z")
    p.add_argument("--push", action="store_true", help="git push origin HEAD + tag")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("--preview", action="store_true", help="Show changes without writing")
    p.set_defaults(func=cmd_auto)

    # eval
    p = sub.add_parser("eval", help="Evaluate the trained model on the held-out test set")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--data", default=None, help="Path to test.jsonl")
    p.set_defaults(func=cmd_eval)

    # info
    p = sub.add_parser("info", help="Show model artifacts + their sizes")
    p.set_defaults(func=cmd_info)

    # list
    p = sub.add_parser("list", help="List every version file detected in the project root")
    p.add_argument("--root", default=".", help="Project root (default: cwd)")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # `import re, shutil` for cmd_apply's _apply_bump
    global re, shutil
    import re, shutil
    main()
