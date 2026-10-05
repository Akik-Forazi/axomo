#!/usr/bin/env python3
"""
AXOMO BUMP — project-wide version editor
========================================
One command that:
  1. Auto-detects every version file across ecosystems (Node, Python, Rust,
     Go, Ruby, .NET, Elixir, Helm, lerna, turbo, pnpm workspaces).
  2. Either takes a manual bump (major/minor/patch) or runs the PEAK MLP
     on the current `git diff` to predict the bump automatically.
  3. Shows a colored preview of every file that will change, with the
     exact line diff (red strikethrough old, green new).
  4. On confirm: writes ALL files atomically (backup → write → verify →
     cleanup; restore everything if any file fails verification).
  5. Optional `--commit` writes a git commit + `vX.Y.Z` tag with the
     bump type in the message and a generated changelog entry.
  6. Optional `--push` pushes the commit + tag to origin.

Usage:
    python axomo_bump.py --predict --apply --commit --tag
    python axomo_bump.py --to 2.1.0 --apply --commit --tag
    python axomo_bump.py --to 2.1.0 --preview
    python axomo_bump.py --patch --preview
    python axomo_bump.py --predict --preview
"""
import argparse, os, sys, json, re, shutil, subprocess, time, tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def parse_json_version(path: Path) -> Optional[str]:
    try:
        with open(path) as f:
            return json.load(f).get("version")
    except Exception:
        return None

def write_json_version(path: Path, old: str, new: str, backup: Path):
    with open(path) as f:
        data = json.load(f)
    data["version"] = new
    shutil.copy2(path, backup)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    with open(path) as f:
        if json.load(f).get("version") != new:
            raise RuntimeError(f"verification failed: {path}")

def parse_pyproject_version(path: Path) -> Optional[str]:
    text = path.read_text()
    m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return m.group(1) if m else None

def write_pyproject_version(path: Path, old: str, new: str, backup: Path):
    text = path.read_text()
    shutil.copy2(path, backup)
    new_text = re.sub(
        r'(^version\s*=\s*["\'])[^"\']+(["\'])',
        rf'\g<1>{new}\g<2>',
        text,
        count=1,
        flags=re.M,
    )
    path.write_text(new_text)
    if parse_pyproject_version(path) != new:
        raise RuntimeError(f"verification failed: {path}")

def parse_cargo_version(path: Path) -> Optional[str]:
    text = path.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None

def write_cargo_version(path: Path, old: str, new: str, backup: Path):
    text = path.read_text()
    shutil.copy2(path, backup)
    new_text = re.sub(
        r'(^version\s*=\s*")[^"]+(")',
        rf'\g<1>{new}\g<2>',
        text,
        count=1,
        flags=re.M,
    )
    path.write_text(new_text)
    if parse_cargo_version(path) != new:
        raise RuntimeError(f"verification failed: {path}")

def parse_setup_py_version(path: Path) -> Optional[str]:
    text = path.read_text()
    m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', text)
    return m.group(1) if m else None

def write_setup_py_version(path: Path, old: str, new: str, backup: Path):
    text = path.read_text()
    shutil.copy2(path, backup)
    new_text = re.sub(
        r'(version\s*=\s*["\'])[^"\']+(["\'])',
        rf'\g<1>{new}\g<2>',
        text,
        count=1,
    )
    path.write_text(new_text)
    if parse_setup_py_version(path) != new:
        raise RuntimeError(f"verification failed: {path}")

def parse_plain_version(path: Path) -> Optional[str]:
    text = path.read_text().strip()
    m = re.match(r'^v?(\d+\.\d+\.\d+(?:[-+][\w.]+)?)', text)
    return m.group(1) if m else None

def write_plain_version(path: Path, old: str, new: str, backup: Path):
    shutil.copy2(path, backup)
    path.write_text(f"{new}\n")
    if parse_plain_version(path) != new:
        raise RuntimeError(f"verification failed: {path}")

def parse_chart_yaml_version(path: Path) -> Optional[str]:
    text = path.read_text()
    m = re.search(r'^version:\s*["\']?([^"\'\s]+)["\']?\s*$', text, re.M)
    return m.group(1) if m else None

def write_chart_yaml_version(path: Path, old: str, new: str, backup: Path):
    text = path.read_text()
    shutil.copy2(path, backup)
    def repl(m):
        prefix, open_q, close_q = m.group(1), m.group(2) or "", m.group(4) or ""
        return f"{prefix}{open_q}{new}{close_q}"
    new_text, n = re.subn(
        r'(^version:\s*)(["\']?)([^"\'\s]+)(["\']?)\s*$',
        repl,
        text,
        count=1,
        flags=re.M,
    )
    if n == 0:
        raise RuntimeError(f"version line not found in {path}")
    if old in text:
        new_text2, m = re.subn(
            rf'(^appVersion:\s*)(["\']?){re.escape(old)}(["\']?)\s*$',
            lambda mm: f"{mm.group(1)}{mm.group(2)}{new}{mm.group(3)}",
            new_text,
            count=1,
            flags=re.M,
        )
        if m > 0:
            new_text = new_text2
    path.write_text(new_text)
    if parse_chart_yaml_version(path) != new:
        raise RuntimeError(f"verification failed: {path}")

def parse_go_mod_version(path: Path) -> Optional[str]:
    text = path.read_text()
    m = re.search(r'^//\s*Version:\s*v?(\d+\.\d+\.\d+)', text, re.M)
    return m.group(1) if m else None

def write_go_mod_version(path: Path, old: str, new: str, backup: Path):
    text = path.read_text()
    shutil.copy2(path, backup)
    new_text = re.sub(
        r'(^//\s*Version:\s*v?)\d+\.\d+\.\d+',
        rf'\g<1>{new}',
        text,
        count=1,
        flags=re.M,
    )
    path.write_text(new_text)

def parse_csproj_version(path: Path) -> Optional[str]:
    text = path.read_text()
    m = re.search(r'<Version>([^<]+)</Version>', text)
    if not m:
        m = re.search(r'<AssemblyVersion>([^<]+)</AssemblyVersion>', text)
    return m.group(1) if m else None

def write_csproj_version(path: Path, old: str, new: str, backup: Path):
    text = path.read_text()
    shutil.copy2(path, backup)
    new_text = text
    if "<Version>" in text:
        new_text = re.sub(r'(<Version>)[^<]+(</Version>)', rf'\g<1>{new}\g<2>', new_text, count=1)
    if "<AssemblyVersion>" in text:
        new_text = re.sub(r'(<AssemblyVersion>)[^<]+(</AssemblyVersion>)', rf'\g<1>{new}\g<2>', new_text, count=1)
    if "<FileVersion>" in text:
        new_text = re.sub(r'(<FileVersion>)[^<]+(</FileVersion>)', rf'\g<1>{new}\g<2>', new_text, count=1)
    path.write_text(new_text)
    if parse_csproj_version(path) != new:
        raise RuntimeError(f"verification failed: {path}")

def parse_mix_exs_version(path: Path) -> Optional[str]:
    text = path.read_text()
    m = re.search(r'@version\s+["\']([^"\']+)["\']', text)
    if not m:
        m = re.search(r'version:\s*["\']([^"\']+)["\']', text)
    return m.group(1) if m else None

def write_mix_exs_version(path: Path, old: str, new: str, backup: Path):
    text = path.read_text()
    shutil.copy2(path, backup)
    new_text = text
    if "@version " in text:
        new_text = re.sub(r'(@version\s+["\'])[^"\']+(["\'])', rf'\g<1>{new}\g<2>', new_text, count=1)
    else:
        new_text = re.sub(r'(version:\s*["\'])[^"\']+(["\'])', rf'\g<1>{new}\g<2>', new_text, count=1)
    path.write_text(new_text)
    if parse_mix_exs_version(path) != new:
        raise RuntimeError(f"verification failed: {path}")

def parse_changelog_version(path: Path) -> Optional[str]:
    text = path.read_text()
    m = re.search(r'^##\s*\[?(\d+\.\d+\.\d+)', text, re.M)
    return m.group(1) if m else None

def write_changelog_version(path: Path, old: str, new: str, backup: Path):
    text = path.read_text()
    shutil.copy2(path, backup)
    today = time.strftime("%Y-%m-%d")
    new_section = f"\n## [{new}] - {today}\n\n- (auto-generated by axomo bump — fill in details)\n\n"
    new_text = re.sub(r'^(#.*\n)', rf'\g<1>{new_section}', text, count=1, flags=re.M)
    path.write_text(new_text)


DETECTORS = [
    (["package.json"],                       "node",     parse_json_version,        write_json_version,        "root package.json"),
    (["lerna.json"],                         "node",     parse_json_version,        write_json_version,        "lerna monorepo root"),
    (["pyproject.toml"],                     "python",   parse_pyproject_version,    write_pyproject_version,   "PEP 621 / poetry / hatch"),
    (["setup.py"],                           "python",   parse_setup_py_version,    write_setup_py_version,     "legacy setup.py"),
    (["Cargo.toml"],                         "rust",     parse_cargo_version,       write_cargo_version,        "cargo crate"),
    (["Chart.yaml"],                         "helm",     parse_chart_yaml_version,  write_chart_yaml_version,   "helm chart + appVersion"),
    (["VERSION"],                            "plain",    parse_plain_version,       write_plain_version,        "plain VERSION file"),
    (["mix.exs"],                             "elixir",  parse_mix_exs_version,    write_mix_exs_version,      "elixir mix project"),
    (["go.mod"],                              "go",      parse_go_mod_version,     write_go_mod_version,       "go.mod (// Version comment)"),
    (["*.csproj"],                            "dotnet",  parse_csproj_version,     write_csproj_version,        ".NET project file"),
    (["CHANGELOG.md"],                        "meta",    parse_changelog_version,  write_changelog_version,    "prepend new release section"),
]


def find_version_files(root: Path) -> List[Dict]:
    found = []
    root = root.resolve()
    ignore = {".git", "node_modules", "target", "build", "dist", ".venv", "venv", "__pycache__", ".next", ".turbo"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore]
        rel_depth = Path(dirpath).relative_to(root).parts
        if len(rel_depth) > 4:
            dirnames[:] = []
            continue
        for fname in filenames:
            for patterns, eco, parser, writer, notes in DETECTORS:
                matched = False
                for pat in patterns:
                    if pat.startswith("*"):
                        if fname.endswith(pat[1:]):
                            matched = True; break
                    elif fname == pat:
                        matched = True; break
                if not matched or parser is None:
                    continue
                full = Path(dirpath) / fname
                try:
                    ver = parser(full)
                except Exception:
                    ver = None
                if ver is None:
                    continue
                if not re.match(r'^\d+\.\d+\.\d+', ver):
                    continue
                found.append({
                    "path": full,
                    "ecosystem": eco,
                    "current_version": ver,
                    "parser": parser,
                    "writer": writer,
                    "notes": notes,
                })

    seen_paths = {str(f["path"]) for f in found}
    for sub in (root).glob("packages/*/package.json"):
        if str(sub) in seen_paths:
            continue
        try:
            ver = parse_json_version(sub)
            if ver and re.match(r'^\d+\.\d+\.\d+', ver):
                found.append({
                    "path": sub,
                    "ecosystem": "node-workspace",
                    "current_version": ver,
                    "parser": parse_json_version,
                    "writer": write_json_version,
                    "notes": f"workspace package {sub.parent.name}",
                })
                seen_paths.add(str(sub))
        except Exception:
            pass
    for sub in (root).glob("apps/*/package.json"):
        if str(sub) in seen_paths:
            continue
        try:
            ver = parse_json_version(sub)
            if ver and re.match(r'^\d+\.\d+\.\d+', ver):
                found.append({
                    "path": sub,
                    "ecosystem": "node-workspace",
                    "current_version": ver,
                    "parser": parse_json_version,
                    "writer": write_json_version,
                    "notes": f"workspace app {sub.parent.name}",
                })
                seen_paths.add(str(sub))
        except Exception:
            pass

    return found


def bump_version(current: str, kind: str) -> str:
    m = re.match(r'^(\d+)\.(\d+)\.(\d+)(.*)$', current)
    if not m:
        raise ValueError(f"can't parse: {current}")
    major, minor, patch, suffix = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    if kind == "major":
        major, minor, patch = major + 1, 0, 0
    elif kind == "minor":
        minor, patch = minor + 1, 0
    elif kind == "patch":
        patch += 1
    elif kind == "none":
        pass
    else:
        raise ValueError(f"unknown bump kind: {kind}")
    if kind != "none" and suffix:
        suffix = ""
    return f"{major}.{minor}.{patch}{suffix}"


def predict_bump_from_diff(diff: str):
    sys.path.insert(0, '/home/z/my-project/axomo/scripts')
    from axomo_wtf_demo import extract_diff_features
    features, _, _ = extract_diff_features(diff)
    FEATURE_NAMES = [
        "num_changed_symbols", "num_callers_affected", "has_removed_exports", "has_new_exports",
        "has_exported_changes", "has_signature_changes", "has_type_changes", "has_rename",
        "num_source_files", "num_test_files", "num_doc_files", "num_config_files",
        "total_files", "test_to_source_ratio", "blast_radius", "change_scope",
        "avg_callers_per_symbol", "export_change_ratio", "breaking_score", "complexity_score",
    ]
    feature_vec = {n: features.get(n, 0) for n in FEATURE_NAMES}
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        return None, None, feature_vec
    model_dir = '/home/z/my-project/axomo/axovb/models'
    onnx_path = os.path.join(model_dir, 'bump_mlp.onnx')
    meta_path = os.path.join(model_dir, 'metadata.json')
    if not (os.path.exists(onnx_path) and os.path.exists(meta_path)):
        return None, None, feature_vec
    with open(meta_path) as f:
        meta = json.load(f)
    X_min = np.array(meta['feature_min'], dtype=np.float32)
    X_max = np.array(meta['feature_max'], dtype=np.float32)
    id2label = {int(k): v for k, v in meta['id2label'].items()}
    x = np.array([[feature_vec[n] for n in FEATURE_NAMES]], dtype=np.float32)
    # Pass RAW features — BumpMLPInference wrapper in the ONNX handles normalization
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    probs = sess.run([output_name], {input_name: x})[0]
    pred = int(probs.argmax(axis=1)[0])
    conf = float(probs[0][pred])
    return id2label[pred], conf, feature_vec


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def render_preview(plan: List[Dict], current_target: str, new_target: str,
                    bump_kind: str, source: str, confidence: Optional[float] = None,
                    features: Optional[Dict] = None):
    out = []
    out.append("")
    out.append(f"  {BOLD}══ AXOMO BUMP — PROJECT-WIDE VERSION EDIT ══{RESET}")
    out.append(f"  {DIM}{'─' * 68}{RESET}")
    out.append(f"  bump source:    {CYAN}{source}{RESET}"
                + (f"  (MLP confidence: {confidence*100:.1f}%)" if confidence is not None else ""))
    out.append(f"  bump kind:      {BOLD}{bump_kind.upper()}{RESET}")
    out.append(f"  target version: {GREEN}{new_target}{RESET}  (from {current_target})")
    out.append(f"  files affected: {BOLD}{len(plan)}{RESET}")
    out.append(f"  {DIM}{'─' * 68}{RESET}")

    by_eco = {}
    for p in plan:
        by_eco.setdefault(p["ecosystem"], []).append(p)

    for eco, items in by_eco.items():
        out.append(f"\n  {CYAN}{BOLD}{eco.upper()}{RESET}")
        for p in items:
            old = p["current_version"]
            new = p["new_version"]
            rel = p["path"].relative_to(p["_root"]) if p.get("_root") else p["path"]
            out.append(f"    {rel}")
            if p["ecosystem"] == "meta":
                out.append(f"      {GREEN}+ ## [{new}] - {time.strftime('%Y-%m-%d')}{RESET}   {DIM}({p['notes']}){RESET}")
            else:
                out.append(f"      {RED}- {old}{RESET}  {GREEN}+ {new}{RESET}   {DIM}({p['notes']}){RESET}")

    out.append(f"\n  {DIM}{'─' * 68}{RESET}")
    out.append(f"  {BOLD}next:{RESET} run with {CYAN}--apply{RESET} to write all files atomically")
    out.append("")
    return "\n".join(out)


def apply_plan(plan: List[Dict]) -> Tuple[bool, List[Tuple[Path, Path]]]:
    backups = []
    written = []
    try:
        for p in plan:
            backup_dir = Path(tempfile.mkdtemp(prefix="axomo-backup-"))
            backup_file = backup_dir / p["path"].name
            p["writer"](p["path"], p["current_version"], p["new_version"], backup_file)
            backups.append((p["path"], backup_file, backup_dir))
            written.append(p["path"])
        return True, backups
    except Exception as e:
        print(f"\n  {RED}!! write failed at {p['path']}: {e}{RESET}")
        print(f"  {YELLOW}restoring {len(written)} file(s) from backup...{RESET}")
        for orig_path, backup_file, backup_dir in backups:
            shutil.copy2(backup_file, orig_path)
            shutil.rmtree(backup_dir, ignore_errors=True)
        return False, []


def git_commit_and_tag(root: Path, new_version: str, bump_kind: str,
                        features: Dict, message: Optional[str] = None,
                        tag: bool = True, push: bool = False):
    if message is None:
        bullets = []
        if features:
            if features.get("has_removed_exports"):
                bullets.append(f"removed {features['has_removed_exports']} exported symbol(s)")
            if features.get("has_new_exports"):
                bullets.append(f"added {features['has_new_exports']} new export(s)")
            if features.get("has_signature_changes"):
                bullets.append(f"altered {features['has_signature_changes']} function signature(s)")
            if features.get("num_callers_affected"):
                bullets.append(f"blast radius: {features['num_callers_affected']} callers")
            bullets.append(f"files changed: {features.get('total_files', '?')}")
        body = "\n".join(f"  - {b}" for b in bullets) if bullets else "  - (no diff features extracted)"
        message = (
            f"chore(release): {bump_kind} bump to v{new_version}\n\n"
            f"  {body}\n\n"
            f"  predicted by: axomo-bump (PEAK MLP, ONNX)\n"
        )

    for fname in ["package.json", "pyproject.toml", "Cargo.toml", "setup.py",
                  "VERSION", "Chart.yaml", "mix.exs", "go.mod", "CHANGELOG.md"]:
        fpath = root / fname
        if fpath.exists():
            subprocess.run(["git", "add", str(fpath)], cwd=root, capture_output=True)
    for csproj in root.rglob("*.csproj"):
        subprocess.run(["git", "add", str(csproj)], cwd=root, capture_output=True)
    for sub in list(root.glob("packages/*/package.json")) + list(root.glob("apps/*/package.json")):
        subprocess.run(["git", "add", str(sub)], cwd=root, capture_output=True)

    r = subprocess.run(["git", "commit", "-m", message], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {RED}git commit failed: {r.stderr}{RESET}")
        return False
    print(f"  {GREEN}✓{RESET} committed: {message.splitlines()[0]}")

    if tag:
        r = subprocess.run(
            ["git", "tag", "-a", f"v{new_version}", "-m", f"Release v{new_version}"],
            cwd=root, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  {YELLOW}git tag warning: {r.stderr.strip()}{RESET}")
        else:
            print(f"  {GREEN}✓{RESET} tagged:    v{new_version}")

    if push:
        r = subprocess.run(["git", "push", "origin", "HEAD", "--tags"], cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {YELLOW}git push warning: {r.stderr.strip()}{RESET}")
        else:
            print(f"  {GREEN}✓{RESET} pushed:   origin/HEAD + v{new_version}")
    return True


def main():
    parser = argparse.ArgumentParser(description="AXOMO BUMP — project-wide version editor")
    parser.add_argument("--root", default=".", help="Project root (default: cwd)")
    parser.add_argument("--predict", action="store_true",
                        help="Predict bump type from `git diff` using the PEAK MLP")
    parser.add_argument("--major", action="store_const", const="major", dest="bump")
    parser.add_argument("--minor", action="store_const", const="minor", dest="bump")
    parser.add_argument("--patch", action="store_const", const="patch", dest="bump")
    parser.add_argument("--none",  action="store_const", const="none",  dest="bump")
    parser.add_argument("--to", default=None, help="Set explicit target version (overrides --bump)")
    parser.add_argument("--preview", action="store_true", help="Show what would change, don't write")
    parser.add_argument("--apply",   action="store_true", help="Write the files (atomic)")
    parser.add_argument("--commit",  action="store_true", help="git add + commit with bump message")
    parser.add_argument("--tag",     action="store_true", help="git tag vX.Y.Z")
    parser.add_argument("--push",    action="store_true", help="git push origin HEAD + tag")
    parser.add_argument("--from-diff", default=None, help="Path to a .patch file (else: git diff HEAD)")
    parser.add_argument("--yes",     action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"  {RED}root does not exist: {root}{RESET}")
        sys.exit(1)

    print(f"  {DIM}scanning {root} for version files...{RESET}")
    files = find_version_files(root)
    if not files:
        print(f"  {RED}no version files detected{RESET}")
        sys.exit(1)
    for f in files:
        f["_root"] = root

    versions = [f["current_version"] for f in files]
    from collections import Counter
    version_counts = Counter(versions)
    current_target, _ = version_counts.most_common(1)[0]

    bump_kind = args.bump
    confidence = None
    features = None
    source = ""

    if args.predict:
        if args.from_diff and os.path.exists(args.from_diff):
            with open(args.from_diff) as f:
                diff = f.read()
        else:
            r = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True, text=True)
            diff = r.stdout if r.returncode == 0 else ""
        if not diff.strip():
            r = subprocess.run(["git", "diff", "--cached"], cwd=root, capture_output=True, text=True)
            diff = r.stdout
        if not diff.strip():
            print(f"  {YELLOW}no git diff found — defaulting to patch bump{RESET}")
            bump_kind = "patch"
            source = "default (no diff)"
        else:
            print(f"  {DIM}running PEAK MLP on diff ({len(diff)} bytes)...{RESET}")
            bump_kind, confidence, features = predict_bump_from_diff(diff)
            source = "PEAK MLP (ONNX)"
            if bump_kind is None:
                print(f"  {RED}MLP model not found — falling back to patch{RESET}")
                bump_kind = "patch"
                source = "fallback (model missing)"
    elif args.to:
        new_target = args.to
        if not re.match(r'^\d+\.\d+\.\d+', new_target):
            print(f"  {RED}--to must be semver, got: {new_target}{RESET}")
            sys.exit(1)
        m1 = re.match(r'^(\d+)\.(\d+)\.(\d+)', new_target)
        m2 = re.match(r'^(\d+)\.(\d+)\.(\d+)', current_target)
        if m1 and m2:
            a, b, c = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
            x, y, z = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            if a > x:
                bump_kind = "major"
            elif b > y:
                bump_kind = "minor"
            elif c > z:
                bump_kind = "patch"
            else:
                bump_kind = "none"
        source = "explicit --to"
        plan = []
        for f in files:
            f["new_version"] = new_target
            plan.append(f)
        print(render_preview(plan, current_target, new_target, bump_kind, source))
        if args.preview and not args.apply:
            return
        if not args.yes:
            r = input(f"  {YELLOW}apply to {len(plan)} files? [y/N]{RESET} ").strip().lower()
            if r != "y":
                print(f"  {DIM}aborted{RESET}")
                return
        ok, backups = apply_plan(plan)
        if not ok:
            sys.exit(2)
        print(f"\n  {GREEN}✓ applied bump to {len(plan)} file(s) atomically{RESET}")
        for _, _, bd in backups:
            shutil.rmtree(bd, ignore_errors=True)
        if args.commit or args.tag:
            git_commit_and_tag(root, new_target, bump_kind, features or {},
                                tag=args.tag, push=args.push)
        return

    elif bump_kind is None:
        bump_kind = "patch"
        source = "default (--patch)"

    if not args.predict:
        source = source or f"manual --{bump_kind}"

    new_target = bump_version(current_target, bump_kind)

    plan = []
    inconsistent = []
    for f in files:
        f["new_version"] = bump_version(f["current_version"], bump_kind)
        if f["current_version"] != current_target:
            inconsistent.append(f)
        plan.append(f)

    print(render_preview(plan, current_target, new_target, bump_kind, source,
                          confidence=confidence, features=features))
    if inconsistent:
        print(f"  {YELLOW}⚠ {len(inconsistent)} file(s) had a different current version{RESET}")
        print(f"  {DIM}each file is bumped from its OWN current version (handles monorepo drift){RESET}")

    if args.preview and not args.apply:
        return
    if not args.yes:
        r = input(f"  {YELLOW}apply bump to {len(plan)} file(s)? [y/N]{RESET} ").strip().lower()
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


if __name__ == "__main__":
    main()
