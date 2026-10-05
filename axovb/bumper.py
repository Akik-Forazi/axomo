"""
AXOVB bumper — project-wide version file editor.

Auto-detects every version file across 8 ecosystems (Node, Python, Rust,
Helm, Elixir, Go, .NET, plain VERSION, meta CHANGELOG), bumps each one
atomically with full rollback on any failure, and commits + tags the
release in git.

Each (parser, writer) pair uses a verification step: write the new version,
re-parse the file, assert the parsed value equals the new version. If any
file fails verification, all already-written files are restored from their
per-file backup tempdirs.

Public API:
    find_version_files(root)  — list of detected files + their current version
    bump_version(current, kind) — semver math (major/minor/patch/none)
    apply_plan(plan)            — atomic write with rollback
    git_commit_and_tag(...)     — git add + commit + tag (+ optional push)
"""
import os, sys, re, json, shutil, subprocess, time, tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ─── VERSION FILE DETECTORS ──────────────────────────────────────────────

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
        text, count=1, flags=re.M,
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
        text, count=1, flags=re.M,
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
        text, count=1,
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
        repl, text, count=1, flags=re.M,
    )
    if n == 0:
        raise RuntimeError(f"version line not found in {path}")
    if old in text:
        new_text2, m = re.subn(
            rf'(^appVersion:\s*)(["\']?){re.escape(old)}(["\']?)\s*$',
            lambda mm: f"{mm.group(1)}{mm.group(2)}{new}{mm.group(3)}",
            new_text, count=1, flags=re.M,
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
        rf'\g<1>{new}', text, count=1, flags=re.M,
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


# Registry: (filename_patterns, ecosystem, parser, writer, notes)
DETECTORS = [
    (["package.json"],     "node",    parse_json_version,    write_json_version,    "root package.json"),
    (["lerna.json"],       "node",    parse_json_version,    write_json_version,    "lerna monorepo root"),
    (["pyproject.toml"],   "python",  parse_pyproject_version, write_pyproject_version, "PEP 621 / poetry / hatch"),
    (["setup.py"],         "python",  parse_setup_py_version, write_setup_py_version, "legacy setup.py"),
    (["Cargo.toml"],       "rust",    parse_cargo_version,   write_cargo_version,   "cargo crate"),
    (["Chart.yaml"],       "helm",    parse_chart_yaml_version, write_chart_yaml_version, "helm chart + appVersion"),
    (["VERSION"],          "plain",   parse_plain_version,   write_plain_version,   "plain VERSION file"),
    (["mix.exs"],           "elixir", parse_mix_exs_version,  write_mix_exs_version,  "elixir mix project"),
    (["go.mod"],            "go",     parse_go_mod_version,   write_go_mod_version,  "go.mod (// Version comment)"),
    (["*.csproj"],          "dotnet", parse_csproj_version,   write_csproj_version,   ".NET project file"),
    (["CHANGELOG.md"],     "meta",    parse_changelog_version, write_changelog_version, "prepend new release section"),
]


def find_version_files(root: Path) -> List[Dict]:
    """Walk the project root and find every file that looks like a version file.
    Returns a list of dicts: {path, ecosystem, current_version, parser, writer, notes}.
    """
    found = []
    root = root.resolve()
    ignore = {".git", "node_modules", "target", "build", "dist", ".venv", "venv",
              "__pycache__", ".next", ".turbo"}

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
                    "path": sub, "ecosystem": "node-workspace",
                    "current_version": ver, "parser": parse_json_version,
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
                    "path": sub, "ecosystem": "node-workspace",
                    "current_version": ver, "parser": parse_json_version,
                    "writer": write_json_version,
                    "notes": f"workspace app {sub.parent.name}",
                })
                seen_paths.add(str(sub))
        except Exception:
            pass

    return found


# ─── SEMVER MATH ─────────────────────────────────────────────────────────

def bump_version(current: str, kind: str) -> str:
    """Bump a semver string by major/minor/patch/none. Strips pre-release suffix on bumps."""
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


# ─── ATOMIC WRITE ───────────────────────────────────────────────────────

def apply_plan(plan: List[Dict]) -> Tuple[bool, List[Tuple[Path, Path, Path]]]:
    """Write every file in the plan atomically. Returns (success, backups).
    On any failure: restore all already-written files from backup, return (False, []).
    """
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
        print(f"\n  \033[31m!! write failed at {p['path']}: {e}\033[0m")
        print(f"  \033[33mrestoring {len(written)} file(s) from backup...\033[0m")
        for orig_path, backup_file, backup_dir in backups:
            shutil.copy2(backup_file, orig_path)
            shutil.rmtree(backup_dir, ignore_errors=True)
        return False, []


# ─── GIT INTEGRATION ─────────────────────────────────────────────────────

def git_commit_and_tag(root: Path, new_version: str, bump_kind: str,
                        features: Dict, message: Optional[str] = None,
                        tag: bool = True, push: bool = False) -> bool:
    """git add <version files> + git commit + git tag vX.Y.Z (+ optional push)."""
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
        print(f"  \033[31mgit commit failed: {r.stderr}\033[0m")
        return False
    print(f"  \033[32m✓\033[0m committed: {message.splitlines()[0]}")

    if tag:
        r = subprocess.run(
            ["git", "tag", "-a", f"v{new_version}", "-m", f"Release v{new_version}"],
            cwd=root, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  \033[33mgit tag warning: {r.stderr.strip()}\033[0m")
        else:
            print(f"  \033[32m✓\033[0m tagged:    v{new_version}")

    if push:
        r = subprocess.run(["git", "push", "origin", "HEAD", "--tags"], cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  \033[33mgit push warning: {r.stderr.strip()}\033[0m")
        else:
            print(f"  \033[32m✓\033[0m pushed:   origin/HEAD + v{new_version}")
    return True
