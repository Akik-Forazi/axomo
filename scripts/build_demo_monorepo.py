#!/usr/bin/env python3
"""Build the demo monorepo with version files across 8 ecosystems."""
import os, subprocess, sys
from pathlib import Path

ROOT = Path('/home/z/my-project/axomo/demo-monorepo')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'packages' / 'auth').mkdir(parents=True, exist_ok=True)
    (ROOT / 'packages' / 'api').mkdir(parents=True, exist_ok=True)
    (ROOT / 'packages' / 'ui').mkdir(parents=True, exist_ok=True)
    (ROOT / 'apps' / 'web').mkdir(parents=True, exist_ok=True)
    (ROOT / 'dotnet').mkdir(parents=True, exist_ok=True)

    files = {
        'package.json': """{
  "name": "axomo-monorepo-demo",
  "version": "1.4.2",
  "private": true,
  "workspaces": ["packages/*", "apps/*"]
}
""",
        'lerna.json': """{
  "version": "1.4.2",
  "npmClient": "yarn",
  "packages": ["packages/*", "apps/*"]
}
""",
        'pyproject.toml': """[project]
name = "axomo-py"
version = "1.4.2"
description = "Python bindings for the axomo CLI"
""",
        'Cargo.toml': """[package]
name = "axomo-rs"
version = "1.4.2"
edition = "2021"

[dependencies]
serde = "1.0"
""",
        'setup.py': """from setuptools import setup
setup(name="axomo-py", version="1.4.2", packages=["axomo"])
""",
        'VERSION': "1.4.2\n",
        'Chart.yaml': """apiVersion: v2
name: axomo
description: AXOMO Helm chart
version: "1.4.2"
appVersion: "1.4.2"
""",
        'mix.exs': """defmodule Axomo.MixProject do
  use Mix.Project
  def project do
    [
      app: :axomo,
      version: "1.4.2",
    ]
  end
end
""",
        'go.mod': """module github.com/axomo/axomo-go

go 1.21

// Version: 1.4.2
""",
        'dotnet/Axomo.csproj': """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <Version>1.4.2</Version>
    <AssemblyVersion>1.4.2</AssemblyVersion>
    <FileVersion>1.4.2</FileVersion>
  </PropertyGroup>
</Project>
""",
        'CHANGELOG.md': """# Changelog

## [1.4.2] - 2025-09-10

- fixed something

## [1.4.1] - 2025-08-15

- initial release
""",
        'packages/auth/package.json': """{
  "name": "@axomo/auth",
  "version": "1.4.2",
  "main": "index.js"
}
""",
        'packages/api/package.json': """{
  "name": "@axomo/api",
  "version": "1.4.2",
  "main": "index.js"
}
""",
        'packages/ui/package.json': """{
  "name": "@axomo/ui",
  "version": "1.4.2",
  "main": "index.js"
}
""",
        'apps/web/package.json': """{
  "name": "@axomo/web",
  "version": "1.4.2",
  "private": true
}
""",
    }

    for rel, content in files.items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        print(f"  wrote {p.relative_to(ROOT)}")

    # init git
    if not (ROOT / '.git').exists():
        subprocess.run(['git', 'init', '-q'], cwd=ROOT)
        subprocess.run(['git', 'config', 'user.email', 'demo@axomo.dev'], cwd=ROOT)
        subprocess.run(['git', 'config', 'user.name', 'AXOMO Demo'], cwd=ROOT)
        subprocess.run(['git', 'add', '.'], cwd=ROOT)
        subprocess.run(['git', 'commit', '-qm', 'init: monorepo at 1.4.2'], cwd=ROOT)
        print(f"  git initialized + initial commit")
    print(f"\n  Demo monorepo ready at {ROOT}")


if __name__ == '__main__':
    main()
