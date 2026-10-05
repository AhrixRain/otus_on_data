#!/usr/bin/env python3
"""Linux-only failure modes that Windows hides:

1. IMPORT CASE: 'import foo' resolving on Windows because foo.py/Foo.py differ
   only by case. On Linux the import fails or resolves to a different module.
2. MODULE COLLISIONS: the same module basename present in more than one of the
   directories the code puts on sys.path, so resolution depends on insert order.
3. GIT CASE COLLISIONS: two tracked paths differing only by case.
"""
from __future__ import annotations
import ast
import collections
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
SEARCH_DIRS = [
    ROOT / "scripts_joint" / "upsilon",
    ROOT / "scripts_joint",
    ROOT / "scripts_sota",
    ROOT / "scripts",
    ROOT,
]
SKIP = {".git", "outputs", "data", "litreview", "__pycache__", "node_modules"}

stdlib = set(getattr(sys, "stdlib_module_names", set()))

# first-party modules available per search dir
available: dict[str, list[str]] = collections.defaultdict(list)
for directory in SEARCH_DIRS:
    if not directory.is_dir():
        continue
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix == ".py":
            available[entry.stem].append(str(directory.relative_to(ROOT)))
        elif entry.is_dir() and (entry / "__init__.py").exists():
            available[entry.name].append(str(directory.relative_to(ROOT)) + "/")

lower_map: dict[str, list[str]] = collections.defaultdict(list)
for name in available:
    lower_map[name.lower()].append(name)

first_party = []
for base, dirs, files in __import__("os").walk(ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP]
    first_party += [Path(base) / f for f in files if f.endswith(".py")]

case_bugs = []
unknown = collections.Counter()
for path in first_party:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        continue
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    for name in modules:
        if name in available or name in stdlib:
            continue
        if name.lower() in lower_map:
            case_bugs.append(
                f"{path.relative_to(ROOT)}: imports {name!r} but only "
                f"{lower_map[name.lower()]} exists (case mismatch)"
            )
        else:
            unknown[name] += 1

print("=" * 72)
print("1. IMPORT CASE MISMATCHES (fatal on Linux, invisible on Windows)")
print(f"   {len(case_bugs)} found")
for item in case_bugs:
    print(f"   BUG {item}")

print("\n2. MODULE COLLISIONS ACROSS sys.path DIRECTORIES")
collisions = {name: dirs for name, dirs in available.items() if len(dirs) > 1}
print(f"   {len(collisions)} names present in more than one search dir")
for name, dirs in sorted(collisions.items()):
    print(f"   {name}: {dirs}")

print("\n3. GIT PATHS DIFFERING ONLY BY CASE")
try:
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    by_lower: dict[str, list[str]] = collections.defaultdict(list)
    for item in listing:
        by_lower[item.lower()].append(item)
    dups = {k: v for k, v in by_lower.items() if len(v) > 1}
    print(f"   {len(dups)} case-colliding path groups in git")
    for key, value in dups.items():
        print(f"   {value}")
except Exception as error:  # noqa: BLE001
    print(f"   git unavailable: {error}")

print("\n4. UNRESOLVED TOP-LEVEL IMPORTS (third-party, expected)")
print(f"   {len(unknown)} distinct names")
for name, count in unknown.most_common(25):
    print(f"   {name} (imported by {count} files)")
