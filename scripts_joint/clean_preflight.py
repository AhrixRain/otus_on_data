#!/usr/bin/env python
"""Remove dry-run / smoke preflight output directories once they are green.

Working agreement (CLAUDE.md section 3): a `--dry-run` or `--smoke` output
directory is transient. Once its checks pass, delete it before starting the
next piece of work; it must not accumulate in `outputs/`.

The default mode only lists what it would remove. Pass `--apply` to delete.
Only directories whose name carries a `dryrun` or `smoke` token are touched,
and only when they are strictly inside a requested root.

    python scripts_joint/clean_preflight.py --root outputs
    python scripts_joint/clean_preflight.py --root outputs --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# A preflight directory name carries "dryrun" or "smoke" as a token, optionally
# followed by a counter (runG0_stable_smoke2) or wrapped in separators.
_PREFLIGHT_TOKEN = re.compile(r"(?:^|[_-])(?:dryrun|smoke)(?:[_-]|$|[0-9])", re.IGNORECASE)


def is_preflight_name(name: str) -> bool:
    """True when a directory name is a dry-run or smoke preflight artifact."""
    return bool(_PREFLIGHT_TOKEN.search(str(name)))


def find_preflight_dirs(roots: list[Path]) -> list[Path]:
    """Every preflight directory strictly below one of ``roots``, sorted.

    The roots themselves are never returned, even if their name matches, and
    symlinked directories are not followed.
    """
    found: list[Path] = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_dir():
                continue
            if path.resolve() == root:
                continue
            if is_preflight_name(path.name):
                found.append(path)
    return sorted(set(found))


def directory_size_bytes(path: Path) -> int:
    total = 0
    for item in Path(path).rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def remove_preflight_dirs(paths: list[Path], *, apply: bool) -> tuple[int, int]:
    """Delete (or list) the given directories; returns (count, bytes)."""
    count = 0
    total = 0
    for path in paths:
        size = directory_size_bytes(path)
        if apply:
            shutil.rmtree(path)
        count += 1
        total += size
        print(f"{'removed ' if apply else 'would remove '}{path}  ({size / 1024:.1f} KiB)")
    return count, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help="root directory to scan (repeatable; default: outputs under the repo root)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete; without it the command only lists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = args.root or [REPO_ROOT / "outputs"]
    directories = find_preflight_dirs(roots)
    if not directories:
        print("no dry-run/smoke preflight directories found")
        return 0
    count, total = remove_preflight_dirs(directories, apply=args.apply)
    verb = "removed" if args.apply else "would remove"
    print(f"\n{verb} {count} director{'y' if count == 1 else 'ies'}, {total / 1024:.1f} KiB")
    if not args.apply:
        print("re-run with --apply to delete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
