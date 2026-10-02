#!/usr/bin/env python3
"""Normalise text line endings to LF.

Why this exists: a `git clone` applies ``.gitattributes`` and hands you LF
files. A `scp`/`rsync` copy does **not** -- it transfers bytes verbatim -- and
139 tracked text files in this repository currently sit in the working tree
with CRLF endings (git reports them clean, because the ``text`` attribute
normalises before comparing, so nothing warns you).

The practical damage is narrow but sharp:

* a ``.py`` with a shebang and CRLF fails when executed directly --
  ``./scripts_joint/run_joint.py`` gives ``bad interpreter: /usr/bin/env python^M``
  -- although ``python scripts_joint/run_joint.py`` works fine;
* the same applies to any ``.sh`` file (none of the shell scripts are affected
  today, but a contributor's editor can reintroduce it);
* ``.yaml`` block scalars are *not* affected: PyYAML normalises line breaks
  inside ``>-`` / ``|`` per the YAML spec, verified against
  ``configs_joint/cms_Joint_runH.yaml`` (identical parsed values, no ``\\r``).

Usage
-----
    python deploy/hpc3/to_lf.py --check     # report only, exit 1 if any CRLF (default action)
    python deploy/hpc3/to_lf.py --apply     # rewrite those files in place

``outputs/``, ``data/`` and ``logs/`` are skipped by default so that committed
artifacts are never rewritten (CLAUDE.md section 3); pass
``--include-artifacts`` if you really mean to touch them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".pyi", ".sh", ".sbatch", ".bash", ".zsh",
    ".yaml", ".yml", ".json", ".jsonc", ".md", ".rst",
    ".txt", ".csv", ".tsv", ".cfg", ".toml", ".ini", ".conf",
    ".ipynb", ".tex", ".bib", ".dat",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cu",
    # .ps1 is deliberately absent: .gitattributes keeps PowerShell scripts CRLF.
}

SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".ipynb_checkpoints", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "build", "dist", ".region_cache",
}

# Never rewrite retained artifacts or bulk data unless asked explicitly.
ARTIFACT_DIR_NAMES = {"outputs", "data", "logs"}


def iter_text_files(root: Path, include_artifacts: bool):
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in SKIP_DIR_NAMES
            and (include_artifacts or name not in ARTIFACT_DIR_NAMES)
        )
        for name in sorted(filenames):
            path = here / name
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def analyse(data: bytes) -> tuple[int, int]:
    """Return (crlf pairs, lone CRs)."""
    crlf = data.count(b"\r\n")
    lone = data.count(b"\r") - crlf
    return crlf, lone


def main() -> int:
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=repo_default,
                        help="tree to scan (default: the repository root)")
    parser.add_argument("--check", action="store_true",
                        help="report only and exit 1 if any file needs rewriting "
                             "(this is the default when --apply is absent)")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite CRLF files as LF")
    parser.add_argument("--include-artifacts", action="store_true",
                        help="also scan outputs/, data/ and logs/ (off by default)")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    if args.check and args.apply:
        parser.error("--check and --apply are mutually exclusive")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    offenders: list[tuple[Path, int, int]] = []
    scanned = 0
    for path in iter_text_files(root, args.include_artifacts):
        try:
            data = path.read_bytes()
        except OSError as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
            continue
        scanned += 1
        if b"\r" not in data:
            continue
        crlf, lone = analyse(data)
        if crlf or lone:
            offenders.append((path, crlf, lone))

    print(f"scanned {scanned} text files under {root}")
    for path, crlf, lone in offenders:
        rel = path.relative_to(root).as_posix()
        note = f"{crlf} CRLF" + (f", {lone} lone CR" if lone else "")
        if args.apply:
            data = path.read_bytes().replace(b"\r\n", b"\n")
            path.write_bytes(data)
            print(f"  rewrote  {rel}  ({note})")
        elif not args.quiet:
            print(f"  CRLF     {rel}  ({note})")

    if not offenders:
        print("all text files already use LF endings")
        return 0

    if args.apply:
        print(f"rewrote {len(offenders)} file(s) to LF")
        return 0

    print(f"\n{len(offenders)} file(s) contain CRLF. Fix with:")
    print(f"    python {Path(__file__).as_posix()} --apply")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
