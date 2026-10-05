#!/usr/bin/env python3
"""Case-exact config reference check that does NOT rely on the filesystem.

Compares each config's relative data reference against the REAL directory
listing, component by component, with a case-sensitive string comparison. This
works on a case-insensitive mount (/mnt/c) where (ROOT/'DATA').exists() is
uselessly true.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
DATA = ROOT / "data"


def listing(directory: Path) -> set[str]:
    try:
        return {entry.name for entry in directory.iterdir()}
    except (FileNotFoundError, NotADirectoryError):
        return set()


def resolve_case_exact(relative: str) -> tuple[bool, str | None, list[str]]:
    """Walk the reference under data/ comparing each component exactly."""
    parts = [p for p in relative.split("/") if p not in ("", ".")]
    current = DATA
    checked: list[str] = []
    for part in parts:
        names = listing(current)
        if not names:
            return False, f"directory {current.relative_to(ROOT)} not found", checked
        if part in names:
            checked.append(part)
            current = current / part
            continue
        lowered = {n.lower(): n for n in names}
        if part.lower() in lowered:
            return False, f"CASE MISMATCH {part!r} -> real name {lowered[part.lower()]!r}", checked
        return False, f"no such entry {part!r} in {current.relative_to(ROOT)}", checked
    return True, None, checked


def main() -> int:
    pattern = re.compile(
        r"^\s*(?:file|theory_prior_file|cms_root_file|root_file|prior_file|cms_cache)\s*:\s*(\S+)\s*$",
        re.M,
    )
    configs = sorted(ROOT.glob("configs*/*.yaml"))
    total = ok = 0
    problems: list[str] = []
    for cfg in configs:
        text = cfg.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            value = match.group(1).strip().strip("\"'")
            if value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", value):
                continue
            total += 1
            resolved, reason, _ = resolve_case_exact(value)
            if resolved:
                ok += 1
            else:
                problems.append(f"{cfg.name}: data/{value}  <- {reason}")
    print(f"configs scanned        : {len(configs)}")
    print(f"relative refs resolved : {ok}/{total} case-exact under data/")
    print(f"unresolved             : {len(problems)}")
    for item in problems:
        print(f"  {item}")
    print()
    print("data/ top-level entries (exact names, this is what a Linux clone must reproduce):")
    for name in sorted(listing(DATA)):
        print(f"  {name}")
    print()
    print("data/legacy entries:")
    for name in sorted(listing(DATA / "legacy")):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
