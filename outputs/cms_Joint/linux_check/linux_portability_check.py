#!/usr/bin/env python3
"""Linux portability check for the OTUS repository.

Runs ON Linux. Answers: can this checkout execute here?
Checks, in order:
  1. platform/python report
  2. every first-party .py compiles and parses under a Python 3.9 grammar
  3. Windows-only constructs in first-party code
  4. config data references resolve with EXACT case (case-sensitive FS)
  5. locale-dependent text IO, silent CPU fallback, shebang/exec-bit issues
"""

from __future__ import annotations

import ast
import os
import platform
import re
import stat
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
SKIP_DIRS = {".git", "outputs", "data", "litreview", "node_modules", "__pycache__", ".venv", "venv"}

WINDOWS_ISMS = {
    "drive_letter_path": re.compile(r"[A-Za-z]:[\\/][A-Za-z0-9_]"),
    "unc_path": re.compile(r"\\\\\\\\[A-Za-z0-9_]"),
    "darwin_loader_path": re.compile(r"DYLD_LIBRARY_PATH"),
    "macos_caffeinate": re.compile(r"\bcaffeinate\b"),
    "shell_true": re.compile(r"shell\s*=\s*True"),
    "os_system": re.compile(r"\bos\.system\("),
    "winreg_ctypes": re.compile(r"winreg|ctypes\.windll|os\.startfile"),
    "pure_windows_path": re.compile(r"PureWindowsPath"),
    "hgfs_share": re.compile(r"/mnt/hgfs"),
}
TEXT_IO = re.compile(r"\.(read_text|write_text)\(\s*\)")
OPEN_NO_ENCODING = re.compile(r"\bopen\((?!.*encoding=)")
CUDA_DEFAULT = re.compile(r'default\s*=\s*"cuda"')


def first_party_py():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                yield Path(base) / name


def main() -> int:
    print("=" * 72)
    print(f"1. PLATFORM   {platform.platform()}")
    print(f"   python     {sys.version.split()[0]}  ({sys.executable})")
    print(f"   repo       {ROOT}")
    print(f"   case-sensitive FS: {not (ROOT / 'DATA').exists()} (probe file DATA must not exist)")
    findings = {"parse_fail": [], "windows": {}, "text_io": [], "cuda_default": [], "no_exec": [], "shebang": []}

    files = sorted(first_party_py())
    print(f"\n2. PARSE / COMPILE  ({len(files)} first-party .py outside outputs, data, litreview)")
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            findings["parse_fail"].append((path, f"not UTF-8: {error}"))
            continue
        try:
            ast.parse(source, filename=str(path), feature_version=(3, 9))
        except SyntaxError as error:
            findings["parse_fail"].append((path, f"{error.msg} (line {error.lineno})"))
        for label, pattern in WINDOWS_ISMS.items():
            for match in pattern.finditer(source):
                line = source[: match.start()].count("\n") + 1
                findings["windows"].setdefault(label, []).append(f"{path.relative_to(ROOT)}:{line}")
        for match in TEXT_IO.finditer(source):
            line = source[: match.start()].count("\n") + 1
            findings["text_io"].append(f"{path.relative_to(ROOT)}:{line} {match.group(1)}() without encoding=")
        for match in CUDA_DEFAULT.finditer(source):
            line = source[: match.start()].count("\n") + 1
            findings["cuda_default"].append(f"{path.relative_to(ROOT)}:{line}")
        if path.suffix == ".py" and source.startswith("#!"):
            if "python" in source.split("\n")[0] and "python3" not in source.split("\n")[0]:
                findings["shebang"].append(f"{path.relative_to(ROOT)}: {source.splitlines()[0]}")

    print(f"   parse failures (py3.9 grammar): {len(findings['parse_fail'])}")
    for path, message in findings["parse_fail"][:20]:
        print(f"     FAIL {path.relative_to(ROOT)}: {message}")

    print("\n3. WINDOWS-ONLY CONSTRUCTS")
    if not findings["windows"]:
        print("   none")
    for label, hits in sorted(findings["windows"].items()):
        print(f"   {label}: {len(hits)}")
        for hit in hits[:8]:
            print(f"     {hit}")

    print("\n4. CONFIG DATA REFERENCES, EXACT CASE")
    configs = sorted(ROOT.glob("configs*/*.yaml"))
    refs, missing = 0, []
    for cfg in configs:
        text = cfg.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"^\s*(?:file|theory_prior_file|cms_root_file|root_file|prior_file)\s*:\s*(\S+)\s*$", text, re.M):
            value = match.group(1).strip().strip('"\'')
            if value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", value):
                continue
            refs += 1
            if not (ROOT / "data" / value).exists():
                missing.append(f"{cfg.name}: data/{value}")
    print(f"   {refs} relative data references across {len(configs)} configs; {len(missing)} unresolved")
    for item in missing[:20]:
        print(f"     MISSING {item}")

    print("\n5. TEXT IO / DEVICE / EXEC BIT")
    print(f"   read_text()/write_text() without encoding: {len(findings['text_io'])}")
    for hit in findings["text_io"][:10]:
        print(f"     {hit}")
    print(f"   argparse default=\"cuda\": {len(findings['cuda_default'])}")
    for hit in findings["cuda_default"][:10]:
        print(f"     {hit}")
    shells = [p for p in ROOT.rglob("*.sh") if ".git" not in p.parts]
    not_exec = [p for p in shells if not (p.stat().st_mode & stat.S_IXUSR)]
    print(f"   .sh files: {len(shells)}, missing exec bit: {len(not_exec)}")
    for path in not_exec[:10]:
        print(f"     {path.relative_to(ROOT)}")
    print(f"   python shebangs that are not python3: {len(findings['shebang'])}")
    return 0 if not findings["parse_fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
