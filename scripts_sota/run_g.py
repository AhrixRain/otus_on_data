#!/usr/bin/env python
"""Run G/G0 training entry point.

The shared stage trainer and CLI implementation live in ``run_e.py`` for
checkpoint compatibility with completed Runs E/F.  This named entry keeps Run
G commands and provenance unambiguous without forking the training logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "scripts_sota"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from run_e import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
