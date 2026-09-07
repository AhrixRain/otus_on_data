#!/usr/bin/env python
"""Convenience launcher for the deeper, balanced-selection cms_Joint Run C."""

from pathlib import Path

from run_joint import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs_joint" / "cms_Joint_runC.yaml"
    raise SystemExit(main(config))
