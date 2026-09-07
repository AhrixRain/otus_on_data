#!/usr/bin/env python
"""Convenience launcher for cms_Joint Run B with the dedicated Z dimuon prior."""

from pathlib import Path

from run_joint import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs_joint" / "cms_Joint_runB.yaml"
    raise SystemExit(main(config))
