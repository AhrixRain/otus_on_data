#!/usr/bin/env python
"""Convenience launcher for the first cms_Joint experiment: Run A."""

from pathlib import Path

from run_joint import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs_joint" / "cms_Joint_runA.yaml"
    raise SystemExit(main(config))
