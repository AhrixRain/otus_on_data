#!/usr/bin/env python
"""Launch cms_Joint Run E with one complete training pass per epoch."""

from pathlib import Path

from run_joint import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs_joint" / "cms_Joint_runE.yaml"
    raise SystemExit(main(config))
