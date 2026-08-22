#!/usr/bin/env python
"""Launch uncapped cms_Joint Run C full-scale training."""

from pathlib import Path

from run_joint import main


if __name__ == "__main__":
    config = (
        Path(__file__).resolve().parents[1]
        / "configs_joint"
        / "cms_Joint_runC_fullScale.yaml"
    )
    raise SystemExit(main(config))
