#!/usr/bin/env python
"""Launch full-scale cms_Joint Run D with the showered inclusive Z prior."""

from pathlib import Path

from run_joint import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs_joint" / "cms_Joint_runD.yaml"
    raise SystemExit(main(config))
