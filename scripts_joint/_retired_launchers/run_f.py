#!/usr/bin/env python
"""Launch cms_Joint Run F with CMS-matched inclusive 0j+1j J/psi and Z priors."""

from pathlib import Path

from run_joint import main


if __name__ == "__main__":
    config = (
        Path(__file__).resolve().parents[1]
        / "configs_joint"
        / "cms_Joint_runF.yaml"
    )
    raise SystemExit(main(config))
