#!/usr/bin/env python
"""Generate all standard Run F plots after training and Upsilon transfer.

This includes:
  1. training loss curves (log and linear)
  2. per-region all-element diagnostic plots
  3. paper-style density/ratio plots
  4. optional Upsilon transfer plots (if --with-upsilon is given)
"""


from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_JOINT = REPO_ROOT / "scripts_joint"

DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "Run_F"
DEFAULT_CONFIG = REPO_ROOT / "configs_joint" / "cms_Joint_runF.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with-upsilon", action="store_true")
    parser.add_argument("--upsilon-overwrite", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(item) for item in cmd))
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.check_call([str(item) for item in cmd], env=env)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = args.config.expanduser().resolve()
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Run F checkpoint not found: {checkpoint}")

    # 1. Loss curves
    loss_script = SCRIPTS_JOINT / "plot_joint_losses.py"
    run([
        sys.executable, loss_script,
        "--run-a", run_dir,
        "--run-b", run_dir,
        "--output-a", run_dir / "loss_curve.png",
        "--output-b", run_dir / "loss_curve_linear.png",
    ])
    # Overwrite the second file with a linear-y version.
    run([
        sys.executable, loss_script,
        "--run-a", run_dir,
        "--run-b", run_dir,
        "--output-a", run_dir / "loss_curve_linear.png",
        "--output-b", run_dir / "loss_curve_linear_tmp.png",
        "--linear-y",
    ])
    (run_dir / "loss_curve_linear_tmp.png").unlink(missing_ok=True)

    # 2. Per-region all-element diagnostics
    diag_script = SCRIPTS_JOINT / "plot_joint_run.py"
    run([
        sys.executable, diag_script,
        "--config", config,
        "--checkpoint", checkpoint,
        "--output-dir", run_dir / "plots",
        "--device", args.device,
    ])

    # 3. Paper-style density/ratio plots
    paper_script = SCRIPTS_JOINT / "plot_joint_paperstyle.py"
    run([
        sys.executable, paper_script,
        "--config", config,
        "--checkpoint", checkpoint,
        "--output-dir", run_dir / "plots_paperstyle",
        "--device", args.device,
    ])

    # 4. Optional Upsilon transfer plots
    if args.with_upsilon:
        ups_script = SCRIPTS_JOINT / "run_f_upsilon_test.py"
        cmd = [sys.executable, ups_script, "--run-dir", run_dir, "--device", args.device]
        if args.upsilon_overwrite:
            cmd.append("--overwrite")
        run(cmd)

    print("\nRun F plots complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
