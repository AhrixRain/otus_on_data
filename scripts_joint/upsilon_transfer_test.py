#!/usr/bin/env python
"""Run the post-training Upsilon transfer test for a cms_Joint run.

This script does NOT train or tune on Upsilon. It decodes the frozen run
checkpoint with the full 0j+1j Upsilon prior and then runs the existing
comparison/evaluation/plotting scripts.
"""


from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSILON_DIR = REPO_ROOT / "scripts_joint" / "upsilon"

DEFAULT_PRIOR = (
    REPO_ROOT
    / "data"
    / "cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5"
)
DEFAULT_CMS = REPO_ROOT / "experiments" / "cms_upsilon" / "data" / "Ymumu.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to decode with. Defaults to <run-dir>/best_model.pt. "
        "Use last_model.pt to compare arms whose best_model.pt was chosen by "
        "different rules - the A/B narrow arms fell back to the final "
        "checkpoint (see global_gate_fallback.txt) while the smeared arm was "
        "gate-selected, so their best_model.pt files are not comparable.",
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--cms", type=Path, default=DEFAULT_CMS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
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
    checkpoint = args.checkpoint or (run_dir / "best_model.pt")
    if not checkpoint.is_absolute():
        checkpoint = (run_dir / checkpoint) if not checkpoint.exists() else checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Train the run first with scripts_joint/run_joint.py --run <ID>"
        )

    # Keep per-checkpoint results side by side rather than overwriting a
    # previous transfer test (CLAUDE.md section 3: new analysis files into a run
    # directory are fine; changing an existing artifact is not).
    transfer_dir = run_dir / "upsilon_transfer"
    if args.checkpoint is not None:
        transfer_dir = run_dir / f"upsilon_transfer_{checkpoint.stem}"
    decoded_dir = transfer_dir / "decoded"
    decoded_path = decoded_dir / "upsilon_0j1j_prior_decoded_xspace.hdf5"
    decoded_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_decode:
        cmd = [
            sys.executable,
            UPSILON_DIR / "decode_prior.py",
            "--run-dir", run_dir,
            "--checkpoint", checkpoint,
            "--prior", args.prior,
            "--output", decoded_path,
            "--device", args.device,
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        run(cmd)
    else:
        if not decoded_path.exists():
            raise FileNotFoundError(f"--skip-decode requested but {decoded_path} is missing")

    # Raw prior vs CMS (for reference; no model involved)
    raw_plot_dir = transfer_dir / "prior_vs_cms"
    run([
        sys.executable,
        UPSILON_DIR / "compare_prior_cms.py",
        "--prior", args.prior,
        "--cms", args.cms,
        "--output-dir", raw_plot_dir,
        "--sample-label", "Upsilon 0j+1j prior",
    ])

    # Decoded x-space vs CMS
    decoded_plot_dir = transfer_dir / "decoded_vs_cms"
    run([
        sys.executable,
        UPSILON_DIR / "compare_prior_cms.py",
        "--prior", decoded_path,
        "--prior-dataset", "FDL/xData",
        "--cms", args.cms,
        "--output-dir", decoded_plot_dir,
        "--sample-label", f"{run_dir.name}/{checkpoint.stem} decoded prior",
    ])

    # Full quantitative z->x report + plots
    quant_dir = transfer_dir / "quantitative_z_to_x"
    run([
        sys.executable,
        UPSILON_DIR / "evaluate_z_to_x.py",
        "--decoded", decoded_path,
        "--cms", args.cms,
        "--output-dir", quant_dir,
        "--device", args.device,
    ])

    print(f"\n{run_dir.name} Upsilon transfer test complete.")
    print(f"Decoded:  {decoded_path}")
    print(f"Raw:      {raw_plot_dir}")
    print(f"Decoded:  {decoded_plot_dir}")
    print(f"Quant:    {quant_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
