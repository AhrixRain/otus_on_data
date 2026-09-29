#!/usr/bin/env python
"""Run the fake J/psi -> Upsilon scaling probe end to end.

Steps:
1. Build a fake Upsilon(1S) prior from the J/psi CKKW-L signal component.
2. Build a true Upsilon(1S) subset with the same event count.
3. Decode both with the frozen Run E checkpoint(s).
4. Compare the decoded mass shifts and widths.

Outputs are written under:
    experiments/prior_JpsiScalingTest/outputs/
    experiments/prior_JpsiScalingTest/plots/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

MAKE_FAKE = HERE / "make_fake_prior.py"
MAKE_TRUE = HERE / "make_true_upsilon1s_subset.py"
DECODE = REPO_ROOT / "scripts_joint" / "upsilon" / "decode_prior.py"
COMPARE = HERE / "compare_probe.py"
RUN_DIR = REPO_ROOT / "outputs" / "cms_Joint" / "Run_E"

DEFAULT_FAKE = HERE / "outputs" / "fake_jpsi_to_upsilon1s.hdf5"
DEFAULT_TRUE = HERE / "outputs" / "true_upsilon1s_subset.hdf5"
DECODED_DIR = HERE / "outputs" / "decoded"
JSON_DIR = HERE / "outputs"
PLOT_DIR = HERE / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        choices=["best", "last"],
        default=["best", "last"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(str(item) for item in cmd))
    subprocess.run([str(item) for item in cmd], cwd=REPO_ROOT, check=True)


def main() -> int:
    args = parse_args()
    DECODED_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    run([sys.executable, MAKE_FAKE, "--max-events", args.max_events,
         "--output", DEFAULT_FAKE])
    run([sys.executable, MAKE_TRUE, "--max-events", args.max_events,
         "--output", DEFAULT_TRUE])

    checkpoint_paths = {
        "best": RUN_DIR / "best_model.pt",
        "last": RUN_DIR / "last_model.pt",
    }

    for checkpoint in args.checkpoints:
        checkpoint_path = checkpoint_paths[checkpoint]
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        fake_decoded = DECODED_DIR / f"{checkpoint}_fake_jpsi_scaled_to_upsilon1s.hdf5"
        true_decoded = DECODED_DIR / f"{checkpoint}_true_upsilon1s.hdf5"

        run([
            sys.executable, DECODE,
            "--prior", DEFAULT_FAKE,
            "--checkpoint", checkpoint_path,
            "--output", fake_decoded,
            "--device", args.device,
            "--max-events", args.max_events,
            "--overwrite",
        ])
        run([
            sys.executable, DECODE,
            "--prior", DEFAULT_TRUE,
            "--checkpoint", checkpoint_path,
            "--output", true_decoded,
            "--device", args.device,
            "--max-events", args.max_events,
            "--overwrite",
        ])

        output_json = JSON_DIR / f"comparison_{checkpoint}.json"
        output_plot = PLOT_DIR / f"fake_vs_true_{checkpoint}.png"
        run([
            sys.executable, COMPARE,
            "--fake-decoded", fake_decoded,
            "--true-decoded", true_decoded,
            "--checkpoint", checkpoint,
            "--output-json", output_json,
            "--output-plot", output_plot,
            "--output-prior-plot", PLOT_DIR / "fake_prior_vs_true_prior.png",
        ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
