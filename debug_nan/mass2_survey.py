"""Survey the encoder/decoder p4 outputs for float32-degenerate pair masses.

`cylindrical_physics_features(..., mass_from_energy=True)` -- the decoder's
condition -- computes ``mass2 = (E1+E2)**2 - |p1+p2|**2`` in float32 and then
``sqrt(clamp(mass2, min=0.0))``.  For a boosted, nearly collinear pair the
difference of squares cancels and float32 can return ``mass2 <= 0``; the
clamped sqrt then has an infinite derivative and the backward pass turns NaN.

This script measures how often that happens for the real model outputs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[1]
for _directory in (REPO, REPO / "scripts", REPO / "scripts_sota", REPO / "scripts_joint"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cms_data import load_config  # noqa: E402
from joint_data import load_joint_regions, resolve_joint_config  # noqa: E402
from joint_model import build_joint_autoencoder  # noqa: E402
from joint_trainer import restore_joint_checkpoint  # noqa: E402

RUN_DIR = REPO / "outputs" / "cms_Joint" / "Run_H_A2frozen"
CONFIG_PATH = REPO / "configs_joint" / "cms_Joint_runH_A2frozen.yaml"


def naive_mass2(values: torch.Tensor) -> torch.Tensor:
    """Exactly what `cylindrical_physics_features(mass_from_energy=True)` does."""
    pair = values[:, 0:4] + values[:, 4:8]
    return pair[:, 3] ** 2 - (pair[:, 0] ** 2 + pair[:, 1] ** 2 + pair[:, 2] ** 2)


def survey(tag: str, values: torch.Tensor) -> None:
    mass2 = naive_mass2(values)
    total = mass2.numel()
    non_positive = int((mass2 <= 0.0).sum())
    physical = float((mass2.detach() - 2 * 0.1056583755**2).max())
    pair = values[:, 0:4] + values[:, 4:8]
    pmag = torch.linalg.norm(pair[:, :3], dim=1)
    print(
        f"{tag:>28}: n={total} mass2<=0: {non_positive} "
        f"({100.0 * non_positive / max(1, total):.4g}%) | "
        f"min mass2={float(mass2.min()):.4g} "
        f"max|p_pair|={float(pmag.max()):.4g} "
        f"p99.99|p|={float(torch.quantile(pmag.double(), 0.9999)):.4g}"
    )
    if non_positive:
        bad = mass2 <= 0.0
        print(f"      |p_pair| of failures: min={float(pmag[bad].min()):.4g} "
              f"max={float(pmag[bad].max()):.4g}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=300000)
    parser.add_argument("--checkpoint", default=str(RUN_DIR / "last_model.pt"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = resolve_joint_config(load_config(CONFIG_PATH))
    device = torch.device("cpu") if args.cpu else torch.device("cuda")
    region_arrays, _, _, _ = load_joint_regions(
        config, num_samples=None, use_cache=True, log=lambda *a: None
    )
    model = build_joint_autoencoder(
        config["model"],
        region_arrays,
        float(config.get("muon_mass_gev", 0.1056583755)),
        config["model"].get("daughter_masses"),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    restore_joint_checkpoint(model, checkpoint)
    model.eval()

    rng = np.random.default_rng(1234)
    with torch.no_grad():
        for name in config["region_order"]:
            arrays = region_arrays[name]
            for key in ("x_train", "z_train"):
                values = arrays[key]
                take = min(args.events, len(values))
                index = np.sort(rng.choice(len(values), size=take, replace=False))
                batch = torch.as_tensor(
                    np.ascontiguousarray(values[index]), dtype=torch.float32, device=device
                )
                survey(f"{name}/{key} (input)", batch)
                if key == "x_train":
                    encoded = model.encoder(batch)
                    survey(f"{name}/encoded(x) [dec in]", encoded)
                    decoded = model.decoder(encoded)
                    survey(f"{name}/dec(enc(x))", decoded)
                else:
                    decoded = model.decoder(batch)
                    survey(f"{name}/dec(z)", decoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
