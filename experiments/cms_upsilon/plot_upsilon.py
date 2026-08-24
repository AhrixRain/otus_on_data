"""Plot the Upsilon(1S), Upsilon(2S), and Upsilon(3S) peaks in CMS dimuon data.

The default input is the 20k-event CMS Open Data Ymumu.csv outreach sample.
A reduced CMS NanoAOD ROOT file with Muon_* branches is also supported.
"""

from __future__ import annotations

import argparse
import csv
import math
import urllib.request
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "Ymumu.csv"
DEFAULT_OUTPUT = HERE / "figures" / "cms_upsilon_mumu.png"
CMS_CSV_URL = "https://opendata.cern.ch/record/5206/files/Ymumu.csv?download=1"

MASS_RANGE = (8.5, 11.2)
PDG_MASSES_GEV = np.array([9.4603, 10.0233, 10.3552])
STATE_LABELS = ("Upsilon(1S)", "Upsilon(2S)", "Upsilon(3S)")


def download_default_data(destination: Path) -> None:
    """Download the small CMS outreach sample unless it is already present."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CMS Open Data to {destination} ...")
    urllib.request.urlretrieve(CMS_CSV_URL, destination)


def invariant_mass(
    e1: np.ndarray,
    px1: np.ndarray,
    py1: np.ndarray,
    pz1: np.ndarray,
    e2: np.ndarray,
    px2: np.ndarray,
    py2: np.ndarray,
    pz2: np.ndarray,
) -> np.ndarray:
    """Return invariant masses from two four-vectors, clipping roundoff at 0."""
    mass_squared = (
        (e1 + e2) ** 2
        - (px1 + px2) ** 2
        - (py1 + py2) ** 2
        - (pz1 + pz2) ** 2
    )
    return np.sqrt(np.clip(mass_squared, 0.0, None))


def _find_column(names: Iterable[str], candidates: Iterable[str]) -> str:
    """Find a column name while tolerating capitalization used by CMS CSVs."""
    lookup = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Missing column; tried {', '.join(candidates)}")


def load_csv_masses(path: Path) -> np.ndarray:
    """Reconstruct dimuon masses from a CMS outreach CSV file."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.dtype.names is None:
        raise ValueError(f"No named CSV columns found in {path}")
    data = np.atleast_1d(data)

    names = data.dtype.names
    fields = {
        key: _find_column(names, candidates)
        for key, candidates in {
            "e1": ("E1", "energy1"),
            "px1": ("px1",),
            "py1": ("py1",),
            "pz1": ("pz1",),
            "e2": ("E2", "energy2"),
            "px2": ("px2",),
            "py2": ("py2",),
            "pz2": ("pz2",),
        }.items()
    }
    return invariant_mass(*(np.asarray(data[fields[key]], dtype=float) for key in fields))


def load_root_masses(path: Path, step_size: str = "100 MB") -> np.ndarray:
    """Reconstruct masses from a reduced NanoAOD ROOT file in memory-safe chunks."""
    try:
        import awkward as ak
        import uproot
    except ImportError as exc:
        raise RuntimeError("ROOT input requires the awkward and uproot packages") from exc

    branches = [
        "nMuon",
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_mass",
        "Muon_charge",
    ]
    selected_chunks: list[np.ndarray] = []
    source = f"{path}:Events"

    for arrays in uproot.iterate(source, branches, step_size=step_size, library="ak"):
        exactly_two = arrays["nMuon"] == 2
        arrays = arrays[exactly_two]
        opposite_sign = arrays["Muon_charge"][:, 0] != arrays["Muon_charge"][:, 1]
        arrays = arrays[opposite_sign]

        pt = arrays["Muon_pt"]
        eta = arrays["Muon_eta"]
        phi = arrays["Muon_phi"]
        mu_mass = arrays["Muon_mass"]

        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px**2 + py**2 + pz**2 + mu_mass**2)

        masses = invariant_mass(
            *(
                ak.to_numpy(component[:, index])
                for index in (0, 1)
                for component in (energy, px, py, pz)
            )
        )
        in_window = (masses >= MASS_RANGE[0]) & (masses <= MASS_RANGE[1])
        selected_chunks.append(masses[in_window])

    if not selected_chunks:
        return np.array([], dtype=float)
    return np.concatenate(selected_chunks)


def gaussian_area(x: np.ndarray, area: float, mean: float, sigma: float) -> np.ndarray:
    """Gaussian normalized so that `area` is its integral in event units."""
    return area * np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (
        math.sqrt(2.0 * math.pi) * sigma
    )


def spectrum_model(x: np.ndarray, *parameters: float) -> np.ndarray:
    """Quadratic exponential background plus three Gaussian signal peaks."""
    log_norm, slope, curvature = parameters[:3]
    centered = x - 9.8
    result = np.exp(log_norm + slope * centered + curvature * centered**2)
    for offset in range(3, 12, 3):
        result = result + gaussian_area(x, *parameters[offset : offset + 3])
    return result


def fit_spectrum(
    centers: np.ndarray, counts: np.ndarray, bin_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the binned spectrum with Poisson-like uncertainties."""
    density = counts / bin_width
    uncertainty = np.sqrt(np.maximum(counts, 1.0)) / bin_width

    background_guess = max(float(np.percentile(density, 25)), 1.0)
    peak_areas = []
    for mass in PDG_MASSES_GEV:
        local = np.abs(centers - mass) < 0.12
        excess = np.maximum(density[local] - background_guess, 0.0)
        peak_areas.append(max(float(np.sum(excess) * bin_width), 10.0))

    initial = [
        math.log(background_guess),
        -0.2,
        0.0,
        peak_areas[0],
        PDG_MASSES_GEV[0],
        0.075,
        peak_areas[1],
        PDG_MASSES_GEV[1],
        0.090,
        peak_areas[2],
        PDG_MASSES_GEV[2],
        0.105,
    ]
    lower = [
        -10.0,
        -5.0,
        -2.0,
        0.0,
        9.35,
        0.025,
        0.0,
        9.92,
        0.025,
        0.0,
        10.25,
        0.025,
    ]
    upper = [
        20.0,
        5.0,
        2.0,
        np.inf,
        9.56,
        0.25,
        np.inf,
        10.12,
        0.25,
        np.inf,
        10.46,
        0.25,
    ]
    parameters, covariance = curve_fit(
        spectrum_model,
        centers,
        density,
        p0=initial,
        bounds=(lower, upper),
        sigma=uncertainty,
        absolute_sigma=True,
        maxfev=100_000,
    )
    return parameters, covariance


def save_fit_results(path: Path, parameters: np.ndarray, covariance: np.ndarray) -> None:
    """Write fitted peak positions, widths, and yields to a compact CSV table."""
    errors = np.sqrt(np.diag(covariance))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "state",
                "pdg_mass_GeV",
                "fitted_mass_GeV",
                "mass_error_GeV",
                "sigma_GeV",
                "sigma_error_GeV",
                "fitted_yield",
                "yield_error",
            ]
        )
        for index, label in enumerate(STATE_LABELS):
            offset = 3 + 3 * index
            writer.writerow(
                [
                    label,
                    f"{PDG_MASSES_GEV[index]:.4f}",
                    f"{parameters[offset + 1]:.6f}",
                    f"{errors[offset + 1]:.6f}",
                    f"{parameters[offset + 2]:.6f}",
                    f"{errors[offset + 2]:.6f}",
                    f"{parameters[offset]:.1f}",
                    f"{errors[offset]:.1f}",
                ]
            )


def plot_spectrum(
    masses: np.ndarray,
    output: Path,
    bins: int = 135,
    do_fit: bool = True,
    sample_label: str = "2011 DoubleMu (education sample)",
) -> None:
    """Make a spectrum matching the plotting conventions used in this project."""
    masses = masses[np.isfinite(masses)]
    masses = masses[(masses >= MASS_RANGE[0]) & (masses <= MASS_RANGE[1])]
    if masses.size == 0:
        raise ValueError(f"No dimuon candidates found in {MASS_RANGE[0]}-{MASS_RANGE[1]} GeV")

    counts, edges = np.histogram(masses, bins=bins, range=MASS_RANGE)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)

    axis.stairs(
        counts,
        edges,
        color="C0",
        linewidth=2.0,
        label="CMS data",
        zorder=2,
    )

    for mass in PDG_MASSES_GEV:
        axis.axvline(mass, color="0.45", linestyle=":", linewidth=1.2, zorder=0)

    if do_fit:
        parameters, covariance = fit_spectrum(centers, counts, bin_width)
        fine_x = np.linspace(*MASS_RANGE, 2000)
        total_curve = spectrum_model(fine_x, *parameters) * bin_width
        axis.plot(
            fine_x,
            total_curve,
            color="C1",
            linewidth=2.0,
            label="3 peaks + background fit",
        )

        centered = fine_x - 9.8
        background = np.exp(
            parameters[0] + parameters[1] * centered + parameters[2] * centered**2
        ) * bin_width
        axis.plot(
            fine_x,
            background,
            color="C2",
            linestyle="--",
            linewidth=1.5,
            label="Fitted background",
        )

        save_fit_results(output.with_name("cms_upsilon_fit_results.csv"), parameters, covariance)

    axis.set_xlabel(r"$m_{\mu\mu}$ [GeV]", fontsize=15)
    axis.set_ylabel(f"Events / {bin_width * 1000:.0f} MeV", fontsize=15)
    axis.set_title(r"$\Upsilon\rightarrow\mu^+\mu^-$ peaks", fontsize=17, pad=10)
    axis.set_xlim(*MASS_RANGE)
    axis.set_ylim(bottom=0.0)
    axis.tick_params(axis="both", labelsize=13)
    y_max = axis.get_ylim()[1]
    annotation_positions = ((9.68, 0.88), (9.90, 0.65), (10.48, 0.61))
    for mass, label, (text_x, text_y_fraction) in zip(
        PDG_MASSES_GEV, STATE_LABELS, annotation_positions
    ):
        peak_bin = int(np.argmin(np.abs(centers - mass)))
        axis.annotate(
            label.replace("Upsilon", r"$\Upsilon$"),
            xy=(mass, counts[peak_bin]),
            xytext=(text_x, text_y_fraction * y_max),
            fontsize=11,
            ha="center",
            arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.9},
        )

    axis.legend(loc="upper left", frameon=False, fontsize=11)
    axis.text(
        0.98,
        0.96,
        "CMS Open Data",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=13,
    )
    axis.text(
        0.98,
        0.90,
        sample_label,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    axis.grid(alpha=0.18)

    figure.savefig(output, dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA,
        help="CMS Ymumu CSV or reduced NanoAOD ROOT file",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output PNG path")
    parser.add_argument("--bins", type=int, default=135, help="number of bins from 8.5 to 11.2 GeV")
    parser.add_argument("--no-fit", action="store_true", help="draw the data only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if input_path == DEFAULT_DATA.resolve():
        download_default_data(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() == ".csv":
        masses = load_csv_masses(input_path)
        sample_label = "2011 DoubleMu (education sample)"
    elif input_path.suffix.lower() == ".root":
        masses = load_root_masses(input_path)
        sample_label = "2012 DoubleMuParked Open Data"
    else:
        raise ValueError("Input must be a CMS .csv or reduced NanoAOD .root file")

    print(f"Loaded {len(masses):,} opposite-sign dimuon candidates")
    plot_spectrum(
        masses,
        output_path,
        bins=args.bins,
        do_fit=not args.no_fit,
        sample_label=sample_label,
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.pdf')}")
    if not args.no_fit:
        print(f"Wrote {output_path.with_name('cms_upsilon_fit_results.csv')}")


if __name__ == "__main__":
    main()
