"""Paired-benchmark region loader -- read paired, train unpaired, score paired.

Why this module exists
----------------------
Every metric the joint pipeline reports is a MARGINAL, and the 2026-09-04 A/B
proved a map that does nothing can satisfy marginals (memory.md section 7.2).
The only test that separates

    "the encoder inverted the detector response event by event"

from

    "the encoder produced four-vectors whose histogram happens to match"

is a comparison against each event's own withheld truth partner. The upstream
OTUS benchmarks ``data/ppzee*.hdf5`` and ``data/ppttbar*.hdf5`` carry that
pairing; the CMS open data never can. ``scripts_joint/paired_closure.py`` is
the scorer; this module is the loader that feeds it.

THE PRIME DIRECTIVE: the pairing must never reach training
----------------------------------------------------------
If the pairing leaks into training, ``residual_rms_vs_identity`` stops meaning
anything and starts looking *excellent*, which is the failure mode that would
waste the most time. Three mechanisms enforce it, in increasing order of how
hard they are to fool:

1. ``x`` and ``z`` are drawn through **independent permutations** of the source
   file, seeded ``seed`` and ``seed + 1``, exactly as ``cms_data.split_unpaired``
   does for the CMS regions. Row ``i`` of ``x_train`` is not the partner of row
   ``i`` of ``z_train``.
2. The six split arrays handed back are asserted to be exactly
   ``_SPLIT_KEYS``. The row indices live in a separate return value that goes
   into the split manifest and is never passed to the trainer.
3. A measured guard, not a structural argument: the per-column Pearson
   correlation between ``x_train`` and ``z_train`` must be near zero, AND the
   same statistic on the *unshuffled source* must be large. A guard that
   cannot fire proves nothing (CLAUDE.md section 4), so the positive control
   runs on every load and its value is recorded.

File layout, artifact-measured 2026-09-07
-----------------------------------------
``FDL`` and ``ROL`` are flat top-level DATASETS in these files, not groups, and
the files carry no root attributes at all -- there is no recorded convention
metadata, so every convention below was established numerically or read out of
``dataGenerationCode/ppzeeDataGenCode/FinalData/FinalData_ppzee.py``.

* ``FDL`` [N, 8] -- MadGraph final-state ``e-`` then ``e+``, each
  ``(px, py, pz, E)``. This is ``z``.
* ``ROL`` [N, 12] -- Delphes ``e-``, ``e+``, then **MET** as a massless
  four-vector built from its stored (pT, eta, phi). ``x`` is columns 0:8;
  columns 8:12 are the MET and are dropped, because ``z`` has no MET partner
  to score them against. source-verified: ``utilityFunctions/configs.py``
  records ``'ppzee': {'z_dim': 8, 'x_dim': 12}  # x_dim = 12 includes MET,
  this is removed in experiments``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cms_data import file_fingerprint, resolve_path


# Channels routed here instead of through the CMS ROOT + theory-prior path.
PAIRED_CHANNELS = frozenset({"ppzee", "ppttbar", "paired_benchmark"})

SPLIT_KEYS = ("x_train", "x_val", "x_test", "z_train", "z_val", "z_test")

# Bumped whenever the selection/splitting semantics below change, so a manifest
# written by older code cannot be mistaken for one written by this code.
PAIRED_PIPELINE_VERSION = 1

# Guard thresholds. Both are asserted on every load; see the prime directive.
#
# MAX_LEAKED_CORRELATION is the floor of the leakage threshold. It cannot be
# the whole threshold, because the sample correlation of two genuinely
# independent draws of size n has standard deviation about 1/sqrt(n - 3): at
# n = 100 a chance |rho| of 0.26 across eight columns is unremarkable, and a
# flat 0.05 would reject an honest split. The threshold below is therefore the
# larger of the floor and a LEAK_SIGMA-sigma band, which keeps the guard's
# power where it matters -- a leaked pairing reads |rho| ~ 0.98, which is 9.7
# sigma even at n = 100 and 340 sigma at the bench's n = 120,000.
MAX_LEAKED_CORRELATION = 0.05
LEAK_SIGMA = 6.0
MIN_SOURCE_CORRELATION = 0.5


def leakage_threshold(n: int) -> float:
    """Largest |rho| between x and z that independent shuffling can produce."""
    if n <= 4:
        return 1.0
    return max(MAX_LEAKED_CORRELATION, LEAK_SIGMA / np.sqrt(float(n - 3)))


def normalize_channel(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def is_paired_channel(value: Any) -> bool:
    """True for channels served by this module rather than by ``cms_data``."""
    return normalize_channel(value) in PAIRED_CHANNELS


def _column_slice(spec: Any, default: tuple[int, int]) -> slice:
    if spec is None:
        start, stop = default
    else:
        values = list(spec)
        if len(values) != 2:
            raise ValueError(f"column spec must be [start, stop], got {spec!r}")
        start, stop = int(values[0]), int(values[1])
    if stop - start != 8:
        raise ValueError(
            f"paired regions need exactly 8 columns, got [{start}:{stop}]"
        )
    return slice(start, stop)


def load_paired_source(
    path: Path,
    *,
    z_dataset: str = "FDL",
    x_dataset: str = "ROL",
    z_columns: Any = None,
    x_columns: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one paired benchmark file as row-aligned ``(z, x)`` in float64.

    The returned arrays ARE partners row by row. Nothing downstream of
    ``load_paired_region`` ever sees them in this order.
    """
    import h5py

    z_slice = _column_slice(z_columns, (0, 8))
    x_slice = _column_slice(x_columns, (0, 8))
    with h5py.File(path, "r") as handle:
        for name in (z_dataset, x_dataset):
            if name not in handle:
                raise KeyError(
                    f"{path} has no dataset {name!r}; available: "
                    f"{', '.join(sorted(handle.keys()))}"
                )
        z_raw = handle[z_dataset]
        x_raw = handle[x_dataset]
        for name, obj in ((z_dataset, z_raw), (x_dataset, x_raw)):
            # These files store FDL/ROL as flat datasets, but the unified CMS
            # priors store FDL as a GROUP holding zData. Fail loudly rather
            # than silently mis-slicing one for the other.
            if not isinstance(obj, h5py.Dataset):
                raise TypeError(
                    f"{path}:{name} is a {type(obj).__name__}, expected a "
                    "dataset. Paired benchmark files store FDL/ROL flat."
                )
        z_data = np.asarray(z_raw[:], dtype=np.float64)[:, z_slice]
        x_data = np.asarray(x_raw[:], dtype=np.float64)[:, x_slice]
    if len(z_data) != len(x_data):
        raise ValueError(
            f"{path}: {z_dataset} has {len(z_data)} rows and {x_dataset} has "
            f"{len(x_data)}; a paired file must have one x per z"
        )
    if len(z_data) < 3:
        raise ValueError(f"{path}: only {len(z_data)} paired rows")
    for label, arr in (("z", z_data), ("x", x_data)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{path}: {label} contains non-finite values")
    return z_data, x_data


def _per_column_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson rho per column over the common prefix; 0.0 for constant columns."""
    n = min(len(left), len(right))
    a = np.asarray(left[:n], dtype=np.float64)
    b = np.asarray(right[:n], dtype=np.float64)
    out = np.zeros(a.shape[1], dtype=np.float64)
    for index in range(a.shape[1]):
        u, v = a[:, index], b[:, index]
        if u.std() == 0.0 or v.std() == 0.0:
            continue
        out[index] = float(np.corrcoef(u, v)[0, 1])
    return out


def check_unpaired(
    x_values: np.ndarray,
    z_values: np.ndarray,
    *,
    source_corr: np.ndarray | None = None,
) -> tuple[float, float]:
    """Raise unless ``x_values`` and ``z_values`` are event-wise independent.

    Returns ``(max_abs_rho, threshold)``. Exposed as a public function so the
    check can be tested directly against a deliberately row-aligned pair,
    which is the only way to show that the guard actually fires.
    """
    corr = _per_column_correlation(x_values, z_values)
    max_leak = float(np.max(np.abs(corr)))
    n = min(len(x_values), len(z_values))
    threshold = leakage_threshold(n)
    if max_leak > threshold:
        detail = ""
        if source_corr is not None:
            detail = (
                " The unshuffled source correlates at "
                f"{float(np.max(np.abs(source_corr))):.4f}, so this check has power."
            )
        raise AssertionError(
            f"x and z correlate at |rho| = {max_leak:.4f} > {threshold:.4f} over "
            f"{n} rows. The pairing has leaked into training.{detail}"
        )
    return max_leak, threshold


def _fit_split_sizes(total: int, split_config: dict[str, Any]) -> dict[str, int]:
    """Ratio-then-cap train/val sizing, matching ``cms_data.split_unpaired``.

    Only train and val are drawn from the fit file. The test split comes from
    a separate generation (``paired_test_file``) and is taken whole, capped by
    ``test_max``, rather than as a ratio of the fit file -- applying the
    train/val ratios to an already-held-out file would silently discard 90% of
    it and make the result incomparable with the upstream number, which is
    quoted on all 160,000 rows of ``ppzee_test.hdf5``.
    """
    ratios = {
        "train": int(total * float(split_config["train_ratio"])),
        "val": int(total * float(split_config["val_ratio"])),
    }
    sizes: dict[str, int] = {}
    for key in ("train", "val"):
        cap = split_config.get(f"{key}_max")
        size = ratios[key] if cap is None else min(ratios[key], int(cap))
        if size < 1:
            raise ValueError(
                f"paired split {key!r} would have {size} rows from {total} source rows"
            )
        sizes[key] = size
    if sizes["train"] + sizes["val"] > total:
        raise ValueError(
            f"train + val ({sizes['train'] + sizes['val']}) exceeds the "
            f"{total} rows in the fit file"
        )
    return sizes


def paired_cache_metadata(config: dict[str, Any], num_samples: int | None) -> dict[str, Any]:
    """Provenance for the manifest -- everything that can change the arrays.

    Deliberately parallel to ``cms_data.data_cache_metadata`` but built here so
    that adding a paired channel cannot perturb the CMS metadata, the CMS cache
    key, or ``cms_data._pipeline_semantic_fingerprint`` (which hashes the whole
    of ``cms_data.py``, so editing that file would invalidate every existing
    CMS region cache).
    """
    paths = config["paths"]
    metadata: dict[str, Any] = {
        "paired_pipeline_version": PAIRED_PIPELINE_VERSION,
        "num_samples": None if num_samples is None else int(num_samples),
        "float_type": config.get("float_type", "float32"),
        "seed": int(config.get("seed", 0)),
        "data_split": config["data_split"],
        "data": config.get("data", {}),
        "paired_train_file": file_fingerprint(paths["paired_train_file"]),
        "paired_test_file": file_fingerprint(paths["paired_test_file"]),
    }
    return metadata


def _index_digest(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.int64).tobytes()
    ).hexdigest()


def load_paired_region(
    config: dict[str, Any],
    *,
    num_samples: int | None = None,
    log=print,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Load one paired-benchmark region as six UNPAIRED split arrays.

    Returns ``(arrays, info, pair_index)``.

    ``arrays`` is exactly the six ``SPLIT_KEYS``, each [n, 8] in ``float_type``
    -- the same contract the CMS muon path emits, so the trainer, the model and
    every metric are unchanged.

    ``pair_index[key]`` is the source-file row index of each row of
    ``arrays[key]``. It is the ONLY object that can reconstruct the pairing and
    it goes into the split manifest, never into training. Train and validation
    indices refer to ``paths.paired_train_file``; test indices refer to
    ``paths.paired_test_file``.
    """
    paths = config["paths"]
    data_config = config.get("data", {})
    split_config = config["data_split"]
    seed = int(config.get("seed", 0))
    dtype = np.dtype(config.get("float_type", "float32")).name

    reader = {
        "z_dataset": str(data_config.get("z_dataset", "FDL")),
        "x_dataset": str(data_config.get("x_dataset", "ROL")),
        "z_columns": data_config.get("z_columns"),
        "x_columns": data_config.get("x_columns"),
    }
    train_path = Path(paths["paired_train_file"])
    test_path = Path(paths["paired_test_file"])
    if train_path.resolve() == test_path.resolve():
        raise ValueError(
            "paired_train_file and paired_test_file are the same file; the "
            "held-out claim would be vacuous"
        )

    z_fit, x_fit = load_paired_source(train_path, **reader)
    z_held, x_held = load_paired_source(test_path, **reader)
    log(f"paired source {train_path.name}: {len(z_fit)} rows (train/val)")
    log(f"paired source {test_path.name}: {len(z_held)} rows (test)")

    # The positive control for the leakage guard below: on the source files the
    # rows ARE partners, so this must be large. If it is not, the guard has no
    # power and the whole measurement is void.
    source_corr = _per_column_correlation(z_fit, x_fit)
    if float(np.max(np.abs(source_corr))) < MIN_SOURCE_CORRELATION:
        raise ValueError(
            f"{train_path}: FDL and ROL[0:8] do not look like event-by-event "
            f"partners (max |rho| = {np.max(np.abs(source_corr)):.4f} < "
            f"{MIN_SOURCE_CORRELATION}). Nothing downstream would be valid."
        )
    log(
        "paired source per-column rho (z vs x, unshuffled): "
        + ", ".join(f"{value:.4f}" for value in source_corr)
    )

    if num_samples is not None and int(num_samples) > 0:
        cap = int(num_samples)
        z_fit, x_fit = z_fit[:cap], x_fit[:cap]
        z_held, x_held = z_held[:cap], x_held[:cap]

    fit_sizes = _fit_split_sizes(len(z_fit), split_config)
    test_cap = split_config.get("test_max")
    n_held = len(z_held) if test_cap is None else min(len(z_held), int(test_cap))
    if n_held < 1:
        raise ValueError(f"paired test split would have {n_held} rows")
    held_sizes = {"test": n_held}

    # Four independent permutations. x and z are shuffled apart, and the fit
    # and held-out files are shuffled apart, so no two of the six arrays are
    # row-aligned with each other. Seeds mirror cms_data.split_unpaired's
    # seed / seed + 1 convention.
    perm_x_fit = np.random.default_rng(seed).permutation(len(x_fit))
    perm_z_fit = np.random.default_rng(seed + 1).permutation(len(z_fit))
    perm_x_held = np.random.default_rng(seed + 2).permutation(len(x_held))
    perm_z_held = np.random.default_rng(seed + 3).permutation(len(z_held))

    n_train, n_val, n_test = fit_sizes["train"], fit_sizes["val"], held_sizes["test"]
    pair_index = {
        "x_train": perm_x_fit[:n_train],
        "x_val": perm_x_fit[n_train : n_train + n_val],
        "x_test": perm_x_held[:n_test],
        "z_train": perm_z_fit[:n_train],
        "z_val": perm_z_fit[n_train : n_train + n_val],
        "z_test": perm_z_held[:n_test],
    }
    source_arrays = {
        "x_train": x_fit, "x_val": x_fit, "x_test": x_held,
        "z_train": z_fit, "z_val": z_fit, "z_test": z_held,
    }
    arrays = {
        key: source_arrays[key][pair_index[key]].astype(dtype, copy=False)
        for key in SPLIT_KEYS
    }

    # --- guard 2: nothing but the six split arrays reaches the caller -------
    if set(arrays) != set(SPLIT_KEYS):
        raise AssertionError(f"paired loader emitted unexpected keys: {sorted(arrays)}")
    for key, value in arrays.items():
        if value.ndim != 2 or value.shape[1] != 8:
            raise AssertionError(f"{key} has shape {value.shape}, expected (n, 8)")
        if value.dtype != np.dtype(dtype):
            raise AssertionError(f"{key} dtype {value.dtype} != {dtype}")
        if not np.isfinite(value).all():
            raise AssertionError(f"{key} contains non-finite values")

    # --- guard 1: the permutations are genuinely different ------------------
    alignment = {}
    for split in ("train", "val", "test"):
        x_idx, z_idx = pair_index[f"x_{split}"], pair_index[f"z_{split}"]
        n = min(len(x_idx), len(z_idx))
        coincidences = int(np.count_nonzero(x_idx[:n] == z_idx[:n]))
        alignment[split] = coincidences
        # Two independent permutations of n elements agree in Poisson(1) many
        # positions. 50 is astronomically improbable and catches an accidental
        # "both sides got the same permutation".
        if coincidences > 50:
            raise AssertionError(
                f"{split}: x and z share {coincidences} aligned row indices out "
                f"of {n}; the permutations are not independent"
            )

    # --- guard 3: the measured leakage check --------------------------------
    max_leak, threshold = check_unpaired(
        arrays["x_train"], arrays["z_train"], source_corr=source_corr
    )
    log(
        f"paired leakage guard: max |rho|(x_train, z_train) = {max_leak:.4f} "
        f"<= {threshold:.4f} (source {np.max(np.abs(source_corr)):.4f}); "
        f"aligned-index coincidences {alignment}"
    )

    info = {
        "enabled": False,
        "hit": False,
        "status": "not_cached",
        "reason": (
            "paired benchmark HDF5 files load and split in about a second; no "
            "on-disk cache layer is used for them"
        ),
        "paired_train_file": str(train_path),
        "paired_test_file": str(test_path),
        "leakage_guard": {
            "max_abs_corr_x_train_z_train": max_leak,
            "max_abs_corr_source": float(np.max(np.abs(source_corr))),
            "source_corr_per_column": source_corr.tolist(),
            "aligned_index_coincidences": alignment,
            "threshold": threshold,
        },
    }
    pair_index_block = {
        "purpose": (
            "Source-file row index of each split row. This is the withheld "
            "pairing: it is recorded here for scoring and is never passed to "
            "the trainer. See scripts_joint/paired_data.py."
        ),
        "paired_pipeline_version": PAIRED_PIPELINE_VERSION,
        "seed": seed,
        "sources": {
            "train": str(train_path), "val": str(train_path), "test": str(test_path),
        },
        "reader": reader,
        "splits": {
            key: {
                "count": int(len(pair_index[key])),
                "sha256": _index_digest(pair_index[key]),
                "index": pair_index[key].astype(np.int64).tolist(),
            }
            for key in SPLIT_KEYS
        },
    }
    log(
        "paired splits: "
        + ", ".join(f"{key}={arrays[key].shape}" for key in SPLIT_KEYS)
    )
    return arrays, info, pair_index_block


def materialize_pairs(
    manifest_path: Path,
    region: str,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the withheld ``(z, x)`` partners for one split, for scoring only.

    Reads the pair index out of a written ``joint_split_manifest.json`` and
    re-reads the source file. Returns ``(z, x)`` row-aligned in the order of
    ``arrays[f"x_{split}"]`` -- i.e. ``z[i]`` is the truth partner of the event
    the model saw as ``x_{split}[i]``.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    block = manifest["regions"][region]["pair_index"]
    source = Path(block["sources"][split])
    z_all, x_all = load_paired_source(source, **block["reader"])
    index = np.asarray(block["splits"][f"x_{split}"]["index"], dtype=np.int64)
    if _index_digest(index) != block["splits"][f"x_{split}"]["sha256"]:
        raise ValueError(f"{manifest_path}: pair index for x_{split} failed its digest")
    return z_all[index], x_all[index]


def resolve_paired_region_paths(
    region_paths: dict[str, Any],
    data_root: Path,
    region_name: str,
) -> None:
    """Resolve the two source paths in place; raise if either is missing."""
    for key in ("paired_train_file", "paired_test_file"):
        if not region_paths.get(key):
            raise KeyError(f"Paired region {region_name!r} has no {key}")
        region_paths[key] = str(resolve_path(region_paths[key], data_root))
