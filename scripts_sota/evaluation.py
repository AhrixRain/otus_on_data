"""Modern generative-model diagnostics for Run E (Tier A, item 3).

Implements the three evaluations missing from the current ``scripts/eval.py``:

  * ``c2st_score`` -- classifier two-sample test with a small logistic
    classifier.  AUC near 0.5 is the modern "the two samples are
    indistinguishable" criterion; AUC near 1 means the generator is easily
    detected.
  * ``fixed_z_stochasticity`` -- decode the same z many times and report the
    per-event spread (Virat's diagnostic, made JSON-first).
  * ``cycle_coverage`` / ``pseudo_pair_coverage`` -- rank-based coverage
    diagnostics.  True paired (z, x) samples do not exist for this dataset, so
    cycle coverage uses ``x`` as truth for ``D(E(x))`` and pseudo-pair
    coverage uses a nearest-neighbour match in standardized cylindrical
    features.  Both are clearly labeled as proxies, not detector calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from physics import invariant_mass_np, validate_daughter_masses


def _pt_eta_phi_mass(p4: np.ndarray, daughter_masses) -> dict[str, np.ndarray]:
    p1, p2 = p4[:, 0:4], p4[:, 4:8]
    pt1 = np.hypot(p1[:, 0], p1[:, 1])
    pt2 = np.hypot(p2[:, 0], p2[:, 1])
    eta1 = np.arcsinh(p1[:, 2] / np.maximum(pt1, 1e-8))
    eta2 = np.arcsinh(p2[:, 2] / np.maximum(pt2, 1e-8))
    phi1 = np.arctan2(p1[:, 1], p1[:, 0])
    phi2 = np.arctan2(p2[:, 1], p2[:, 0])
    mass = invariant_mass_np(p4, daughter_masses=daughter_masses, stable=True)
    pair_px = p1[:, 0] + p2[:, 0]
    pair_py = p1[:, 1] + p2[:, 1]
    pair_pt = np.hypot(pair_px, pair_py)
    return {
        "mass": mass,
        "pt_minus": pt1,
        "pt_plus": pt2,
        "eta_minus": eta1,
        "eta_plus": eta2,
        "phi_minus": phi1,
        "phi_plus": phi2,
        "pair_pt": pair_pt,
    }


def cylindrical_features_np(
    p4: np.ndarray,
    daughter_masses,
    *,
    stable_mass: bool = True,
) -> np.ndarray:
    """14D cylindrical + pair physics features for C2ST and matching."""
    p1, p2 = p4[:, 0:4], p4[:, 4:8]
    pt1 = np.hypot(p1[:, 0], p1[:, 1])
    pt2 = np.hypot(p2[:, 0], p2[:, 1])
    eta1 = np.arcsinh(p1[:, 2] / np.maximum(pt1, 1e-8))
    eta2 = np.arcsinh(p2[:, 2] / np.maximum(pt2, 1e-8))
    phi1 = np.arctan2(p1[:, 1], p1[:, 0])
    phi2 = np.arctan2(p2[:, 1], p2[:, 0])
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    mass = invariant_mass_np(
        p4, daughter_masses=daughter_masses, stable=bool(stable_mass)
    )
    pair_px = p1[:, 0] + p2[:, 0]
    pair_py = p1[:, 1] + p2[:, 1]
    pair_pt = np.hypot(pair_px, pair_py)
    pair_e = p1[:, 3] + p2[:, 3]
    pair_pz = p1[:, 2] + p2[:, 2]
    rapidity = 0.5 * np.log(
        np.maximum(pair_e + pair_pz, 1e-8) / np.maximum(pair_e - pair_pz, 1e-8)
    )
    return np.stack(
        [
            np.log(np.maximum(pt1, 1e-8)),
            eta1,
            np.sin(phi1),
            np.cos(phi1),
            np.log(np.maximum(pt2, 1e-8)),
            eta2,
            np.sin(phi2),
            np.cos(phi2),
            np.log(np.maximum(mass, 1e-8)),
            np.log(np.maximum(pair_pt, 1e-8)),
            rapidity,
            np.cos(dphi),
            np.sin(dphi),
            eta1 - eta2,
        ],
        axis=1,
    ).astype(np.float32)


def _standardize_features(train: np.ndarray, apply_to: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (apply_to - mean) / std, mean, std


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    y = y_true[order]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)[y == 1]
    return float((ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _train_logistic(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    weights = torch.zeros(x.shape[1], 1, dtype=torch.float32, device=device)
    bias = torch.zeros(1, dtype=torch.float32, device=device)
    weights.requires_grad_(True)
    bias.requires_grad_(True)
    optimizer = torch.optim.Adam([weights, bias], lr=lr, weight_decay=weight_decay)
    for _ in range(iterations):
        optimizer.zero_grad()
        logits = xt @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), yt)
        loss.backward()
        optimizer.step()
    return torch.cat([weights.detach().squeeze(1), bias.detach()]).cpu().numpy()


def _logistic_score(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x @ theta[:-1] + theta[-1])))


def _mlp_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    iterations: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Train a small nonlinear C2ST without adding a scikit-learn dependency."""
    torch.manual_seed(seed)
    network = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-4)
    x_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    batch_size = min(1024, len(train_x))
    network.train()
    for _ in range(max(1, int(iterations))):
        indices = torch.randint(0, len(train_x), (batch_size,), device=device)
        logits = network(x_tensor[indices]).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.inference_mode():
        values = torch.as_tensor(test_x, dtype=torch.float32, device=device)
        return torch.sigmoid(network(values).squeeze(1)).cpu().numpy()


def c2st_score(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    daughter_masses: Sequence[float] | None = None,
    max_samples: int = 20000,
    folds: int = 5,
    seed: int = 12345,
    device: torch.device | None = None,
    classifier: str = "linear",
    mlp_iterations: int = 250,
    stable_mass: bool = True,
) -> dict[str, float]:
    """Classifier two-sample test on 14D cylindrical physics features."""
    masses = validate_daughter_masses(daughter_masses)
    real = np.asarray(real, dtype=np.float32)
    fake = np.asarray(fake, dtype=np.float32)
    n = min(len(real), len(fake), int(max_samples))
    if n < 100:
        raise ValueError("c2st_score needs at least 100 events per sample")
    rng = np.random.default_rng(seed)
    idx_real = rng.choice(len(real), size=n, replace=False)
    idx_fake = rng.choice(len(fake), size=n, replace=False)
    features = np.concatenate(
        [
            cylindrical_features_np(real[idx_real], masses, stable_mass=stable_mass),
            cylindrical_features_np(fake[idx_fake], masses, stable_mass=stable_mass),
        ],
        axis=0,
    )
    labels = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    aucs = []
    fold_idx = np.array_split(rng.permutation(2 * n), folds)
    for held in fold_idx:
        mask = np.ones(2 * n, dtype=bool)
        mask[held] = False
        train_x, train_mean, train_std = _standardize_features(features[mask], features[mask])
        if len(np.unique(labels[mask])) < 2:
            continue
        test_x = (features[held] - train_mean) / train_std
        if classifier == "linear":
            theta = _train_logistic(train_x, labels[mask], seed=seed, device=device)
            scores = _logistic_score(test_x, theta)
        elif classifier == "mlp":
            scores = _mlp_score(
                train_x,
                labels[mask],
                test_x,
                iterations=mlp_iterations,
                seed=seed + len(aucs),
                device=device,
            )
        else:
            raise ValueError(f"Unknown C2ST classifier {classifier!r}; expected linear or mlp")
        aucs.append(_auc(labels[held], scores))
    if not aucs:
        raise RuntimeError("C2ST folds were degenerate")
    auc_mean = float(np.mean(aucs))
    return {
        "c2st_auc_mean": auc_mean,
        "c2st_auc_std": float(np.std(aucs)),
        "detectability_auc": float(max(auc_mean, 1.0 - auc_mean)),
        "samples_per_class": int(n),
        "folds": int(len(aucs)),
        "classifier": classifier,
        "interpretation": "0.5 indistinguishable; >0.6 detectable; >0.8 easy to detect",
    }


def c2st_suite(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    daughter_masses: Sequence[float] | None = None,
    max_samples: int = 10000,
    folds: int = 5,
    seed: int = 12345,
    device: torch.device | None = None,
    mlp_iterations: int = 250,
    stable_mass: bool = True,
) -> dict[str, dict[str, float]]:
    """Linear and nonlinear classifier two-sample tests on equal samples."""
    return {
        classifier: c2st_score(
            real,
            fake,
            daughter_masses=daughter_masses,
            max_samples=max_samples,
            folds=folds,
            seed=seed,
            device=device,
            classifier=classifier,
            mlp_iterations=mlp_iterations,
            stable_mass=stable_mass,
        )
        for classifier in ("linear", "mlp")
    }


@torch.inference_mode()
def _decode_repeated(
    model,
    z: np.ndarray,
    draws: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return [K, N, 8] decoder draws for one fixed z batch."""
    model.eval()
    out = []
    for _ in range(draws):
        decoded = []
        for start in range(0, len(z), batch_size):
            batch = torch.as_tensor(z[start : start + batch_size], dtype=torch.float32, device=device)
            decoded.append(model.decode(batch).detach().cpu().numpy())
        out.append(np.concatenate(decoded, axis=0))
    return np.stack(out, axis=0)


@torch.inference_mode()
def _encode_once(
    model,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        out.append(model.encode(batch).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _rank_uniformity(ranks: np.ndarray) -> dict[str, float]:
    # ranks are fractions of draws below the truth value.
    hist, _ = np.histogram(ranks, bins=10, range=(0.0, 1.0))
    expected = len(ranks) / 10.0
    return {
        "rank_mean": float(ranks.mean()),
        "rank_std": float(ranks.std()),
        "chi2_uniform": float(np.sum((hist - expected) ** 2 / expected) / 9.0),
    }


def _coverage_report(
    truth_observables: dict[str, np.ndarray],
    draw_observables: dict[str, np.ndarray],
    phi_keys: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Central coverage + rank uniformity for repeated draws.

    ``draw_observables[obs]`` has shape [K, N].  For phi keys the residual is
    wrapped relative to the first draw before quantiles are computed.
    """
    phi_keys = phi_keys or set()
    report: dict[str, dict[str, float]] = {}
    for key in truth_observables:
        truth = truth_observables[key]
        draws = draw_observables[key]
        if key in phi_keys:
            reference = draws[0:1]
            draws = np.arctan2(np.sin(draws - reference), np.cos(draws - reference))
        low = np.quantile(draws, 0.025, axis=0)
        high = np.quantile(draws, 0.975, axis=0)
        low50 = np.quantile(draws, 0.25, axis=0)
        high50 = np.quantile(draws, 0.75, axis=0)
        below = (draws < truth[None, :]).mean(axis=0)
        report[key] = {
            "central_95_coverage": float(np.mean((truth >= low) & (truth <= high))),
            "central_50_coverage": float(np.mean((truth >= low50) & (truth <= high50))),
            "mean_abs_rank_error": float(np.mean(np.abs(below - 0.5))),
            **_rank_uniformity(below),
        }
    return report


def _observable_stack(p4_draws: np.ndarray, daughter_masses) -> dict[str, np.ndarray]:
    """p4_draws has shape [K, N, 8]; returns [K, N] for each observable."""
    k, n, _ = p4_draws.shape
    out = {}
    for start in range(0, k, 64):
        chunk = p4_draws[start : start + 64].reshape(-1, 8)
        obs = _pt_eta_phi_mass(chunk, daughter_masses)
        for key, value in obs.items():
            out.setdefault(key, []).append(value.reshape(start + 64 if start + 64 < k else k - start, n))
    return {key: np.concatenate(value, axis=0) for key, value in out.items()}


def fixed_z_stochasticity(
    model,
    z: np.ndarray,
    *,
    daughter_masses,
    n_events: int = 256,
    draws: int = 32,
    batch_size: int = 256,
    device: torch.device | None = None,
    seed: int = 777,
) -> dict:
    """Virat-style fixed-z stochasticity report."""
    device = device or torch.device("cpu")
    masses = validate_daughter_masses(daughter_masses)
    rng = np.random.default_rng(seed)
    n = min(int(n_events), len(z))
    idx = rng.choice(len(z), size=n, replace=False)
    fixed = np.asarray(z[idx], dtype=np.float32)
    repeated = _decode_repeated(model, fixed, int(draws), int(batch_size), device)
    obs = _observable_stack(repeated, masses)
    per_event_width = {key: np.std(value, axis=0) for key, value in obs.items()}
    phi_width = {
        key: np.std(
            np.arctan2(
                np.sin(obs[key] - obs[key][0:1]),
                np.cos(obs[key] - obs[key][0:1]),
            ),
            axis=0,
        )
        for key in ("phi_minus", "phi_plus")
    }
    report = {
        "fixed_events": int(n),
        "draws_per_event": int(draws),
        "median_per_event_std": {
            key: float(np.median(width)) for key, width in per_event_width.items()
        },
        "mean_per_event_std": {
            key: float(np.mean(width)) for key, width in per_event_width.items()
        },
        "median_per_event_phi_std_rad": {
            key: float(np.median(width)) for key, width in phi_width.items()
        },
        "mass_fraction_below_1e-5_gev": float(np.mean(per_event_width["mass"] < 1e-5)),
    }
    if report["mass_fraction_below_1e-5_gev"] > 0.9:
        report["warning"] = "decoder stochasticity appears collapsed"
    return report


def cycle_coverage(
    model,
    x: np.ndarray,
    *,
    daughter_masses,
    n_events: int = 512,
    draws: int = 32,
    batch_size: int = 256,
    device: torch.device | None = None,
    seed: int = 2026,
) -> dict:
    """Rank coverage of D(E(x)) around fixed x (paired proxy for the cycle)."""
    device = device or torch.device("cpu")
    masses = validate_daughter_masses(daughter_masses)
    rng = np.random.default_rng(seed)
    n = min(int(n_events), len(x))
    idx = rng.choice(len(x), size=n, replace=False)
    fixed = np.asarray(x[idx], dtype=np.float32)
    z = _encode_once(model, fixed, int(batch_size), device)
    repeated = _decode_repeated(model, z, int(draws), int(batch_size), device)
    truth_obs = _pt_eta_phi_mass(fixed, masses)
    draw_obs = _observable_stack(repeated, masses)
    return {
        "proxy": "cycle_x_to_z_to_x",
        "events": int(n),
        "draws_per_event": int(draws),
        "coverage": _coverage_report(truth_obs, draw_obs, phi_keys={"phi_minus", "phi_plus"}),
    }


def pseudo_pair_coverage(
    model,
    z: np.ndarray,
    x: np.ndarray,
    *,
    daughter_masses,
    n_events: int = 512,
    draws: int = 32,
    batch_size: int = 256,
    device: torch.device | None = None,
    seed: int = 2027,
) -> dict:
    """Nearest-neighbour pseudo-pair coverage for D(z) against x.

    True detector-level pairs are unavailable, so each fixed z is matched to
    its nearest CMS event in standardized cylindrical features.  The resulting
    coverage numbers are a useful diagnostic but are *not* detector
    calibration.
    """
    device = device or torch.device("cpu")
    masses = validate_daughter_masses(daughter_masses)
    rng = np.random.default_rng(seed)
    n = min(int(n_events), len(z), len(x))
    z_idx = rng.choice(len(z), size=n, replace=False)
    x_idx = rng.choice(len(x), size=n, replace=False)
    z_fixed = np.asarray(z[z_idx], dtype=np.float32)
    x_candidate = np.asarray(x[x_idx], dtype=np.float32)
    zf = cylindrical_features_np(z_fixed, masses)
    xf = cylindrical_features_np(x_candidate, masses)
    zf = (zf - zf.mean(axis=0)) / np.maximum(zf.std(axis=0), 1e-8)
    xf = (xf - xf.mean(axis=0)) / np.maximum(xf.std(axis=0), 1e-8)
    distances = np.sum((zf[:, None, :] - xf[None, :, :]) ** 2, axis=2)
    match = x_candidate[np.argmin(distances, axis=1)]
    repeated = _decode_repeated(model, z_fixed, int(draws), int(batch_size), device)
    truth_obs = _pt_eta_phi_mass(match, masses)
    draw_obs = _observable_stack(repeated, masses)
    return {
        "proxy": "nearest_neighbour_pseudo_pairs",
        "events": int(n),
        "draws_per_event": int(draws),
        "coverage": _coverage_report(truth_obs, draw_obs, phi_keys={"phi_minus", "phi_plus"}),
    }


def write_evaluation_report(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
