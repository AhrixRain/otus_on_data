# scripts/ layout

Root `scripts/` contains only the active CMS OTUS pipeline and its CLI tools.
Everything historical has been moved to `scripts/legacy/`.

## Core modules (root)

- `cms_data.py` — ROOT/HDF5 loading, selection, splitting, and shared data cache.
- `cms_model.py` — model construction and checkpoint I/O.
- `cms_training.py` — loaders, stage scheduler, training/evaluation loops, history logger.
- `loss.py` — dual-space OT/SWD loss factories and physics features.
- `physics.py` — invariant-mass and daughter-mass helpers.
- `metrics.py` — histogram, KS/W1, residual and ratio metrics.
- `device_utils.py` — CUDA/MPS/CPU device selection.
- `encoder_diagnostics.py` — training-time encoder alignment callback.

## CLI / tools (root)

- `train.py`, `eval.py`, `plot.py`, `preflight.py`
- `plot_loss.py`, `eval_verdict.py`, `sample_generator.py`

## Subdirectories

- `prior_build/` — MG5 LHE -> prior HDF5 tooling:
  `lhe_to_prior_hdf5.py`, `smear_prior.py`, `reweight_prior.py`, `mix_prior.py`.
- `diagnostics/` — current data/prior/model health checks:
  `diagnose_jpsi_new_density.py`, `prior_kinematics.py`, `data_kinematics.py`,
  `jpsi_mismatch_check.py`, `jpsi_f32_check.py`, `zee_health_check.py`,
  `jpsi_new_prior_comparison.py`.
- `legacy/` — frozen diagnostics for the archived v3.x J/psi investigations.
  They are retained for reproducibility of archived outputs, not used by the
  active pipeline. Run them from the repo root with the `scripts` directory on
  `PYTHONPATH`; each script adds that path itself when possible.
- `mg5_cards/` — committed MG5 process/run cards used by `prior_build/`.
