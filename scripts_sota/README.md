# scripts_sota — Run E/F and Run G0 components

This directory is the Run E implementation area.  It deliberately does **not**
modify `scripts/` so the Run D provenance pipeline remains untouched.

## Tier A checklist

| Item | Module | Status |
|---|---|---|
| 1. Physics ground-cost Sinkhorn / Monge gap | `ot.py` | implemented |
| 2. Learned / max-SW slicing | `max_swd.py` | implemented |
| 3. C2ST + fixed-z stochasticity + coverage | `evaluation.py` | implemented |
| 4. Cylindrical residual-flow model | `cylindrical_flow.py` | implemented |
| 5. Mass-aware checkpoint gates | `selection.py` | implemented |
| Run E loss wrapper | `sota_loss.py` | implemented |
| Run E training loop / CLI | `trainer.py`, `run_e.py` | implemented |
| Run G0 reciprocal bridge | `reciprocal_bridge.py` | prototype implemented |
| Run G0 locked evaluation | `g0_contract.py`, `run_g0.py` | implemented |

## Quick checks

```bash
# Component unit tests
python -m unittest tests.test_sota

# Dry-run config resolution + data/model/loss construction (no training)
python scripts_sota/run_e.py \
  --config configs_sota/cms_Jpsi_ptj5_runE_tierA.yaml \
  --device auto --num-samples 10000 --dry-run

# Tiny one-epoch smoke (synthetic-like small cache; use only for code checks)
python scripts_sota/run_e.py \
  --config configs_sota/cms_Jpsi_ptj5_runE_tierA.yaml \
  --device auto --num-samples 10000 --smoke --run-name run_e_smoke

# Run G0 configuration/model construction only
python scripts_sota/run_g.py \
  --config configs_sota/cms_Jpsi_ptj5_runG0_reciprocal_bridge.yaml \
  --device auto --num-samples 10000 --dry-run --run-name runG0_dryrun

# Run G0 code-path smoke (three tiny epochs; not a physics result)
python scripts_sota/run_g.py \
  --config configs_sota/cms_Jpsi_ptj5_runG0_reciprocal_bridge.yaml \
  --device auto --num-samples 10000 --smoke --run-name runG0_smoke

# Full stable Run G0 (the config enforces an 8 GiB CUDA allocator ceiling)
python scripts_sota/run_g.py \
  --config configs_sota/cms_Jpsi_ptj5_runG0_reciprocal_bridge.yaml \
  --device cuda --run-name runG0_stable_full_20260821

# Locked equal-count Run E / Run F baseline comparison
python scripts_sota/run_g0.py \
  --config configs_sota/cms_Jpsi_ptj5_runG0_reciprocal_bridge.yaml \
  --checkpoint runE=outputs/cms_Jpsi_sota/runE_full_20260816_213140/best_model.pt \
  --checkpoint runF=outputs/cms_Jpsi_sota/runF_full/best_model.pt \
  --device auto
```

## Design notes

- The Sinkhorn/Monge-gap ground metric uses 14 standardized cylindrical +
  pair-physics features.  `log m(mumu)` is boosted 4x inside
  `PhysicsGroundCost`; change `loss.sota.ground_feature_weights` to alter it.
- Max-SW directions are optimized by one adversarial ascent step inside each
  distribution-loss call.  The model optimizer never sees the direction
  parameters; direction updates are detached from the generative model.
- The coverage diagnostics are explicitly labeled proxies: CMS data has no
  paired truth `(z, x)`, so `pseudo_pair_coverage` uses nearest-neighbour
  matching and `cycle_coverage` uses `x` as truth for `D(E(x))`.
- `sota_selection_gates` make the checkpoint score `inf` when any mass gate
  fails.  This prevents another epoch-270-style selection that looks good on
  coordinate losses but misses the resonance shape.
- Run G0 writes `g0_split_manifest.json` before training. Its SHA-256 covers
  every split array plus the complete data-selection/splitting contract.
- Run G0 uses stored-energy mass in theory z-space and the stable on-shell
  mass in detector x-space for both validation and final evaluation.
- The nonlinear C2ST and cycle reports remain marginal/proxy diagnostics.
  They do not establish a calibrated detector response without hidden pairs.
