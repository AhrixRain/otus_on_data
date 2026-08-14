# CMS DoubleElectron OTUS on Apple Silicon MPS

This workflow runs the CMS DoubleElectron OTUS training without CUDA-specific paths or install flags. It uses `--device auto`, which selects CUDA if available, otherwise MPS if available, otherwise CPU. On a Mac mini M4 with a current standard PyTorch macOS wheel, it should select `mps`.

## Environment

The environment is managed with a project-local virtual environment (`.venv`)
instead of conda. The single `cms` environment serves the CMS workflows on
this machine.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cms-mps.txt
python -m ipykernel install --user --name cms --display-name "Python (cms)"
```

Activate with `source .venv/bin/activate`; alternatively run commands directly
with `.venv/bin/python` without activation.

The config expects these input files:

```text
experiments/cms_zpeak/data/Run2012B_DoubleElectron.root
experiments/cms_doubleelectron/cms_dyee_mg5_8tev_dy1j_ptj5_fiducial_70_110.hdf5
```

Raw data, ROOT files, HDF5 files, checkpoints, and generated outputs are not committed by this workflow.

## Smoke Test

```bash
python scripts/train.py --config configs/cms_doubleelectron_mps.yaml --device auto --num-samples 10000 --epochs 1 --run-name smoke_mps
python scripts/eval.py --config configs/cms_doubleelectron_mps.yaml --checkpoint outputs/cms_doubleelectron/smoke_mps/best_model.pt --device auto --num-samples 10000
```

For a faster code-path check, train only the first stage with fewer SWD slices:

```bash
python scripts/train.py --config configs/cms_doubleelectron_mps.yaml --device auto --num-samples 10000 --smoke-test --run-name smoke_mps_fast
python scripts/eval.py --config configs/cms_doubleelectron_mps.yaml --checkpoint outputs/cms_doubleelectron/smoke_mps_fast/best_model.pt --device auto --num-samples 10000
```

Each training run writes to:

```text
outputs/cms_doubleelectron/<run_id>/
```

Training outputs:

```text
config.resolved.json
train_log.csv
status.json
history.json
best_model.pt
last_model.pt
```

Evaluation outputs are written to `<run_id>/eval/` by default:

```text
config.resolved.json
metrics.json
mass_ratio.png
residual.png
mass_histograms.npz
```

Plot the training loss curve from `train_log.csv`:

```bash
python scripts/plot_loss.py --run-dir outputs/cms_doubleelectron/<run_name>
python scripts/plot_loss.py --run-dir outputs/cms_doubleelectron/<run_name> --components
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` is not set by default. Set it only if PyTorch reports a specific unsupported MPS operation and you accept CPU fallback for that operation:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --config configs/cms_doubleelectron_mps.yaml --device auto --num-samples 10000 --smoke-test --run-name smoke_mps_fallback
```

## Full Run

The full config runs all enabled stages from `configs/cms_doubleelectron_mps.yaml`:

```text
stage1_anchor_warmup: 50 epochs
stage2_joint_transport: 50 epochs
stage3_decoder_response_mass_protected: 50 epochs
stage4_encoder_distribution_polish: 50 epochs
stage5_z_cycle_inverse_polish_gentle: 40 epochs
```

## Data Cache

`train.py`, `eval.py`, and `plot.py` share a keyed on-disk cache of the
selected/split CMS and MG5 arrays at `<output_root>/.plot_cache/`. The key
covers the resolved config, selection cuts, data-file fingerprints, seed, and
`--num-samples`, so repeated runs (including ablations with different
`--num-samples` values) skip the multi-GB ROOT scan. Regenerating a cached
entry requires a source file to change (mtime/size) or a config/seed/sample-cap
change. `plot.py` can bypass the cache with `--no-data-cache`.

Run full training with a unique run name:

```bash
RUN_NAME=full_mps_$(date +%Y%m%d_%H%M%S)
python scripts/train.py --config configs/cms_doubleelectron_mps.yaml --device auto --run-name "$RUN_NAME"
```

For a detached full run that keeps the Mac awake and writes a readable log:

```bash
RUN_NAME=full_mps_$(date +%Y%m%d_%H%M%S)
RUN_DIR=outputs/cms_doubleelectron/$RUN_NAME
mkdir -p "$RUN_DIR"
nohup caffeinate -dims env PYTHONUNBUFFERED=1 python scripts/train.py \
  --config configs/cms_doubleelectron_mps.yaml \
  --device auto \
  --run-name "$RUN_NAME" \
  --progress auto \
  > "$RUN_DIR/full_run.log" 2>&1 &
```

Evaluate the best checkpoint from that run:

```bash
python scripts/eval.py --config configs/cms_doubleelectron_mps.yaml --checkpoint "outputs/cms_doubleelectron/${RUN_NAME}/best_model.pt" --device auto
```

To check progress without watching continuously:

```bash
RUN_DIR=outputs/cms_doubleelectron/<run_name>
cat "$RUN_DIR/status.json"
tail -n 40 "$RUN_DIR/full_run.log"
```

For a nicer live local Terminal view, use `watch` if it is installed:

```bash
watch -n 30 'cat outputs/cms_doubleelectron/<run_name>/status.json'
```

macOS may not have `watch` by default. This shell loop works without extra packages:

```bash
while true; do clear; date; cat outputs/cms_doubleelectron/<run_name>/status.json; sleep 30; done
```

Use a new `--run-name` for each run. The training script refuses to overwrite an existing directory that already contains `best_model.pt` or `last_model.pt`.

For an intermediate-length test before the full run, use a small per-stage epoch override:

```bash
python scripts/train.py --config configs/cms_doubleelectron_mps.yaml --device auto --num-samples 100000 --epochs 3 --run-name medium_mps
python scripts/eval.py --config configs/cms_doubleelectron_mps.yaml --checkpoint outputs/cms_doubleelectron/medium_mps/best_model.pt --device auto --num-samples 100000
```

## J/psi -> mu mu runs (OTUS-on-CMS)

The J/psi dimuon workflow uses the same CLI with `configs/cms_JpsiDoubleMuons_mps.yaml`
(run: `Jpsi_v3.5`) or the controlled mass-ablation config
`configs/cms_JpsiDoubleMuons_Jpsi_v3.6A_no_explicit_mass.yaml`:

```bash
.venv/bin/python scripts/train.py \
  --config configs/cms_JpsiDoubleMuons_mps.yaml \
  --run-name Jpsi_v3.5_stage_diag --device auto
.venv/bin/python scripts/train.py \
  --config configs/cms_JpsiDoubleMuons_Jpsi_v3.6A_no_explicit_mass.yaml \
  --run-name Jpsi_v3.6A_no_explicit_mass --device auto
```

### v3.8 vanilla SWAE (raw-coordinate latent SWD, all data)

`configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml` runs the pure two-term
SWAE objective `L = beta * L_reco + lambda * L_SW` with the latent
sliced Wasserstein on raw physical coordinates, all selected CMS+MG5 events
under an 80/10/10 split, and deterministic epoch-wise loaders without
replacement. See `docs/Jpsi_v3.8_vanilla_paper_runbook.md` for the full
definition, preflight/dry-run/smoke commands, and the production command
(which must only be started with explicit authorization).

Training now also writes a named checkpoint at every stage boundary
(`checkpoint_stage1_anchor_warmup.pt`, `checkpoint_stage2_joint_transport.pt`,
`checkpoint_stage3_decoder_response_mass_protected.pt`) in addition to
`best_model.pt` / `last_model.pt`. This is diagnostic-only bookkeeping; the
training schedule, loss, and data are unchanged.

### Stage / ablation diagnostic

Compare checkpoints with identical eval/plot settings (e.g. end of Stage 2 vs
the best Stage-3 checkpoint, or v3.5 vs v3.6A):

```bash
.venv/bin/python scripts/stage_diagnostic.py \
  --config configs/cms_JpsiDoubleMuons_mps.yaml \
  --checkpoint stage2_end=outputs/cms_JpsiDoubleMuons/Jpsi_v3.5/checkpoint_stage2_joint_transport.pt \
  --checkpoint best_stage3=outputs/cms_JpsiDoubleMuons/Jpsi_v3.5/best_model.pt \
  --output-dir outputs/cms_JpsiDoubleMuons/Jpsi_v3.5/stage_diagnostic \
  --num-samples 200000 --split test
```

The script reruns the shared `scripts/eval.py` + `scripts/plot.py` pipelines for
each checkpoint with identical arguments and writes
`stage_diagnostic/summary.json` (machine-readable) and `stage_diagnostic/summary.md`
(human-readable, with explicit Stage-3 / mass-ablation verdicts). Add
`--skip-plots` to reuse existing per-checkpoint outputs, or `--force` to rerun.

### Explicit invariant-mass supervision in the J/psi loss

The J/psi loss (`cms_jpsi_doublemuon_loss` in `scripts/loss.py`) contains these
explicit mass terms, all independently disableable:

- `mass_w1` / `pair_mass_w1`: W1 on the standardized invariant mass (identical
  value; two weight knobs).
- `resonance_mass_w1`: W1 on physical mass inside
  `[resonance_mass_center +/- resonance_mass_half_width]` (the narrow J/psi window).
- `mass_kin_swd_components.mll`: the invariant-mass column of the joint
  `[m_ll, pT_ll, y_ll, cos dphi, sin dphi]` sliced-Wasserstein term. Set `mll: 0.0`
  to keep only the four non-mass pair-kinematics columns.

Indirect mass-bearing terms that remain active in v3.6A by design:
`physics_swd` (15-observable joint SWD incl. `m_ll`), `physics_coord_swd`
(per-lepton log-pT/eta/direction/log-E), `x_reco_physics_w1` (per-event paired
physics MSE incl. `m_ll`), and the raw 8-vector terms.
