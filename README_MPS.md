# CMS OTUS on Apple Silicon (MPS)

This workflow runs the CMS OTUS training without CUDA-specific paths or
install flags. It uses `--device auto`, which selects CUDA if available,
otherwise MPS if available, otherwise CPU.

## Interpreter

On the current machine use the `cms` conda environment:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python
```

On machines that use the project-local `.venv` instead, replace that path with
`.venv/bin/python`; all commands below are otherwise identical.

## Repository layout

See `scripts/README.md` and `configs/README.md`. The active pipeline is kept
at the top level of `scripts/`; prior-building tools live in
`scripts/prior_build/`, current health checks in `scripts/diagnostics/`, and
frozen v3.x diagnostics in `scripts/legacy/`.

## Input files

```text
data/Run2012BC_DoubleMuParked_Muons.root            # J/psi CMS data
data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5         # rebuilt prior (ptj=5)
data/cms_jpsi_mumu_mg5_8tev_mixed.hdf5              # prior provenance (ptj=10)
data/Run2012B_DoubleElectron.root                   # Z->ee CMS data
data/cms_dyee_mg5_8tev_dy1j_ptj5_fiducial_70_110.hdf5
```

Config status:

- `configs/cms_Jpsi_newprior_paper_20pct.yaml` documents the already-completed
  ptj=10 paper-objective run.
- `configs/cms_Jpsi_newprior_ptj5_3term_cosine_20pct.yaml` is the completed
  three-term cosine baseline (ptj=5 prior).
- `configs/cms_Jpsi_newprior_ptj5_rund_pairswd_cyclemass_20pct.yaml` is the
  Run D candidate: three-term cosine plus pair-level SWD and a cycle
  relative-mass Huber term.

Raw data, HDF5 files, checkpoints, caches, and generated outputs are not
committed by this workflow.

## Smoke tests

```bash
# J/psi pipeline smoke test (next-round three-term objective + ptj=5 prior)
python scripts/train.py \
  --config configs/cms_Jpsi_newprior_ptj5_3term_cosine_20pct.yaml \
  --device auto --num-samples 10000 --epochs 1 --run-name smoke_jpsi_3term

# Archived DoubleElectron pipeline
python scripts/train.py \
  --config configs/archive/cms_doubleelectron_mps.yaml \
  --device auto --num-samples 10000 --epochs 1 --run-name smoke_ee
```

Each training run writes to:

```text
outputs/<output_root>/<run_id>/
```

with `config.resolved.json`, `train_log.csv`, `status.json`, `history.json`,
`best_model.pt`, and `last_model.pt`.

## Shared data cache

`train.py`, `eval.py`, and `plot.py` share a keyed on-disk cache at
`<output_root>/.plot_cache/`. The cache key covers the resolved config,
selection cuts, input-file fingerprints, seed, and `--num-samples`, so
repeated runs skip the multi-GB ROOT scan. Use `--no-data-cache` in `plot.py`
to bypass it.

Run a fail-fast preflight without starting training:

```bash
python scripts/preflight.py \
  --config configs/cms_Jpsi_newprior_paper_20pct.yaml --device auto
```

## Evaluate / plot / inspect

```bash
python scripts/eval.py \
  --config configs/cms_Jpsi_newprior_paper_20pct.yaml \
  --checkpoint outputs/cms_Jpsi_new/<run>/best_model.pt \
  --device auto --num-samples 100000

python scripts/plot.py \
  --config configs/cms_Jpsi_newprior_paper_20pct.yaml \
  --checkpoint outputs/cms_Jpsi_new/<run>/best_model.pt \
  --device auto

python scripts/plot_loss.py --run-dir outputs/cms_Jpsi_new/<run> --components
python scripts/eval_verdict.py --eval-dir outputs/cms_Jpsi_new/<run>/eval
```

## Prior rebuild

The committed MG5 cards live in `scripts/mg5_cards/` (`ptj = 5.0` for the
current J/psi prior), and the post-processing chain is in
`scripts/prior_build/`. See `docs/jpsi_prior_rebuild_runbook.md` for the full
procedure and acceptance targets.

## Detached full run

Use a unique `--run-name`; the training script refuses to overwrite an
existing directory that already contains `best_model.pt` or `last_model.pt`.

```bash
RUN_NAME=full_mps_$(date +%Y%m%d_%H%M%S)
nohup caffeinate -dims env PYTHONUNBUFFERED=1 \
  python scripts/train.py \
  --config configs/cms_Jpsi_newprior_paper_20pct.yaml \
  --device auto --run-name "$RUN_NAME" --progress auto \
  > "outputs/cms_Jpsi_new/${RUN_NAME}.log" 2>&1 &
```

To check progress:

```bash
RUN_DIR=outputs/cms_Jpsi_new/<run_name>
cat "$RUN_DIR/status.json"
tail -n 40 "outputs/cms_Jpsi_new/${RUN_NAME}.log"
```
