# J/psi v3.8 vanilla-paper runbook

Config: `configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml`

## Scientific definition of v3.8

v3.8 trains the pure vanilla SWAE/OTUS objective from the original OTUS paper
(arXiv:2101.08944, SWAE reference arXiv:1804.01947):

```
L = beta * L_reco + lambda * L_SW
L_reco = mean((x - D(E(x)))^2)      raw-coordinate per-event MSE
L_SW   = SWD_{p=2}(z, E(x))         raw-coordinate latent sliced Wasserstein
```

with `beta = 1.0`, `lambda = 1.0`, `p = 2` (squared projected differences),
`num_slices = 1000` random projections per training batch, Adam with
`lr = 0.001`, training batch size `20000`, no LR decay, no early stopping,
and a deterministic seed of `0`.

The reconstruction loss is the raw per-event MSE over the physical
eight-dimensional coordinates, matching the original `data_loss(x, x_tilde,
p=2)` implementation. The latent sliced-Wasserstein loss is computed on the
raw physical eight-dimensional coordinates: nothing is standardized inside
the SWD comparison (`standardize_raw_matching: false`). The model may still
standardize its internal neural-network inputs/outputs via `raw_io: true`;
that is a model-level convenience and is separate from the loss-level SWD
standardization.

Only the reconstruction MSE and the latent SWD contribute training
gradients. Every mass, physics-feature, generator-distribution, anchor,
cycle, marginal, tail, and MMD weight is zero (the vanilla mode ignores all
of them regardless).

## Exact distinction between v3.7 and v3.8

| | v3.7 vanilla | v3.8 vanilla-paper |
|---|---|---|
| latent SWD coordinates | standardized 8D (default `standardize_raw_matching: true`) | raw physical 8D (`standardize_raw_matching: false`) |
| mode flag | `vanilla_v3_7: true` | `vanilla_swae: true` (generalized; `vanilla_v3_7` retained as an alias) |
| data caps | 500k/100k/100k train/val/test | none: all selected events, 80/10/10 |
| training epoch sampler | random sampling with replacement | shuffled without replacement, deterministic per epoch |
| unequal-domain policy | not defined | cycle-and-reshuffle with equal per-batch cardinalities |
| validation/test iteration | random sampling with replacement | deterministic sequential, complete coverage |
| reconstruction checkpoint name | `best_cycle.pt` | `best_reconstruction.pt` |

Existing v3.7 configs and checkpoints keep their standardized-latent-loss
behavior. They are not reinterpreted.

## Why raw-coordinate latent SWD?

Standardizing the latent vectors before the sliced Wasserstein changes which
differences the objective rewards across the eight physical coordinates
(GeV-scale momenta/energies). v3.8 removes that rescaling so the latent SWD
is the literal `SWD_{p=2}(z, E(x))` on physical coordinates, isolating the
objective change from the data-use change.

## All-data split and loader semantics

The complete CMS ROOT input and the complete MG5 HDF5 prior are read and
selected. Each domain is split deterministically (seeded) into 80% train,
10% validation, 10% test. There is no train/val/test cap and the production
command must not pass `--num-samples`.

Training (`loaders.train_sampler: shuffled_without_replacement`):

- every epoch reshuffles each domain with a deterministic per-epoch seeded
  generator (`seed + epoch * 1000003`);
- the larger domain is visited exactly once before it would reshuffle;
- the smaller domain cycles and is deterministically reshuffled before each
  cycle, so every one of its events is visited at least once per epoch;
- the number of repeated events in the smaller domain is
  `len(larger domain) - len(smaller domain)` and is reported in
  `loader_info.train_repeated_x/z`;
- the final incomplete batch is kept (never dropped), and both domains
  receive identical batch cardinalities at every step, so every SWD
  comparison has equal batch sizes;
- the two domains stay unpaired: batch positions are not physical event
  pairs.

Validation/test (`loaders.eval_sampler: deterministic_sequential`):

- deterministic, in stored order, no random resampling with replacement;
- both complete splits are covered; the smaller split repeats from the
  beginning for the remaining aligned chunks, reported in
  `loader_info.eval_repeated_x/z`.

For the checked-in data this resolves to (see preflight):

- CMS selected candidates: 7,177,600
  (x train/val/test = 5,742,080 / 717,760 / 717,760)
- MG5 prior: 1,000,000
  (z train/val/test = 800,000 / 100,000 / 100,000)
- 288 training steps per epoch at batch 20,000;
  `train_repeated_z = 4,942,080`
- 36 validation steps at batch 20,000; `eval_repeated_z = 617,760`

## Required input paths

```text
data/Run2012BC_DoubleMuParked_Muons.root
data/cms_jpsi_mumu_mg5_8tev_1M.hdf5
```

The reduced CMS ROOT file exposes only `nMuon`, `Muon_pt`, `Muon_eta`,
`Muon_phi`, `Muon_mass`, `Muon_charge`. The muon selection is `pT > 2.0`,
`|eta| < 2.4`, opposite-sign pairs inside the `2.6-3.5` GeV J/psi window,
charge-ordered into `[mu- p4, mu+ p4]` rows. The HDF5 prior key is
`FDL/zData` (`zData` is also accepted).

## Environment setup

Apple Silicon / MPS (this machine):

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Training uses `--device auto`, which selects CUDA, then MPS, then CPU. Set
`PYTORCH_ENABLE_MPS_FALLBACK=1` only if a specific unsupported MPS operation
is reported.

CUDA / Linux:

```bash
python -m pip install -r requirements-cms.txt
python scripts/train.py --config configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml --device auto ...
```

`--device cuda` forces CUDA and fails fast if it is unavailable.

## Preflight (fail-fast data validation, no run directory)

```bash
.venv/bin/python scripts/preflight.py \
  --config configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml --device auto
```

Reports both files' existence, size, SHA-256 fingerprint, ROOT tree names,
total event count, required branches, total selected CMS candidate count,
HDF5 key/shape/dtype/finite checks, split counts, cap status, batches per
epoch, repeated-event counts, selected device, and memory estimates. It may
write a new keyed entry under `outputs/cms_JpsiDoubleMuons/.plot_cache/` but
never creates or overwrites a run directory. Optional:
`--json-output preflight.json`.

## Full-data dry run (loads and splits everything, no training)

```bash
.venv/bin/python scripts/train.py \
  --config configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml \
  --device auto --dry-run
```

Reuses the cache written by preflight and exits without writing files.

## Smoke test (one epoch, small explicit sample cap)

```bash
.venv/bin/python scripts/train.py \
  --config configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml \
  --device auto \
  --run-name Jpsi_v3.8_vanilla_paper_smoke_seed0 \
  --num-samples 5000 \
  --epochs 1 \
  --progress none
```

`--num-samples` and `--epochs` are explicit CLI overrides recorded in
`config.resolved.json`; the on-disk production config is not modified. The
smoke run establishes only that the training path works, not scientific
success.

Verify gradients and reload:

```bash
.venv/bin/python scripts/verify_v38_gradients.py \
  --config configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml \
  --device auto --num-samples 2000 \
  --checkpoint outputs/cms_JpsiDoubleMuons/Jpsi_v3.8_vanilla_paper_smoke_seed0/best_combined.pt

.venv/bin/python scripts/eval_v37.py \
  --config configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml \
  --checkpoint outputs/cms_JpsiDoubleMuons/Jpsi_v3.8_vanilla_paper_smoke_seed0 \
  --device auto --num-samples 5000
```

## Production command (NOT EXECUTED)

```bash
.venv/bin/python scripts/train.py \
  --config configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml \
  --device auto \
  --run-name Jpsi_v3.8_vanilla_paper_all_data_seed0 \
  --progress auto
```

Durable detached logging variant:

```bash
RUN_NAME=Jpsi_v3.8_vanilla_paper_all_data_seed0
RUN_DIR=outputs/cms_JpsiDoubleMuons/$RUN_NAME
mkdir -p "$RUN_DIR"
nohup caffeinate -dims env PYTHONUNBUFFERED=1 .venv/bin/python scripts/train.py \
  --config configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml \
  --device auto \
  --run-name "$RUN_NAME" \
  --progress auto \
  > "$RUN_DIR/train.log" 2>&1 &
```

The command deliberately omits `--num-samples`, `--epochs`, and
`--batch-size`, so the config's 300-epoch schedule, 20,000 batch size, and
loss settings are used verbatim.

## Expected output files

```text
outputs/cms_JpsiDoubleMuons/<run_name>/
  config.resolved.json          resolved semantics (raw SWD, loader policy)
  run_metadata.json             fingerprints, counts, git commit, device, ...
  train_log.csv                 per-log-epoch losses/components/gradients
  status.json / history.json
  last_model.pt                 latest epoch
  best_model.pt                 best checkpoint-selection score
  best_combined.pt              checkpoint selected by the configured validation objective
  best_z_prior.pt               lowest validation latent SWD
  best_reconstruction.pt        lowest validation reconstruction MSE
  checkpoint_final.pt           copy of the final checkpoint
  encoder_alignment_diagnostic/
```

Each checkpoint embeds the resolved config and run metadata (loss settings,
data paths, file fingerprints, selected/split counts, loader policy, batch
size, slices, seed, software/device, Git commit, dirty-worktree status).

## Restart and overwrite behavior

`train.py` refuses to overwrite a run directory that already contains
`best_model.pt`, `last_model.pt`, or `checkpoint_final.pt`. Use a new
`--run-name` for each run. The dry-run and preflight paths never create a
run directory.

## Checkpoint-selection semantics

The selection score is exactly `beta * L_reco,val + lambda * L_SW,val`
(`= L_reco,val + L_SW,val` here), implemented by
`loss.selection_score: {z_prior: 1.0, x_reco: 1.0, x_sim: 0.0, cycle: 0.0}`.
The direct `z -> x` generator path is computed offline as a diagnostic
(`scripts/eval_v37.py`) and never contributes a training gradient.

## Known deviations from the original OTUS paper

- channel: J/psi -> mu+mu- (CMS DoubleMuParked 2012 data), not Z -> e+e-;
- network: the current stochastic `CondNoiseAutoencoder` with four 512-unit
  hidden layers, not the paper's one-layer electron network;
- physical muon daughter masses `[0.105658, 0.105658]` GeV;
- selected CMS detector-level data plus an MG5 truth-level prior;
- one 300-epoch stage (no paper Z->ee anchor/cycle schedule);
- raw-coordinate latent SWD rather than standardization-before-SWD.

v3.8 is a J/psi/CMS experiment using the original paper's core two-term SWAE
objective. It is not a complete reproduction of the original paper.
