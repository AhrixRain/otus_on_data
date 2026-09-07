# cms_Joint

This directory contains the isolated joint-resonance training path. **Run A**
trains one shared stochastic dimuon encoder/decoder on J/psi and Z regions.
The regions never share events or an OT endpoint plan: one J/psi loss and one Z
loss are back-propagated into the same parameters with equal region weight.

Run A deliberately:

- excludes an explicit parent-mass feature and all resonance labels from model
  conditioning (the physical muon kinematics still carry mass information);
- computes condition normalization with equal J/psi and Z contributions;
- selects checkpoints on the worse normalized region/direction metric;
- uses only the J/psi signal window, not the broad sideband/background sample;
- keeps Upsilon completely unopened as the locked zero-shot test.

The broad J/psi support should be introduced only in a later named ablation,
with signal and sideband components sampled separately. Mixing the broad CMS
background into Run A would test background modelling at the same time as
cross-scale transfer and make an Upsilon failure ambiguous.

## Directory layout

| Path | Purpose |
|---|---|
| `run_joint.py` | The single training entry point. `--run <ID>` resolves `configs_joint/cms_Joint_<ID>.yaml` (`--run E`, `--run C_fullScale`, `--run abNarrow`); `--config <path>` still works. |
| `joint_model.py` | Shared encoder/decoder with the auditable condition mask. |
| `joint_trainer.py` | Stage schedule, validation, gated checkpoint selection, resume. |
| `joint_data.py`, `joint_metrics.py` | Region loading / split contract, and the gate metrics. |
| `make_plots.py` | Loss curves, per-region diagnostics, paper-style density plots for one run. |
| `dashboard.py` | Live training dashboard for one run directory. |
| `upsilon_transfer_test.py` | Post-training held-out Upsilon transfer test. |
| `upsilon/` | The Upsilon decode/compare/evaluate scripts these drive. |
| `build_ab_priors.py` | Builds the event-matched prior pair for the width A/B. |
| `identity_reference.py` | NumPy-only identity-map and finite-sample-floor references, plus the `vs_identity` gauge. Imported by `joint_metrics.py`; no torch, so finished runs can be re-scored offline. |
| `identity_baseline.py` | Re-scores a completed run against those references without the model or a GPU. `--run-dir outputs/cms_Joint/<run>`. |
| `paired_closure.py` | Per-event closure against withheld truth pairs (the upstream ppzee/ppttbar datasets). The only test that separates unfolding from marginal matching. |
| `paired_data.py` | Loader for the paired benchmarks: reads paired, splits UNPAIRED for training, and records the withheld pairing in the split manifest for scoring only. |
| `score_paired_closure.py` | Drives the whole per-event score for a finished run: rebuilds the withheld pairs, runs the frozen encoder, writes the report. `--run-dir outputs/cms_Joint/<run>`. |
| `paired_leakage_audit.py` | Read-only audit of a paired run for pairing leakage: the three checks a `residual_rms_vs_identity` below 1.0 has to survive. |
| `_retired_launchers/` | The old per-run `run_a.py` .. `run_f.py` shims. Superseded by `--run`; safe to delete. |

The per-run launchers used to be seven near-identical files that differed only
in a config path. They are kept in `_retired_launchers/` only until you confirm
nothing external calls them.

## Reading a gate number (added 2026-09-04)

No raw gate metric means anything on its own. On the smeared arm of the
prior-width A/B the map `z~ = x` -- no model at all -- scores latent mass
KS 0.0359, W1 0.00219 and width relative error 0.071 against strict targets of
0.04, 0.003 and 0.1, so it passes every one. Every validation therefore now
also records, per direction:

| key | meaning |
|---|---|
| `latent_identity_mass_ks` | what the metric reads if the map is the identity |
| `latent_floor_mass_ks` | what two independent draws of the target distribution read at the same sample size |
| `latent_mass_ks_vs_identity` | `(model - floor) / (identity - floor)`: 1.0 = no better than the input, 0.0 = at the floor, > 1 = worse than doing nothing |
| `latent_mass_ks_headroom` | `(identity - floor) / floor`: how much separation the comparison can resolve at all |

`*_vs_identity` is lower-is-better and non-negative, so it can be added to any
region's `selection.gates` / `selection.targets` with no code change.
`cms_Joint_abNarrowSplit.yaml` gates on it at 0.5. Measured reference points
for that threshold: the Z region of the completed A/B reaches 0.001-0.03, the
smeared J/psi arm sits at 0.19-1.72 and the narrow J/psi arm never falls below
1.0.

There is deliberately no cycle gauge: the identity cycle is `x -> x`, whose
metrics are identically zero, so cycle gates have no power against a no-op.

Re-score a finished run with:

```powershell
python scripts_joint/identity_baseline.py --run-dir outputs/cms_Joint/AB_narrow
```

## Paired closure (added 2026-09-04)

`paired_closure.py` scores a map against the truth partner of each event, not
against a histogram:

    residual_rms_vs_identity = rms(mass(pred) - mass(z_true))
                             / rms(mass(x_input) - mass(z_true))

The denominator is the detector resolution, so < 1 means resolution was
actually inverted. Scoring the published upstream OTUS result on its own paired
ppzee benchmark gives 3.33 -- worse per event than handing back the detector
event -- while its cycle `D(E(x))` reproduces `x` to 1.03 GeV rms. Reproduce
with:

```powershell
python scripts_joint/paired_closure.py --results-npz "experiments/ppzee/otus_results-dataset=ppzee_test.npz"
```

## Scoring our own model per event (added 2026-09-07)

The loader branch that step 3 was waiting on now exists.
`configs_joint/cms_Joint_ppzee.yaml` is a single-region **bench, not a result**
built on Run E's model and losses:

```powershell
python scripts_joint/run_joint.py --run ppzee --device cuda --dry-run
python scripts_joint/run_joint.py --run ppzee --device cuda
python scripts_joint/score_paired_closure.py --run-dir outputs/cms_Joint/ppzee --device cuda
python scripts_joint/paired_leakage_audit.py --run-dir outputs/cms_Joint/ppzee
```

**The pairing must never reach training.** `paired_data.py` draws `x` and `z`
through independent permutations of the source file, asserts that the six
arrays it returns are the only thing the trainer receives, and measures the
per-column correlation between `x_train` and `z_train` against what the same
statistic reads on the unshuffled source -- a guard that cannot fire proves
nothing. The row indices go into `joint_split_manifest.json` under
`regions.<name>.pair_index` and are read for the first time by
`score_paired_closure.py`, after training is over.

Read the result against three numbers, never one:

| | meaning |
|---|---|
| **1.0** | handing back the detector-level event unchanged |
| **3.33** | the upstream OTUS encoder on this same 160,000-event benchmark |
| `displacement_vs_resolution` | how far the map moved from its own input, in units of the detector resolution. Our encoder is a residual flow initialised near the identity, so it scores ~1.0 when it has learned nothing (artifact-measured: 1.044 after two smoke epochs). Without this number a ratio near 1.0 cannot be read. |

Two dataset facts that must be stated whenever the ppzee number is quoted
(artifact-measured 2026-09-07, `outputs/ppzee_layout.json`): the truth has
**identically zero pair pT** (`rho(px1, px2) = -1.0000` exactly), so all of the
reco recoil was manufactured downstream; and `ROL[:, 8:12]` is the Delphes MET,
excluded from `x` because `z` carries no MET partner to score it against.

`data/ppttbar*.hdf5` is **not** a free replication: it is `[N, 24]`, six
four-vectors, while this entire pipeline is `[N, 8]` two-body. See the joint
README note in `memory.md` section 11, Session 30.

### The `--smoke` limitation

Any config that gates a `*_vs_identity` metric cannot complete `--smoke` at the
default 1000 events. The smoke's 100-event validation pool puts the
finite-sample floor at or above the identity value, `gauge()` returns `+inf` by
design, and the trainer's `score < stage_best_score` then never fires, so the
stage saves no checkpoint and the run dies with `Stage ... produced no
validation checkpoint`. This is pre-existing and not specific to ppzee:
artifact-measured 2026-09-07, `--run abNarrowSplit --smoke` fails identically.
Use `--smoke --num-samples 20000`.

## Prior-width A/B (diagnostic)

Runs D and E both selected a **stage-1 deterministic** checkpoint as best,
against a prior that had been hand-smeared to 26.2 MeV -- next to the 28.1 MeV
CMS width. The encoder therefore only had to remove ~7% of the mass width.
Run F used the 14.7 MeV showered prior and failed the J/psi latent mass gate at
all 51 of its validations. Those runs differ in generator, shower, reweighting,
event count and width simultaneously, so the width hypothesis is untested.

`build_ab_priors.py` isolates it. Both arms come from the same source events;
each arm then has the training selection applied on its own terms, and the
smearing amplitude is calibrated against the width measured *after* selection.
The builder prints a controlled-comparison table and aborts if the muon-pT,
pair-pT or |eta| medians move by more than 2% between arms, so mass width is
demonstrably the only variable.

The arms are deliberately not event-matched: requiring an event to pass the
120 MeV window both before and after smearing discards the tail events that
carry the width, which pulled the narrow arm down to 12.8 MeV and capped the
smeared arm at 23.2 MeV in the first attempt.

```powershell
python scripts_joint/build_ab_priors.py --calibrate-only   # inspect the fit
python scripts_joint/build_ab_priors.py                    # write both arms

python scripts_joint/run_joint.py --run abNarrow  --device cuda --dry-run
python scripts_joint/run_joint.py --run abNarrow  --device cuda
python scripts_joint/run_joint.py --run abSmeared --device cuda
```

Read the result from `joint_selection.regions.jpsi.gates.latent_mass_ks` in
each run's `history.json`. If the smeared arm passes and the narrow arm does
not, the Run D/E mass results were carried by the pre-smeared prior and the
latent-width failure is the real finding, not a tuning problem.

## Commands

From the repository root:

```powershell
python scripts_joint/run_joint.py --run A --device cuda --dry-run
python scripts_joint/run_joint.py --run A --device cuda --smoke
python scripts_joint/run_joint.py --run A --device cuda
```

Dry-run and smoke mode default to 1,000 source events per region and write to
`Run_A_dryrun` / `Run_A_smoke`, so they cannot overwrite the full `Run_A`.
Resume the full run with:

```powershell
python scripts_joint/run_joint.py --run A --device cuda --resume
```

The launcher refuses to start a fresh training job in a directory that already
contains a history or best checkpoint. Use `--resume` or a new `--run-name`;
this prevents accidental mixing of two Run A histories.

The full Run A profile is sized for the 12 GiB RTX 4080 Laptop GPU: PyTorch is
capped at 11 GiB, with 36,864 events in each sequential regional training
batch. The three large updates sample 110,592 events per region per epoch,
close to the original Run A event budget. Every CUDA epoch records allocated
and reserved peak memory in `history.json` and prints the reserved peak at
validation. If another GPU-heavy application reduces available memory, close
it or temporarily use `--batch-size 32768 --steps-per-epoch 3`.

Outputs live under `outputs/cms_Joint/<run-name>/` and include the resolved
configuration, data/cache provenance, a content-addressed split manifest,
history, stage-best checkpoints, the accepted `best_model.pt`, and the final
per-region bidirectional evaluation. The run never reads an Upsilon sample.

The generic entry point is `run_joint.py`; use it for future Run B/C configs.

## Run B: dedicated Z dimuon prior

Run B is the controlled prior replacement for Run A. It inherits every Run A
setting and points the Z region at the dedicated one-million-event dimuon MG5
sample. Launch it independently with:

```powershell
python scripts_joint/run_joint.py --run B --device cuda --dry-run
python scripts_joint/run_joint.py --run B --device cuda --smoke
python scripts_joint/run_joint.py --run B --device cuda
```

These commands write to `Run_B_dryrun`, `Run_B_smoke`, and `Run_B`; they do not
reuse or overwrite Run A. Resume only the full Run B directory with:

```powershell
python scripts_joint/run_joint.py --run B --device cuda --resume
```

## Run C: deeper balanced-selection profile

Run C retains the dedicated Z dimuon prior and uses RMS-balanced checkpoint
selection, six 512-unit hidden layers, 20% longer stages, and a smaller CUDA
batch:

```powershell
python scripts_joint/run_joint.py --run C --device cuda --dry-run
python scripts_joint/run_joint.py --run C --device cuda --smoke
python scripts_joint/run_joint.py --run C --device cuda
```

The full run writes to `outputs/cms_Joint/Run_C`. Resume it with:

```powershell
python scripts_joint/run_joint.py --run C --device cuda --resume
```

The 24,576-event batch and four steps per epoch keep the per-epoch event count
near Run B while leaving additional CUDA headroom for the two extra layers.

## Run C full-scale: all selected training events

The full-scale profile removes the J/psi and Z split caps and cycles through
every row in each 80% training partition without replacement before repeating
that pool. The 10% validation and 10% test partitions remain untouched:

```powershell
python scripts_joint/run_joint.py --run C_fullScale --device cuda --dry-run
python scripts_joint/run_joint.py --run C_fullScale --device cuda --smoke
python scripts_joint/run_joint.py --run C_fullScale --device cuda
```

Outputs are isolated under `outputs/cms_Joint/Run_C_fullScale`. Resume with:

```powershell
python scripts_joint/run_joint.py --run C_fullScale --device cuda --resume
```

After both full runs have a `history.json`, generate one total/per-region loss
figure for each run with:

```powershell
python scripts_joint/plot_joint_losses.py
```

The default figures are `outputs/cms_Joint/Run_A/loss_curve.png` and
`outputs/cms_Joint/Run_B/loss_curve.png`. Use `--smooth 1` for unsmoothed
curves, `--linear-y` for a linear loss axis, or pass alternative run
directories with `--run-a` and `--run-b`.

## Run D: showered inclusive Z prior

Run D keeps the complete full-scale Run C training contract and replaces only
the Z prior with the one-million-event MG5+Pythia8 CKKW-L inclusive 0/1-parton
sample:

```powershell
python scripts_joint/run_joint.py --run D --device cuda --dry-run
python scripts_joint/run_joint.py --run D --device cuda --smoke
python scripts_joint/run_joint.py --run D --device cuda
```

The full run writes to `outputs/cms_Joint/Run_D`. Resume it with:

```powershell
python scripts_joint/run_joint.py --run D --device cuda --resume
```

Run D never loads Upsilon during training, validation, or checkpoint
selection. Since the original Run C Upsilon first look has already occurred,
describe a later Run D Upsilon result as post-unblinding held-out transfer.

## Run E: one full training pass per epoch

Run E inherits Run D and changes only the meaning of an epoch. Instead of four
fixed-size updates, each epoch visits every selected training row exactly once
in all four independent partitions (J/psi x/z and Z x/z):

```powershell
python scripts_joint/run_joint.py --run E --device cuda --dry-run
python scripts_joint/run_joint.py --run E --device cuda --smoke
python scripts_joint/run_joint.py --run E --device cuda
```

The number of optimizer updates is derived from the largest training
partition. Shorter partitions use balanced smaller batches, so no event is
repeated merely to fill the epoch. The inherited 432-stage-epoch schedule is
therefore much more training than Run D's four-update epochs; use an explicit
`--epochs` override only for a named duration ablation.

## Windows OpenMP error

If Python reports `OMP: Error #15` for `libiomp5md.dll`, first restart the
notebook kernel or terminal and ensure it uses the same Python environment as
Run A. `OMP_EXCEPTION=1` / `OPM_EXCEPTION=1` are not recognized fixes.

As a temporary opt-in workaround, set Intel's duplicate-runtime escape hatch
**before Python starts**:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python scripts_joint/run_joint.py --run A --device cuda
Remove-Item Env:KMP_DUPLICATE_LIB_OK
```

For a notebook, restart the kernel and make this the first cell, before NumPy,
PyTorch, SciPy, ROOT, or plotting imports:

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

This suppresses the abort; it does not remove the duplicate runtime and Intel
warns that execution may be unsafe. Run A records this variable in
`provenance.json`. Prefer fixing the mixed environment before a final physics
run. `OMP_NUM_THREADS=1` can limit CPU contention but does not fix Error #15.
