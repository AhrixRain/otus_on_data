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

## Commands

From the repository root:

```powershell
python scripts_joint/run_a.py --device cuda --dry-run
python scripts_joint/run_a.py --device cuda --smoke
python scripts_joint/run_a.py --device cuda
```

Dry-run and smoke mode default to 1,000 source events per region and write to
`Run_A_dryrun` / `Run_A_smoke`, so they cannot overwrite the full `Run_A`.
Resume the full run with:

```powershell
python scripts_joint/run_a.py --device cuda --resume
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
python scripts_joint/run_b.py --device cuda --dry-run
python scripts_joint/run_b.py --device cuda --smoke
python scripts_joint/run_b.py --device cuda
```

These commands write to `Run_B_dryrun`, `Run_B_smoke`, and `Run_B`; they do not
reuse or overwrite Run A. Resume only the full Run B directory with:

```powershell
python scripts_joint/run_b.py --device cuda --resume
```

## Run C: deeper balanced-selection profile

Run C retains the dedicated Z dimuon prior and uses RMS-balanced checkpoint
selection, six 512-unit hidden layers, 20% longer stages, and a smaller CUDA
batch:

```powershell
python scripts_joint/run_c.py --device cuda --dry-run
python scripts_joint/run_c.py --device cuda --smoke
python scripts_joint/run_c.py --device cuda
```

The full run writes to `outputs/cms_Joint/Run_C`. Resume it with:

```powershell
python scripts_joint/run_c.py --device cuda --resume
```

The 24,576-event batch and four steps per epoch keep the per-epoch event count
near Run B while leaving additional CUDA headroom for the two extra layers.

## Run C full-scale: all selected training events

The full-scale profile removes the J/psi and Z split caps and cycles through
every row in each 80% training partition without replacement before repeating
that pool. The 10% validation and 10% test partitions remain untouched:

```powershell
python scripts_joint/run_c_fullscale.py --device cuda --dry-run
python scripts_joint/run_c_fullscale.py --device cuda --smoke
python scripts_joint/run_c_fullscale.py --device cuda
```

Outputs are isolated under `outputs/cms_Joint/Run_C_fullScale`. Resume with:

```powershell
python scripts_joint/run_c_fullscale.py --device cuda --resume
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

## Windows OpenMP error

If Python reports `OMP: Error #15` for `libiomp5md.dll`, first restart the
notebook kernel or terminal and ensure it uses the same Python environment as
Run A. `OMP_EXCEPTION=1` / `OPM_EXCEPTION=1` are not recognized fixes.

As a temporary opt-in workaround, set Intel's duplicate-runtime escape hatch
**before Python starts**:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python scripts_joint/run_a.py --device cuda
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
