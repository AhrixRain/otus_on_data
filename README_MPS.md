# CMS DoubleElectron OTUS on Apple Silicon MPS

This workflow runs the CMS DoubleElectron OTUS training without CUDA-specific paths or install flags. It uses `--device auto`, which selects CUDA if available, otherwise MPS if available, otherwise CPU. On a Mac mini M4 with a current standard PyTorch macOS wheel, it should select `mps`.

## Environment

```bash
conda create -n cms-mps python=3.11 -y
conda activate cms-mps
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cms-mps.txt
python -m ipykernel install --user --name cms-mps --display-name "Python (cms-mps)"
```

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

Run full training with a unique run name:

```bash
RUN_NAME=full_mps_$(date +%Y%m%d_%H%M%S)
python scripts/train.py --config configs/cms_doubleelectron_mps.yaml --device auto --run-name "$RUN_NAME" 2>&1 | tee "outputs/cms_doubleelectron/${RUN_NAME}.log"
```

Evaluate the best checkpoint from that run:

```bash
python scripts/eval.py --config configs/cms_doubleelectron_mps.yaml --checkpoint "outputs/cms_doubleelectron/${RUN_NAME}/best_model.pt" --device auto
```

To monitor progress from another terminal:

```bash
tail -f "outputs/cms_doubleelectron/${RUN_NAME}.log"
```

Use a new `--run-name` for each run. The training script refuses to overwrite an existing directory that already contains `best_model.pt` or `last_model.pt`.

For an intermediate-length test before the full run, use a small per-stage epoch override:

```bash
python scripts/train.py --config configs/cms_doubleelectron_mps.yaml --device auto --num-samples 100000 --epochs 3 --run-name medium_mps
python scripts/eval.py --config configs/cms_doubleelectron_mps.yaml --checkpoint outputs/cms_doubleelectron/medium_mps/best_model.pt --device auto --num-samples 100000
```
