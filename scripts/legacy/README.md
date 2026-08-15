# scripts/legacy/

Frozen diagnostics and report builders for the archived v3.x J/psi runs
(now under `outputs/cms_JpsiDoubleMuons/archive/` and
`configs/archive/`).

These scripts are intentionally **not** part of the next-round training path.
They are kept for reproducibility only. Run them from the repo root:

```bash
PYTHONPATH=scripts /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python \
  scripts/legacy/<script>.py --help
```

Do not extend them; new diagnostics belong in `scripts/diagnostics/`.
