# configs/ layout

- Files directly in `configs/` are the active training configurations.
- `configs/archive/` contains frozen configurations for the archived v3.x
  J/psi runs and the earlier CMS DoubleElectron runs. They are kept for
  provenance and old-checkpoint re-evaluation; do not train from them in new
  rounds unless explicitly re-validating an archived experiment.
