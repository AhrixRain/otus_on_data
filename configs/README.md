# configs/ layout

- Files directly in `configs/` are the active training configurations:
  - `cms_Jpsi_newprior_paper_20pct.yaml` — completed provenance run
    (ptj=10 prior, paper two-term objective).
  - `cms_Jpsi_newprior_ptj5_3term_cosine_20pct.yaml` — completed three-term
    cosine-alpha baseline.
  - `cms_Jpsi_newprior_ptj5_rund_pairswd_cyclemass_20pct.yaml` — Run D
    candidate (pair-level SWD + relative-mass Huber cycle).
- `configs/archive/` contains frozen configurations for the archived v3.x
  J/psi runs and the earlier CMS DoubleElectron runs. They are kept for
  provenance and old-checkpoint re-evaluation; do not train from them in new
  rounds unless explicitly re-validating an archived experiment.
