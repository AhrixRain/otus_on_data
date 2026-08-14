# Persistent Memory — otus_on_data

> This file is the project's persistent memory. It is maintained so that any
> future session (human or AI) can reconstruct context quickly without
> re-deriving prior work. **Read this file first in every session.**

## 0. How to use this file

- **Read first.** It is authoritative for project history, established findings,
  and open questions as of the last session that updated it.
- **Update last.** At the end of any session that produced durable findings,
  update the dated sections and append to §11 Session log. Do not rewrite
  history entries; add new entries.
- **Label evidence.** Use these labels exactly:
  - source-verified (read in code/config)
  - artifact-measured (measured from committed/retained files)
  - reported-not-reproduced (from a deleted or untrusted source)
  - hypothesis
  - proposal
- **Keep the working tree honest.** This file is untracked (the root .gitignore
  does not cover root *.md; git status shows it as ?? memory.md). Commit it
  when you want it versioned.
- Never delete the sections; extend them. Trim only with a dated note.

## 1. Project identity

- Repo: /Users/liziqing/Desktop/Codex/otus_on_data
- What: **OTUS** (Optimal-Transport-based Unfolding and Simulation, SWAE) applied
  to **CMS 2012 Open Data J/psi -> mu+mu-** in the DoubleMuParked sample, with a
  MadGraph5 truth-level prior. Sister studies: successful Z->e+e- OTUS
  (reference PDFs), upstream paper experiments ppzee/ppttbar.
- Paper: docs/2101.08944v2.pdf = arXiv:2101.08944v2 "Learning to Simulate
  High Energy Particle Collisions from Unlabeled Data" (Howard, Mandt, Whiteson,
  Yang). Same file as the synced reference OTUS_paper.pdf (MD5
  3c99a7f95e70c97a2152377c9370571a).
- Upstream OTUS repo: https://github.com/yiboyang/otus (reference only;
  network use is forbidden unless separately authorized).
- Synced reference mirror (treat sources/ as strictly read-only):
  /Users/liziqing/.codex/.chatgpt-projects/g-p-6a3a231c9cb48191a8fbb0a03f979b09
- SWAE reference: arXiv:1804.01947 (Kolouri et al.).

## 2. Repository state snapshot (last updated: Aug 2026 sessions)

- **Git:** branch main; commits f73f01a ("update") -> 2842f99 ("plot
  update") -> a640077 ("v3.8 uppdate"). Only the working tree adds
  docs/2101.08944v2.pdf (untracked) and this file.
- **Data files** (untracked, in data/; configs use paths.data_root: data):
  - data/Run2012BC_DoubleMuParked_Muons.root — 2.24 GB, tree Events,
    **61,540,413 events**, only 6 branches: nMuon, Muon_pt, Muon_eta,
    Muon_phi, Muon_mass, Muon_charge (no isolation/ID/IP branches).
    artifact-measured via notebook outputs.
  - data/cms_jpsi_mumu_mg5_8tev_1M.hdf5 — **signal-only** J/psi->mumu prior,
    1M events, dataset FDL/zData, shape (N,8) = [mu- px,py,pz,E, mu+ px,py,pz,E].
  - data/cms_mumu_inclusive_mg5_8tev_1M.hdf5 — **inclusive** dimuon prior
    (continuum included), used only in the exploration notebook so far.
- **Runs** (outputs/cms_JpsiDoubleMuons/): jpsi_v1, jpsi_v2_5pct, Jpsi_v3,
  Jpsi_v3.5 (main staged model), Jpsi_v3.5_stage_diag, Jpsi_v3.6A_no_explicit_mass
  (mass ablation), stage2_vs_stage3_diagnostic, stage_diagnostic_v35_vs_v36a,
  encoder_alignment_diagnostic, stage3_candidateB_validation, v3.7_lambda1/3,
  ab_plots{,_example,_smoke}, **Jpsi_v3.8_vanilla_paper_all_data_seed0**
  (newest; diverged, see §4.4).
- **Provenance gap (important):** for all runs *except* v3.8, the checkpoints
  (.pt), train_log.csv, status.json, history.json, config.resolved.json,
  metrics.json, per-run summary.{json,md}, and the .plot_cache were **deleted**;
  only PNGs and eval/comparisons/*/mass_histograms.npz survive. v3.8's eval
  NPZs and plots are committed to git.
- **Code state:** active pipeline = scripts/ (train, cms_data, cms_model,
  cms_training, loss, physics, metrics, eval, eval_v37, plot, preflight,
  verify_v37/v38_gradients, plot_jpsi_all_runs, diagnostics) +
  utilityFunctions/models.py, func_utils.py (upstream-OTUS-derived model and
  loss helpers, actively imported). v3.8 added: sampler policies
  (loaders.train_sampler: shuffled_without_replacement,
  loaders.eval_sampler: deterministic_sequential), loss.vanilla_swae,
  scripts/preflight.py, scripts/verify_v38_gradients.py,
  tests/test_v38_vanilla.py.

## 3. Established findings (from the read-only failure investigation)

### 3.1 What "failure" means here
1. **Early-run failure** (runs A/B, v1-v3, pre-v3.5): cycle masses of 2.49 GeV
   (run A) and 1.68 GeV (run B) vs the 3.04 GeV data mean; run A corrupted the
   8-vector while satisfying the mass marginal. reported-not-reproduced
   (documented in configs/cms_JpsiDoubleMuons_mps.yaml comments; artifacts deleted).
2. **Residual failure at HEAD (v3.5):** generator z->x is excellent; the
   reconstruction cycle and the encoder's latent sharpness remain compromised.
3. **Ablation fragility:** v3.6A (no explicit mass supervision) degrades the
   generator ~12x.
4. **v3.8 literal-paper objective diverges** (see §4.4).

### 3.2 Ranked root causes
1. **Prior/data composition + support mismatch (dominant, very high confidence).**
   CMS x window [2.6,3.5] GeV is ~43% J/psi signal + ~57% continuum (sideband
   estimate), broad pair-pT support; the training prior is a pure on-shell
   J/psi (mass sigma ~ 5.7 MeV over the full 100k z_test, ~10.5 MeV over the
   old 50k capped sample), possibly with **zero pair pT** (LO 2->1, unverified —
   see §6). The unpaired SWAE then forces the encoder to crush the continuum
   onto the prior (non-injectivity) — this breaks the cycle and caps latent
   sharpness — and forces the decoder to fabricate the continuum from a
   signal-only prior.
2. **Stage-3 schedule + checkpoint selection degrade the cycle (high).** v3.5
   stage 3 (beta=0.15, tau=2.0, encoder frozen) plus the sim-weighted validation
   score (1.0*x_sim + 1.0*z_prior + 0.5*x_reco) select the sim-best =
   cycle-worst checkpoint.
3. **Z-scale loss recipe transferred untuned (historical, fixed in v3.5).** Run A's
   z-scored mass W1 (weight 2.0+2.0, mass sigma ~10 MeV) had ~100x-amplified
   gradients; v3.5 uses resonance W1 in GeV units instead.
4. **Degenerate prior makes the paper objective ill-conditioned (v3.8).** Raw
   SWD against a delta-like prior has huge, noisy gradients; observed late-stage
   divergence.

### 3.3 What works
- **v3.5 generator (z->x simulation) is a genuine success** at the mass level:
  W1 = 0.0047 GeV, KS = 0.012 on 50k test events; pred histogram nearly
  indistinguishable from truth (peak-bin 3055 vs 3213; frac-in-resonance 0.497
  vs 0.497). artifact-measured.
- v3.5 unfolding mean is pinned to 3.0953 vs prior 3.0969 (offset -1.6 MeV);
  the remaining defect is width (24 vs ~6 MeV) and cycle mass peak (3.03-3.06
  vs 3.095).

### 3.4 Known-suspect verdicts (current source)
- Cache keys/fingerprints, cache-hit content validation: **already fixed**
  (scripts/cms_data.py — version 3 + source hash + config + file
  fingerprints; hits re-validate metadata AND arrays incl. the mass window).
- Float32 invariant-mass cancellation: **fixed + not causal here** (stable
  transverse-mass formula in scripts/physics.py used by loader/eval/plots;
  at E~1.5-3 GeV the direct formula error is ~1e-7 GeV).
- Unknown config keys: **rejected loudly** (scripts/loss.py
  validate_loss_config); all checked-in YAMLs validated clean.
- Unequal-cardinality resonance W1: **fixed** (quantile-grid version).
- Charge ordering [mu-, mu+]: **verified correct** in loader.
- Daughter masses: physical muon masses in active configs; legacy checkpoints
  keep massless semantics deliberately.
- Prefix sampling before shuffling: **confirmed in source, minor** (the
  --num-samples cap is applied in file order before the seeded shuffle).
- One-draw stochastic evaluation: **confirmed in source, minor** (fixed seeds,
  single draw; histogram noise << observed effects).
- Checkpoint selection uses validation (not test): **verified**.
- Z-scale plotting bins for the z-space E/py/pz ratio panels
  (scripts/plot.py bins_e, bins_z): **cosmetic artifact** — all J/psi
  events fall in one bin, making those three ratio plots trivially flat.

## 4. Key measured numbers

### 4.1 CMS x_test (full v3.8 eval set; artifact-measured)
- n = 717,760; mass mean 3.0442, std 0.1853, mode bin 3.095 GeV,
  frac in [3.0369, 3.1569] = 0.505.
- Cut flow (experiments/cms_Jpsi_ee/jpsi2mumu_peak.ipynb saved outputs):
  148,124,900 muons (pT>2, |eta|<2.4) -> 52,998,579 events >=2 muons ->
  83,393,605 OS pairs -> **7,177,454 OS pairs in [2.6,3.5]**; 2,803,269 SS pairs
  in window; data peak 3.0925 GeV (5 MeV bins); window mean 3.0439, median 3.0816.

### 4.2 Prior z_test (artifact-measured)
- Full (v3.8): n = 100,000; mass mean 3.09692, std 0.00571, frac-in-resonance 0.999.
- 50k capped (older evals): mean 3.09685, std 0.01047 (tail-dominated; core is a delta).
- Stored energies are not exactly on-shell (source-verified,
  scripts/loss.py comment); z-space mass uses stored-E authority, x-space
  uses the stable p-based formula.

### 4.3 v3.5/v3.6A/candidates (artifact-measured, 50k/5k test sets as noted)
| Checkpoint | z->x W1 [GeV] | z->x KS | cycle KS | cycle fracR | E(x) mass sigma [MeV] |
|---|---|---|---|---|---|
| v3.5 stage1 | 0.0656 | 0.222 | 0.715 | — | — |
| v3.5 stage2 | 0.0255 | 0.127 | 0.403 | 0.568 | 24.1 |
| v3.5 best | **0.0047** | **0.012** | 0.544 | 0.369 | 24.1 |
| v3.5 final | 0.0054 | 0.012 | 0.528 | 0.584 | 24.1 |
| v3.6A best | 0.0582 | 0.141 | 0.270 | — | 44.4 |
| encoder candidates (5k) | 0.088-0.506 | 0.19-0.29 | — | 0.55-0.94 | 44-95 |
| candidateB stage3 ep20/100/best (5k) | ~0.21/0.20/0.21 | — | — | 0.141/0.846/0.705 | 53.7 (frozen) |

### 4.4 v3.8 vanilla-paper run (artifact-measured)
- Config: configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml
  (raw-coordinate SWD, beta=lambda=1, p=2, 1000 slices, batch 20k, 300 epochs,
  no anchors, all data, seed 0).
- Result: **diverged**.
  - simulation z->x: pred mass mean 23.82, std 34.60 GeV, W1 = 21.2 GeV,
    KS = 0.896 (only 0.5% in window);
  - reconstruction: pred mean 2.92, std 19.18, W1 = 1.59;
  - unfolding: pred mean 6.25, std 23.26, W1 = 2.46 vs a 5.7-MeV-wide prior.
  - Loss-curve PNGs (pixel-profiled) suggest late-stage blow-up (losses rise in
    the final octile).
- Interpretation (hypothesis, high confidence): raw SWD against the
  degenerate prior is ill-conditioned; the literal paper objective does not
  transfer to a delta-like signal-only prior vs continuum-laden data.

## 5. Paper (2101.08944v2) reading notes — what matters for this project

- Core objective Eq. (2)/(11): MSE cycle + lambda*SWD(p_E(z), p(z)) on **raw
  physical coordinates**, p=2, L=1000 slices, batch 20k, Adam lr 1e-3; plus
  anchor terms (beta_E=beta_D=50 for 80 epochs -> 0) with c_A = 1 - p_hat.p_hat'
  on one particle's 3-momentum; Minkowski constraint hard-coded.
- Paper's Z->ee prior and data share **identical composition** (same events,
  truth vs Delphes; "if an event failed this selection, the corresponding Z
  event was also removed").
- lambda ablation: too small -> marginals never match; too large -> unphysical
  maps; lambda~1 for Z->ee; theory says lambda should be annealed upward.
- Paper itself notes X->Z is "less well-described... stricter in Z causing a
  sharper peak", lists "mixtures of underlying priors" as future work, and
  provides the restricted-decoder recipe (Eqs. 17-19, 1_S/P_D(S) reweighting)
  for data the prior cannot produce.
- Semi-supervised paired terms (Eqs. 20-21) are proposed for limited (z,x) pairs.

## 6. Open questions (ranked)

1. **Kinematic support of the priors (highest priority):** what are pair pT/eta
   distributions of cms_jpsi_mumu_... and cms_mumu_inclusive_...? Is the
   signal prior at zero pT (LO 2->1)? Does the inclusive prior's window overlap
   the CMS pair-pT spectrum? *Cheap check: load with h5py in the cms env.*
2. **v3.8 divergence mechanism:** gradient blow-up from raw SWD on a delta
   prior? Verify via a rerun's gradient-norm log or a short diagnostic.
3. **Stage-3 checkpoint policy:** should the science product be final
   instead of best, or should the selection score be re-weighted?
4. **Prefix sampling bias** of --num-samples (file-order cap) — quantify.
5. **Notebook oddity:** identical OS-pair counts at pT>2 and pT>3 GeV
   (7,177,454) — plausibly the DoubleMuParked trigger's ~3 GeV leg floor;
   unexplained, not pursued.

## 7. Proposed fix roadmap (design only — not implemented)

Paper-grounded, one-factor-at-a-time:
- **F1 (primary) — composition-matched prior:** use the inclusive MG5 file
  windowed to [2.6, 3.5] GeV (precomputed file copy, or a
  theory_prior_mass_window loader option) so p(z) and p(x) share the same
  process mixture and mass support; or regenerate the prior with a physical
  J/psi pT spectrum if the inclusive file also lacks pT support.
- **F2 — paper anchor warmup:** beta_E=beta_D=50 -> 0 (first ~80 of 300 epochs)
  on the mu- 3-momentum (prevents charge-inversion solutions).
- **F3 — lambda treatment:** lambda scan {0.1, 1, 10} + upward annealing;
  consider p=1 for the latent SWD (tail-robust); optional gradient clipping as
  a guard.
- **Fallback — paper's restricted decoder (§6.3.2):** restrict the decoder data
  term to the signal region with 1_S/P_D(S) reweighting, making the scope
  explicit (OTUS models the signal region; continuum handled separately).
- **Precondition E1 (no retraining):** dump prior kinematics (see §6.1).
- Sequencing: F1 first as a single controlled change against the frozen v3.8
  config; then F2, F3 separately; never bundle.

## 8. Environment & tooling constraints (from the last DSH sessions)

- **Session 4 (2026-08-15) — conda → .venv.** The project environment is now a
  project-local virtual environment `.venv/` (Python 3.12.13 arm64 via homebrew)
  holding the `cms` environment, installed from `requirements-cms-mps.txt`
  (uproot/awkward/h5py/numpy/scipy/torch-MPS/jupyter etc.). Activate with
  `source .venv/bin/activate`, or run `.venv/bin/python` directly. All runbook
  commands and script docstrings now use `.venv/bin/python` instead of
  `conda run -n cms`; `.venv/` is git-ignored. source-verified (created +
  installed this session). pip/network to pypi worked this session (the install
  was explicitly authorized).

- The **cms conda environment was not present** on the DSH session PATH
  (conda: command not found); no Python with numpy/uproot/h5py/torch was
  found on the machine; **do not pip install** (network forbidden unless
  authorized). The env may exist for the user's own shell; all runbook
  commands assume it.
- /usr/bin/python3 -m py_compile scripts/*.py utilityFunctions/*.py tests/*.py
  works (stdlib only).
- YAML validation possible with system ruby
  (ruby -ryaml -e 'YAML.load_file(ARGV[0])' file.yaml) or Node.
- NPZ (numpy .npz) files were parsed with custom Node scripts
  (/tmp/otus_npz_read.js, /tmp/otus_npz_w1.js, /tmp/otus_npz_hist.js) —
  **ephemeral**: recreate in /tmp if needed (ZIP + NPY header parsing; handles
  f4/f8/i8).
- PDF text extraction: gs -q -dNOPAUSE -dBATCH -sDEVICE=txtwrite -sOutputFile=out.txt in.pdf.
- The current model has **no image input**; PNGs were pixel-profiled with a
  Node PNG decoder (zlib + filters) for curve-shape evidence only.
- ROOT/HDF5 binary parsing was attempted but not completed (no uproot/h5py;
  the ROOT file has a byte-swapped header — parsing needs care).
- In the agent's run_code TS context, require is unavailable — write
  Node scripts with the file tools and execute via bash.

## 9. Conventions & guardrails

- Do not modify, delete, move, or overwrite data files, caches, checkpoints,
  plots, or outputs without an explicit instruction. Preserve untracked files
  exactly (data/, docs/, this file).
- Do not start long training runs unless asked. Prefer --dry-run,
  scripts/preflight.py, and smoke tests for verification.
- Do not regenerate caches to test cache behavior; label such hypotheses
  "plausible but unverified" and propose controlled experiments instead.
- .gitignore ignores *.json, *.pt, *.npz (except **/mass_histograms.npz),
  outputs/**/*.csv — provenance files are never committed by default;
  **keep them on disk anyway**.
- Config paths resolve relative to paths.data_root (currently data);
  cms_root_file: Run2012BC_DoubleMuParked_Muons.root,
  theory_prior_file: cms_jpsi_mumu_mg5_8tev_1M.hdf5.
- Cache lives at <output_root>/.plot_cache and is keyed by selection,
  config, seed, sample cap, file fingerprints, and the pipeline source hash.
- Citation style for findings: file:line (code), notebook cell (ipynb),
  document+section (PDFs), full artifact path (plots/NPZ).

## 10. Quick reference

- Main staged config: configs/cms_JpsiDoubleMuons_mps.yaml (run: Jpsi_v3.5).
- Mass ablation: configs/cms_JpsiDoubleMuons_Jpsi_v3.6A_no_explicit_mass.yaml.
- Vanilla-paper: configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml
  (runbook: docs/Jpsi_v3.8_vanilla_paper_runbook.md; production command NOT
  to be re-run without authorization).
- Evaluation: python scripts/eval.py --config <cfg> --checkpoint <ckpt> --device auto --num-samples <n>.
- Stage comparisons: scripts/stage_diagnostic.py; v3.7/v3.8 three-path eval:
  scripts/eval_v37.py; gradient audits: scripts/verify_v37_gradients.py,
  scripts/verify_v38_gradients.py; data preflight: scripts/preflight.py.
- Loss curves: scripts/plot_loss.py --run-dir <dir> [--components].

## 11. Session log

- **2026-08-14/15 — Session 1 (read-only failure investigation).** Recorded git
  state; read AGENTS.md + all instructions; complete file inventory of repo and
  reference mirror; read all core scripts, configs, tests, reference PDFs
  (ghostscript), and notebooks (cells + outputs); built the run matrix from the
  surviving mass_histograms.npz (custom Node NPZ parser); measured W1/KS/
  peak statistics for every recoverable run; audited caches, configs, numerics
  (stable mass formula), losses, stochastic evaluation, and Z-vs-J/psi
  differences; delivered the full diagnostic report (sections A-M). Findings
  distilled into §3, §4, §6. Marked the session goal complete.
- **2026-08-15 — Session 2 (paper reading + fix design).** Discovered new
  docs/ (paper + v3.8 runbook), commit a640077, and the executed
  Jpsi_v3.8_vanilla_paper_all_data_seed0 run; read arXiv:2101.08944v2 in
  full (main text + supplementary statistics + lambda/beta ablations); measured
  v3.8's divergent results (§4.4); proposed the paper-grounded fix roadmap
  F1-F3 + restricted-decoder fallback (§7). Nothing implemented.
- **2026-08-15 — Session 3 (this file).** Created memory.md at the repo
  root to serve as persistent memory across sessions.
- **2026-08-15 — Session 4 (conda → .venv environment conversion).** Replaced
  conda management with a project-local `.venv` (Python 3.12.13 arm64). Created
  `.venv`, upgraded pip/setuptools/wheel, and installed the `cms` environment
  from `requirements-cms-mps.txt` (Apple Silicon / MPS variant; the CUDA
  `requirements-cms.txt` is Linux-only). Registered the ipykernel as `cms`.
  Added `.venv/` to `.gitignore`. Updated `conda run -n cms` / conda-create
  references to `.venv/bin/python` in README_MPS.md,
  docs/Jpsi_v3.8_vanilla_paper_runbook.md, and the script docstrings
  (encoder_alignment_report.py, stage_diagnostic.py, test_plot.py,
  stage3_candidateB_validation.py, stage3_candidateB_report.py); refreshed the
  requirements-file header comments. Left README.md's upstream py36-otus note
  and the notebook cell outputs untouched.
