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

- Repo: /Users/liziqing/Desktop/Codex/otus_on_data (current machine:
  /Users/ahrimarin/Desktop/otus_on_data; same content, moved 2026-08-14)
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
  ab_plots{,_example,_smoke}, Jpsi_v3.8_vanilla_paper_all_data_seed0
  (diverged, see §4.4), Jpsi_v3.9_F1_restricted (aborted pre-cut warmup,
  2 epochs, kept for provenance), Jpsi_v3.9_F1_restricted_ptmax100
  (TERMINATED at epoch ~20 by user request 2026-08-15; kept on disk.
  Preliminary metrics at that point: train loss ~4.4, eval 15.1, generator
  mass W1 ~1.48/KS 0.56 — early-training immaturity, not a verdict. The user
  will run the final fix later; see docs/Jpsi_F1_fix_runbook.md).
- **ARCHIVED 2026-08-14 (Session 9):** everything listed above was moved
  to outputs/cms_JpsiDoubleMuons/archive/ (incl. .plot_cache), and ALL
  configs/*.yaml (18 J/psi + the Z config cms_doubleelectron_mps.yaml) were
  moved to configs/archive/ (git-staged as renames). outputs/cms_Jpsi_new/
  was created for the upcoming new-prior training. All code references
  (scripts/, tests/, docs/) were updated to the archive paths; suite 98/98
  green; jpsi_new_prior_comparison.py re-verified against the archived cache.
- **Provenance gap (old machine only):** for all runs *except* v3.8, the
  checkpoints (.pt), train_log.csv, status.json, history.json,
  config.resolved.json, metrics.json, per-run summary.{json,md}, and the
  .plot_cache were **deleted** on the old machine; only PNGs and
  eval/comparisons/*/mass_histograms.npz survive there. v3.8's eval NPZs and
  plots are committed to git.
- **Machine-state correction (2026-08-15, current machine):** on
  /Users/ahrimarin/Desktop/otus_on_data the provenance files DO exist for all
  listed runs — jpsi_v1, jpsi_v2_5pct, Jpsi_v3, Jpsi_v3.5 (best_model.pt,
  history.json, train_log.csv, status.json), Jpsi_v3.6A (all stage
  checkpoints), v3.7_lambda1/3 (best_combined/best_cycle/best_z_prior +
  final). Any historical checkpoint can be re-evaluated with the current
  scripts/eval.py harness directly. artifact-measured (directory listings).
- **Code state:** active pipeline = scripts/ (train, cms_data, cms_model,
  cms_training, loss, physics, metrics, eval, eval_v37, plot, preflight,
  verify_v37/v38_gradients, plot_jpsi_all_runs, diagnostics) +
  utilityFunctions/models.py, func_utils.py (upstream-OTUS-derived model and
  loss helpers, actively imported). v3.8 added: sampler policies
  (loaders.train_sampler: shuffled_without_replacement,
  loaders.eval_sampler: deterministic_sequential), loss.vanilla_swae,
  scripts/preflight.py, scripts/legacy/verify_v38_gradients.py,
  tests/test_v38_vanilla.py. v3.9 (2026-08-15) added:
  cms_data.filter_theory_prior + theory_prior_selection cache metadata +
  optional muon_pt_max in the muon loader; scripts/diagnostics/prior_kinematics.py,
  scripts/diagnostics/data_kinematics.py (E1 dumps); tests/test_v39_f1.py (10 tests).

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
   score (1.0*x_sim + 1.0*z_prior + 0.5*x_reco) — 2026-08-15 refinement: the
   score is z_prior-dominated (~9.2 of ~10.5) and the cycle term (weight 0.5,
   scale ~1.7) is masked, so checkpoint selection is effectively blind to the
   cycle; see §6.3 for the measured trajectory.
3. **Z-scale loss recipe transferred untuned (historical, fixed in v3.5).** Run A's
   z-scored mass W1 (weight 2.0+2.0, mass sigma ~10 MeV) had ~100x-amplified
   gradients; v3.5 uses resonance W1 in GeV units instead.
4. **Degenerate prior makes the paper objective ill-conditioned (v3.8).** Raw
   SWD against a delta-like prior has huge, noisy gradients; observed late-stage
   divergence. 2026-08-15 refinement (source-verified + artifact-measured): the
   deeper structural issue is that the vanilla objective never constrains
   D(z~prior) directly — D is only trained through the cycle x->E(x)->D(E(x)),
   so when the latent SWD fails (non-injectivity), D(z~prior) is never
   exercised and the generator degrades catastrophically while the cycle loss
   stays small (§4.4 trajectory).
5. **The paper-vanilla objective is mass/correlation-blind even with the
   matched new prior (2026-08-15, source-verified + artifact-measured).**
   `Jpsi_newprior_paper_20pct` (new mixed MG5 prior) converges on the raw
   latent SWD (`mean_z_ks` ~0.017, every z component KS <=0.014, max KS over
   1000 random linear projections ~0.03) while E(x) puts only **3.1%** of
   events in the [3.0369,3.1569] mass window (z mass KS 0.760, mean
   4.65 vs 3.097 GeV). The lost quantity is the two-particle angular
   correlation: z prior std(Delta_eta,Delta_phi) = (0.207,0.217),
   E(x) = (0.372,0.336). Both training terms are coordinate-level — raw MSE
   and random linear-projection SWD — and the J/psi mass is a narrow nonlinear
   shell (30 MeV) inside GeV-scale coordinate distributions. The x->z->x
   cycle consequently keeps only 13.8% of events in the mass window
   (mass mean 2.903, std 0.511 vs data 0.028), despite good per-component
   residual closure (0.3-0.7 GeV). D(z_prior) is even worse (5.6% in window;
   mass std 1.369) because `tau=rho=nu_d=0` after epoch 80 means the decoder
   is never trained directly on real prior z. The low-pT collapse in
   `paperstyle_xspace_pt_density_ratio.png` is a prior support hole, not a
   cycle failure: the MG5 cards use `ptj>=10`, so z_prior has pair pT
   min 9.70 GeV and 0.06% below 10 GeV vs data 2.2% below 10 GeV (min 5.18);
   D(z_prior) cannot populate data's low-pT bins. Full report:
   `outputs/cms_Jpsi_new/Jpsi_newprior_paper_20pct/density_failure_analysis.md`;
   reproducible metrics: `scripts/diagnostics/diagnose_jpsi_new_density.py`.
   Causal verification: 300 encoder-only Adam steps with
   `raw_SWD + 10*W1(mass)` improved latent mass KS 0.759 -> 0.425 and
   mass-window fraction 3.0% -> 16.0% (component KS stayed <=0.033), while
   the frozen decoder's cycle mass worsened (0.569 -> 0.748) because D was
   only ever trained on the old mass-broken E(x) support
   (`diagnostic_mass_probe_encoder.pt`, `mass_probe_metrics.json`).

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

### 4.3b Fresh same-harness baselines (2026-08-15, eval.py, mass range
[2.6,3.5], 90 bins; artifact-measured on this machine):

| Checkpoint | sim W1 [GeV] | sim KS | reco W1 | reco KS | unfold W1 | unfold KS |
|---|---|---|---|---|---|---|
| v3.5 best_model (ep160) | 0.0086 | 0.045 | 0.1244 | 0.520 | 0.0103 | 0.440 |
| v3.8 best_combined (ep280) | 20.90 | 0.896 | 1.594 | 0.594 | 5.086 | 0.618 |

Scope caveat for the v3.9 verdict: these baselines were evaluated on the FULL
[2.6,3.5] window (continuum included); v3.9's eval is on the signal region
only, so its W1/KS are structurally smaller. The v3.9 verdict must compare
(a) same-scope W1/KS plus shape stats (peak mean/std vs signal-region data),
and (b) out-of-scope behavior (feeding full-window data to the v3.9 model) as
a diagnostic.

### 4.5 E1 kinematics dump (2026-08-15; artifact-measured via
scripts/diagnostics/prior_kinematics.py + scripts/diagnostics/data_kinematics.py)

Prior files (1M events each, FDL/zData, daughters exactly massless: E = |p|):
- signal cms_jpsi_mumu_mg5_8tev_1M.hdf5: pair mass E-based mean 3.0969,
  std 5.7 MeV (p-based std 0.2 MeV: the mass smearing lives in the stored E,
  z-space authority); muon pz std 244 GeV, |eta| up to 12.2 (median 3.1);
  pair pT (vector) median 1.83, max 110 GeV — not pure 2->1 (recoil present);
  100% of events have mass in [3.0369, 3.1569].
- inclusive cms_mumu_inclusive_mg5_8tev_1M.hdf5: **pure 2->1** — pair pT
  (vector) ≡ 0 exactly; pair mass median 3.38, range 2.56-329 GeV; J/psi-window
  events have muon pT = m/2 ~ 1.55 GeV, so **0 of 1M pass** both-muons pT>3 +
  |eta|<2.4 + mass window. Refutes the original F1 "inclusive file windowed"
  design (§7): the inclusive file is unusable as a fiducial-matched prior.
- v3.9 theory_prior_selection (pT>3, |eta|<2.4, mass window) keeps **5,995 of
  1M signal events (0.60%)**; filtered pair pT median 10.1, mean 12.1.

CMS data (Run2012BC_DoubleMuParked_Muons.root):
- 149,322,456 stored muons; **hard 3.0 GeV trigger floor: zero muons below
  3.0 GeV** — answers §6.5 (the pT>2 vs pT>3 OS-pair counts are identical
  because no muons exist in (2,3] GeV). 148,124,900 (99.2%) pass |eta|<2.4
  (matches §4.1's muon count; the "(pT>2, |eta|<2.4)" label there was
  effectively a pT-only count).
- Junk tail: 0.40% of muons above 100 GeV, 0.14% above 200 GeV, 0.027% above
  1 TeV (max ~16.8 TeV) — misreconstructed junk inside the mass window,
  unproducible by the prior (muon pT support ends ~94 GeV) and a raw-SWD
  gradient hazard.
- Signal region [3.0369, 3.1569] (loader selection): pair mass mean 3.094,
  std 31 MeV (E-based); muon pT median 12.4; pair pT median 26.0, mean 27.4,
  p95 47.4. The prior stays ~2.5x softer than the parked-B data even after
  trigger matching (known residual limitation of the MG5 file).
- Per-coordinate support coverage (full x_test vs the full 5,995-event
  filtered prior, artifact-measured 2026-08-15): data px/py span ±27 GeV
  (p99) vs prior ±14-15 GeV; ~83-84% of data events fall inside the prior's
  empirical 1-99% range per transverse coordinate; pz coverage 93-94%; E
  coverage ~96%. The encoder must transport ~16% of data events (the high-pT
  tail) into a ~2x-narrower transverse support — a bounded continuous
  distortion the SWD can in principle absorb, at the cost of unfolding
  fidelity.

### 4.7 CMS-vs-MG5 mismatch re-check, J/psi ~3.09 GeV focus (artifact-measured
2026-08-14 via scripts/diagnostics/jpsi_mismatch_check.py + scripts/diagnostics/jpsi_f32_check.py,
version-3 split caches 0ed04817d72bb34f84ed / 410ce075cc7cc2e067c8; the user's
question: is the training failure a CMS-data/MG5-prior mismatch?)

**Verdict: yes — quantified on four stacked axes (all numbers below).**

- **Composition (mass axis).** Full window [2.6,3.5]: data = 42.2% J/psi +
  57.8% continuum (linear sideband est.; 7,177,600 OS pairs). Prior = 100%
  on-shell J/psi. Even the narrow signal region [3.0369,3.1569] keeps ~15%
  continuum under the window; the prior has 0%. The prior is a delta: filtered
  prior mass std 0.22 MeV (E-based) / 0.20 MeV (p-based) vs data in-window
  std 27.9 MeV — a ~140x width gap, i.e. the generator must learn ~28 MeV of
  resolution smearing from a delta, and the encoder must crush continuum onto
  the delta (non-injectivity).
- **Mass conventions (training-exact).** x-space uses p-based stable mass with
  daughter_masses [0.105658,0.105658] (loss.py:1000, mass_from_energy=False);
  z-space uses stored-E direct formula (loss.py:1007, mass_from_energy=True).
  In these conventions the in-window data mean is 3.09779 vs prior 3.09690 —
  only +0.9 MeV offset, so the mismatch is WIDTH + CONTINUUM, not a mass-scale
  shift. (The massless-daughter convention would fake a -7.3 MeV offset; the
  daughter-mass correction is +8.25 MeV.)
- **Process/physics content.** Prior HDF5 attrs (source-verified):
  process = "p p > jpsiv j, jpsiv > mu+ mu-" (prompt direct production only,
  effective coupling gJpsi=0.01; file carries its own scientific_warning that
  absolute normalization is untrustworthy). No B-decay (non-prompt) component,
  no feed-down — while real CMS 8 TeV J/psi is prompt+non-prompt+feed-down.
- **Kinematic support.** Same filter (both-muons pT>3, |eta|<2.4, mass window)
  keeps 49.7% of full-window data but only 0.60% of the 1M prior. Unfiltered
  prior (v3.5/v3.8 setup): muon pT median 1.49 vs data 11.43 GeV; prior p99 =
  7.10 GeV yet 77.6% of data muons are harder; only 5.7% of full-window data
  lies inside the prior's joint [p0.5,p99.5] box (px/py coverage ~40%). Even
  the trigger-matched filtered prior stays 2.6x softer: muon pT median 4.8 vs
  12.6, pair pT median 10.1 vs 26.0, pair pT p99 39.5 vs 65.7; 69.8% of
  signal-window data inside the filtered prior's box. Unproducible tails:
  0.24% of data pair pT above prior absolute max (110 GeV); data junk muons
  reach 16.8 TeV.
- **Float32 z-mass noise (new, important).** The prior's daughters are
  effectively massless (stored E - |p| per muon: mean ~0, std 0.01 MeV), so
  the E-based z-space mass is an opening-angle mass with float32 cancellation
  at boost: full-prior z mass std = 5.65 MeV in float64 but 10.47 MeV in
  float32 (what training actually sees); p-based stays 0.20 MeV. The filtered
  prior (energies <=76 GeV) has E-based std 0.22 MeV — the 10.5 MeV "width"
  of the unfiltered prior is numerical noise, not physics. The data x-mass
  authority (p-based) is immune (stable formula), but the stored-E column of
  the data differs from p-based mass by mean 8.25 MeV in-window (daughter-mass
  convention + float32 junk-tail rounding; max 539 MeV on junk events).
- **Causal fit to the observed failures:** v3.5's good direction is z->x
  (decoder only smears the delta — unaffected by mismatch) and its broken
  direction is x->z (cycle KS 0.52-0.54, encoder mass sigma stuck at 24 MeV) —
  exactly where the mismatch bites. v3.8's vanilla objective diverges because
  raw SWD against a delta prior is ill-conditioned and D(z~prior) is never
  trained. The v3.9/v3.10 restricted scope removes the 58%->15% continuum
  crush and the trigger-floor asymmetry but keeps the 2.6x pT gap and the
  140x width gap — mitigating, not eliminating, the mismatch. Root fix would
  need a prior with resolution smearing (~28 MeV), continuum admixture, and a
  harder (incl. non-prompt) pT spectrum.

### 4.8 Z->ee health check (artifact-measured 2026-08-14 via
scripts/diagnostics/zee_health_check.py, cache 9bb63a1e38cf0cd9b9c2 version 1;
the user asked to verify the Z data/prior pair is "normal")

**Verdict: Z is normal/healthy — the contrast case that validates the J/psi
verdict. Every dimension where J/psi failed is healthy for Z.**

- Prior (cms_dyee_mg5_8tev_dy1j_ptj5_fiducial_70_110.hdf5, n=623,944): a real
  Z lineshape, NOT a delta — mass mean 91.089, std 4.03 GeV, median 91.187
  (= pole 91.1876, fine mode 91.188); daughters massless (stored E-|p| per
  electron ~4e-6 GeV) but no float32 problem at these energies (E-based ==
  p-based to machine precision); 100% in [70,110].
- Data (1,344,681 OS pairs, full electron selection): mean 90.638 (low-mass
  tail), median 91.148, half-width 3.60 GeV vs prior 1.86 GeV (~1.9x wider —
  resolution + endcap scale mixing, see below). Composition: [70,80) 5.5% /
  [80,100] 90.0% / (100,110] 4.5% vs prior 2.1%/95.3%/2.5%; data off-peak
  continuum density ~4-5x the prior's — a modest, smooth mismatch in the same
  lineshape family (vs J/psi's 58% continuum with 0% in the prior).
- **One real caveat found (skim-level, not OTUS):** the skim's electron energy
  scale is detector-region dependent. Both-barrel pairs: fine mode 91.182
  (spot-on). Both-endcap pairs: fine mode 92.942 (+1.9% high). The mixed
  barrel-endcap pairs create the 91.6 plateau; overall data fine mode 91.617
  (+430 MeV vs pole). This is the classic un-recalibrated 2012 endcap ECAL
  scale effect; the truth prior sits at the pole. It contributes to the 1.9x
  width gap but is absorbable — see the converged runs below.
- Kinematics: essentially matched. electron pT mean 42.09 vs 41.22 (median
  41.2 vs 40.6); pair pT median 12.4 vs 11.9, p99 123 vs 109; |eta| mean 1.03
  vs 1.08. Only 1.3% of data electrons are harder than the prior's p99
  (J/psi: 77.6%).
- Support: 93.4% of data events fully inside the prior's [p0.5,p99.5] box
  (per-coordinate 98.7-99.0%; J/psi: 5.7%). The prior passes the data-side
  kinematic selection (pT>20, |eta|<2.5, mass window) at 100% (J/psi: 0.60%).
- Training-side confirmation: v5_mps (best ep160) eval loss 0.128, mass
  residual mean 0.078 / rms 0.133, chi2_like 4.2e-5, all 80 bins valid;
  v4_mps residual 0.119/0.190. Converged, high-quality fits.
- Alt prior (no-dy1j fiducial file): n=640,449, mean 91.007, std 4.070,
  median 91.166 — same family as the dy1j file; both healthy.

### 4.6 v3.9 F1-restricted pilot — early trajectory (artifact-measured,
2026-08-15, train_log.csv; run ongoing)

| metric @epoch 1 / 10 | v3.9 | v3.8 (same epochs) |
|---|---|---|
| train loss | 57.0 / 4.74 | 3245 / 764 |
| train x (reco) | 48.4 / 0.66 | 1434 / 556 |
| train z (latent SWD) | 8.64 / 4.08 | 1811 / 208 |
| eval loss | 14.9 / 15.1 | 1935 / 914 |
| eval z | 12.4 / 14.6 | 369 / 396 |
| grad enc latent (total) | 131 / 62 | 17530 / 11126 |
| grad dec total | 140 / 41 | 2253 / 2897 |

The reconstruction term converges ~instantly (0.66 by epoch 10) and the
latent SWD keeps decreasing on the train side with ~200x tamer gradients
than v3.8 — bounded support is behaving as hypothesized. Eval z lags train z
(14.6 vs 4.1 at epoch 10) and will be the number to watch at epochs 20/50+.
Pilot caveat (hypothesis, high confidence): the eval SWD computes 1000
projection slices over only z_val=599 events (<1 point/slice), so eval_z is
sparse-slice noisy and biased upward — do not over-interpret its
fluctuations; checkpoint selection (selection score = eval loss) is
consequently noisy too, so the verdict should eval BOTH best_model.pt and
checkpoint_final.pt.

### 4.4 v3.8 vanilla-paper run (artifact-measured)
- Config: configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml
  (raw-coordinate SWD, beta=lambda=1, p=2, 1000 slices, batch 20k, 300 epochs,
  no anchors, all data, seed 0).
- Result: **diverged**.
  - simulation z->x: pred mass mean 23.82, std 34.60 GeV, W1 = 21.2 GeV,
    KS = 0.896 (only 0.5% in window);
  - reconstruction: pred mean 2.92, std 19.18, W1 = 1.59;
  - unfolding: pred mean 6.25, std 23.26, W1 = 2.46 vs a 5.7-MeV-wide prior.
  - Loss trajectory (train_log.csv, artifact-measured 2026-08-15): NOT a
    monotonic blow-up. Train loss plateaued at ~100-400 from epoch 10 onward;
    eval bounced (best 127.3 near epoch 280); a transient spike at epoch 290
    (train 1304, latent grad norm 73,774) recovered by epoch 300 (298 / 4563).
    The catastrophic D(z_test) numbers come from a converged-but-degenerate
    solution: the vanilla objective never constrains D(z~prior) directly —
    D is only trained through the cycle x -> E(x) -> D(E(x)). When the latent
    SWD fails to place E(x) on the prior's support (continuum crush), D is
    never exercised on z ~ p(z) and fabricates garbage there while the cycle
    still looks acceptable. The v3.9 restricted pilot directly tests whether
    matched support lets the SWD succeed, which is what closes this hole.
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

1. ~~Kinematic support of the priors~~ **ANSWERED 2026-08-15**: see §4.5. The
   signal prior is boosted+recoil (not pure 2->1), passes the trigger-matched
   selection at 0.60%; the inclusive prior is pure 2->1 and passes at 0.00%.
2. ~~v3.8 divergence mechanism~~ **REFINED 2026-08-15**: the "divergence" was
   a transient epoch-290 spike (latent grad 73.8k) on an otherwise-plateaued
   run; the catastrophic generator came from a converged-but-degenerate
   solution where D(z~prior) is never trained (§4.4). The v3.9 F1-restricted
   pilot (running 2026-08-15) tests whether matched support closes the hole.
3. ~~Stage-3 checkpoint policy~~ **ANSWERED 2026-08-15** (v3.5 history.json,
   artifact-measured): the score = 1.0*x_sim + 1.0*z_prior + 0.5*x_reco is
   dominated by z_prior (~9.2 of ~10.5) in standardized units; with the
   encoder frozen in stage 3, z_prior is nearly constant, so the score's
   dynamics come from x_sim/x_reco fluctuations at ~20x smaller scale. The
   cycle (x_reco 1.67@ep160 -> 1.84@ep200) is masked by the z_prior scale —
   "best" (ep160) vs "final" (ep200) differ by ~0.25 score units driven
   mostly by x_sim (0.477 vs 0.546). For v3.10's stage-3 policy: re-weight
   the cycle term upward (or z-score each term) so the cycle actually
   influences selection.
4. ~~Prefix sampling bias~~ **ANSWERED 2026-08-15**: negligible. The 1M
   file-prefix cap reproduces the full x_test statistics within ~1% (muon pT
   median 12.4 vs 12.59; pair pT median 26.0 vs 26.05; mass mean 3.086 vs
   3.0943 GeV, std 28.4 vs 28.1 MeV). artifact-measured vs the full-run data
   cache.
5. ~~Notebook oddity~~ **ANSWERED 2026-08-15**: hard 3.0 GeV trigger floor in
   the skim (zero stored muons below 3.0 GeV); no muons exist in (2,3] GeV,
   hence identical pT>2 and pT>3 pair counts (§4.5).

## 7. Fix roadmap — current status (trimmed 2026-08-16)

The original F1-F3 roadmap is superseded by the executed new-prior and
three-term runs; keep only the current chain:

- Prior: rebuilt with ptj=5 -> `data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5`
  (old ptj=10 file retained as provenance).
- Baseline objective: three-term cosine
  `alpha(t)*[SWD(z,E(x)) + SWD(x,D(z))] + lambda_cycle*MSE(x,D(E(x)))`
  (`configs/cms_Jpsi_newprior_ptj5_3term_cosine_20pct.yaml`; completed).
- Run D extension: pair-level SWD over
  `[log m, log pT, y, cos dphi, sin dphi]` plus cycle relative-mass Huber
  (`configs/cms_Jpsi_newprior_ptj5_rund_pairswd_cyclemass_20pct.yaml`;
  dry-run validated, not trained yet).
- All historical v3.x configs live in `configs/archive/`; frozen diagnostics
  in `scripts/legacy/`.

## 8. Environment & tooling constraints (current)

- Current interpreter on this machine:
  `/opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python`
  (Python 3.10.20, torch 2.12.1 MPS, numpy 2.2.5, h5py, scipy, uproot,
  awkward, matplotlib, Pillow). `.venv` instructions in README_MPS.md are the
  portable alternative; `.venv/` does not exist on this machine.
- MG5_aMC 3.7.0 at `~/MG5_aMC_v3_7_0` with the macOS/LHAPDF fixes documented
  in `docs/jpsi_prior_rebuild_runbook.md`.
- Pyflakes is installed in the cms env (`python -m pyflakes ...`).
- Tests: `python -m unittest discover -s tests`.
- Full training runs must be explicitly authorized; prefer preflight/dry-run.

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
- As of 2026-08-14 the historical J/psi split caches live at
  outputs/cms_JpsiDoubleMuons/archive/.plot_cache (scripts
  jpsi_mismatch_check/jpsi_f32_check/jpsi_new_prior_comparison point there);
  the new training's caches will be created under outputs/cms_Jpsi_new/.plot_cache.
- Citation style for findings: file:line (code), notebook cell (ipynb),
  document+section (PDFs), full artifact path (plots/NPZ).

## 10. Quick reference (current)

- Configs:
  - completed baseline: `configs/cms_Jpsi_newprior_paper_20pct.yaml`
  - completed three-term cosine: `configs/cms_Jpsi_newprior_ptj5_3term_cosine_20pct.yaml`
  - Run D candidate: `configs/cms_Jpsi_newprior_ptj5_rund_pairswd_cyclemass_20pct.yaml`
  - all old v3.x configs: `configs/archive/`
- Prior files:
  - active: `data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5` (ptj=5)
  - provenance: `data/cms_jpsi_mumu_mg5_8tev_mixed.hdf5` (ptj=10)
  - original delta prior: `data/cms_jpsi_mumu_mg5_8tev_1M.hdf5`
- Commands:
  - `python scripts/preflight.py --config <cfg>`
  - `python scripts/train.py --config <cfg> --dry-run`
  - `python scripts/train.py --config <cfg> --device auto --run-name <name>`
  - `python scripts/eval.py --config <cfg> --checkpoint <ckpt> --output-dir <dir>`
  - `python scripts/plot.py --config <cfg> --checkpoint <ckpt> --output-dir <dir>`
  - `python scripts/plot_loss.py --run-dir <run>`
- Structure:
  - `scripts/` active pipeline + CLI
  - `scripts/prior_build/`, `scripts/diagnostics/`, `scripts/legacy/`
  - see `scripts/README.md` and `configs/README.md`

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
- **2026-08-15 — Session 5 continued (user handover).** At the user's request
  ("implement the fix first, without actual running, terminate the current
  runs"), terminated the v3.9 pilot at epoch ~20 (kept on disk; preliminary
  metrics recorded in §2) and finalized the fix implementation: v3.10
  config gained the re-weighted checkpoint-selection score (§6.3 finding),
  dry-run validated; wrote docs/Jpsi_F1_fix_runbook.md with all run/eval
  commands; committed. No training runs in progress — the user runs the fix
  later.
- **2026-08-15 — Session 5 (E1 kinematics + v3.9 F1-restricted pilot).** Found
  the committed-but-unrun v3.9 F1 design (commit 48fb0ca31: filter_theory_prior
  + v3.9 config). On this machine the `cms` conda env is the interpreter
  (§8). Ran E1: prior kinematics (both MG5 files), CMS trigger floor, and
  signal-region pair kinematics; all numbers in §4.5; answered §6.1 and §6.5.
  Discovered the skim's junk tail (muons up to ~16.8 TeV in the mass window)
  and the prior's 0.60% pass rate (5,995 events). Added the optional
  muon_pt_max guard to the muon loader (scripts/cms_data.py), set it to
  100 GeV in the v3.9 config, and rewrote the config header with the measured
  E1 facts. Wrote scripts/diagnostics/prior_kinematics.py, scripts/diagnostics/data_kinematics.py,
  tests/test_v39_f1.py (10 tests, all green; full suite passes). First v3.9
  launch (pre-cut config) was killed during epoch 2 and left as
  Jpsi_v3.9_F1_restricted (provenance only); the production pilot is
  Jpsi_v3.9_F1_restricted_ptmax100 (300 epochs, ~181.5k steps, ~13h MPS,
  launched this session). Early signal: epoch-1 train loss ~233 vs v3.8's
  ~3,245 (bounded support confirms the hypothesis direction).
- **2026-08-14 — Session 6 (CMS-vs-MG5 mismatch re-check).** The user asked
  whether the J/psi training failure is a CMS-data/MG5-prior mismatch (with
  the ~3.09 GeV region as the reference point; not restricted to it). Ran a
  fresh quantitative check from the version-3 split caches: composition
  (42.2% signal / 57.8% continuum full window; ~15% continuum even inside
  the signal region; prior 100% signal delta), width (27.9 MeV data vs
  0.2 MeV prior, 140x), kinematic support (same filter keeps 49.7% of data
  vs 0.60% of prior; 77.6% of data muons harder than prior p99; filtered
  prior still 2.6x softer), and a new float32 finding (unfiltered prior's
  10.5 MeV E-based z-mass "width" is cancellation noise; filtered prior is a
  true 0.2 MeV delta). Verified the training-exact mass conventions
  (z: stored-E direct; x: p-based + muon masses; data mean 3.09779 vs prior
  3.09690, +0.9 MeV). Verdict: the mismatch is the dominant root cause and
  explains every observed failure signature; the v3.9/v3.10 fix mitigates but
  does not eliminate it. Wrote scripts/diagnostics/jpsi_mismatch_check.py and
  scripts/diagnostics/jpsi_f32_check.py; findings in §4.7. Then, at the user's request,
  ran the same methodology on the Z->ee pair: healthy on every dimension
  (prior is a real 4-GeV-wide lineshape at the pole, kinematics matched at
  the few-% level, 93.4% support coverage, 100% pass rate, converged v4/v5
  runs). One real caveat found: the skim's endcap electron energy scale is
  +1.9% high (endcap-only mode 92.94 vs pole 91.19; barrel-only 91.18) —
  a skim-level 2012 ECAL calibration artifact, absorbable, not an OTUS bug.
  Wrote scripts/diagnostics/zee_health_check.py; findings in §4.8.
- **2026-08-14 — Session 7 (MG5 on macOS + J/psi prior rebuild, goal round 1).**
  Verified MG5 3.7.0 runs on this Mac (conda python has six; system 3.9.6
  does not); installed brew gcc (gfortran 16.1.0) and conda-forge lhapdf
  6.5.6 + NNPDF31_lo_as_0130 grid (lhaid 315200). Rebuilt sm_onia-c_mass and
  sm_mumass-c_mass from Virat's recipe with massive muons (MM=0.105658; the
  recipe's ymm confusion avoided, 92.9 keV width kept). Fixed two macOS MG5
  blockers: the dyld @rpath/libLHAPDF.dylib 'Reason:' survey crash
  (-Wl,-rpath in Source/make_opts + Template/LO/Source/make_opts[.make_opts]
  + DYLD_LIBRARY_PATH) and the NNPDF23_lo_as_0130_qed systematics errorset
  crash (use_syst=False); also cut_decays=True so generation cuts apply to
  decay muons (92-96% post-filter efficiency vs 0.6% for the old file).
  Generated 200k signal + 50k continuum events; postprocessed, smeared
  (a=0.0127, mass std 27.8 MeV vs data 28.1), reweighted to the data muon-pT
  spectrum (median 6.95 -> 13.30 vs data 12.57), and mixed 85/15 into
  data/cms_jpsi_mumu_mg5_8tev_mixed.hdf5 (94,880 events). All acceptance
  targets met (§7 update; full metrics there). New scripts:
  lhe_to_prior_hdf5.py, smear_prior.py, reweight_prior.py, mix_prior.py;
  cards scripts/mg5_cards/; runbook docs/jpsi_prior_rebuild_runbook.md;
  tests/test_prior_build.py (7/7).
- **2026-08-14 — Session 8 (new-prior comparison figure + discovery narrative).**
  At the user's request, produced a CMS-vs-new-prior comparison in the style of
  experiments/cms_Jpsi_ee/ (where the old jpsi_prior_vs_cms_mumu.png lives):
  scripts/diagnostics/jpsi_new_prior_comparison.py renders a 3x2 panel figure to
  experiments/cms_Jpsi_ee/jpsi_new_prior_vs_cms_mumu.png comparing CMS
  signal-window data (v3.9 cache 0ed04817d72bb34f84ed, n=3,624,454) against the
  OLD 1M delta prior (5,995 events pass the data selection = 0.60%) and the NEW
  mixed prior (89,435/94,880 = 94.26% pass, the ~5.7% loss is smearing-induced
  mass-window spill-out). Panels: mass (2 MeV bins) / mass zoom (1 MeV bins) /
  muon pT / pair pT (both log-y) / muon |eta| / summary table. Freshly measured
  (training-exact x-space convention, m_mu=0.105658): data mass 3.09428 +- 28.1
  MeV, muon pT med 12.57, pair pT med 26.03; old prior 3.10506 +- 1.6 MeV
  (massless-born kinematics through the massive-daughter formula), pT med 4.77,
  pair pT med 10.09, joint-box coverage 69.8%; new prior 3.09697 +- 30.8 MeV,
  pT med 13.30, pair pT med 26.60, coverage 97.4%; sideband continuum under the
  signal window 14.7%. Figure pixel-verified (all three curve colors present);
  the model has no image input so visual QA was programmatic only.
- **2026-08-14 — Session 9 (archive old J/psi work, prep new training).**
  Per user request: created configs/archive/ and
  outputs/cms_JpsiDoubleMuons/archive/; moved ALL previous files into them
  (18 J/psi configs + cms_doubleelectron_mps.yaml -> configs/archive/; all 36
  entries incl. .plot_cache -> outputs/cms_JpsiDoubleMuons/archive/), and
  created outputs/cms_Jpsi_new/ for the new training. Staged exact git
  renames for the 903 previously-tracked files (19 configs + 884 outputs
  files) so history is preserved as renames, not delete+add. Updated 74
  path references across scripts/ (12 files incl. the cms_doubleelectron
  plot notebook's repo-root finder), tests/ (3 files), docs/ (3 runbooks)
  plus 3 split-string references the blanket replace missed
  (plot_jpsi_all_runs.DEFAULT_RUNS_ROOT, notebook lines 34/64). Verified:
  zero stale refs in scripts/tests/docs, py_compile clean, unittest suite
  98/98 OK, and scripts/diagnostics/jpsi_new_prior_comparison.py re-run end-to-end
  against the archived .plot_cache (same numbers as Session 8). Leftover:
  an untracked PLY-generated parser table py.py at repo root (left in
  place, not part of this task).
- **2026-08-14 — Session 10 (paper-faithful training with the NEW prior, 20% data).**
  User: train J/psi->mumu with the new prior, 20% data, preliminary simple
  run, method/loss as close to the paper as possible, and explicitly NOT to
  reference the archived past trainings. Wrote a fresh self-contained config
  configs/cms_Jpsi_newprior_paper_20pct.yaml (no extends): new prior
  cms_jpsi_mumu_mg5_8tev_mixed.hdf5 (94,880 ev), data window = the prior's
  fiducial window (pT>3, |eta|<2.4, mass [3.0369,3.1569], muon_pt_max 100
  junk guard), 20% cap via data_split caps 580000/72500/72500 (=725,000 of
  3,624,454), vanilla_swae loss (raw-coordinate SWD p=2, 1000 slices,
  standardize_raw_matching false), batch 20000, Adam 1e-3, seed 0, and the
  paper anchor warmup (nu_e=nu_d=50, epochs 1-80; then 0, epochs 81-300).
  Preflight PASSED (cache key 47c9d9573426ad4de16c in
  outputs/cms_Jpsi_new/.plot_cache; x 725,000 / z 94,880; MPS). Training
  launched in background (job bash-21) into
  outputs/cms_Jpsi_new/Jpsi_newprior_paper_20pct/; verified running normally
  (epoch 14/80, global step ~393/8700; train_loss 590@ep1 -> ~6.4@ep14;
  eval_loss 194.8@ep1 -> 11.4@ep10; checkpoints written). Left running, not
  monitored, per user request. Log: /tmp/jpsi_new_train.log.
  COMPLETED 2026-08-15 (exit 0, ~3h wall): 300/300 epochs, 8700/8700 steps.
  Best eval loss 7.167 @ epoch 290 (checkpoint best_model.pt); final epoch
  300 eval 7.380, train 3.255 (x-MSE 0.685, z_raw_swd 2.571). Anchor-warmup
  handoff clean: eval 8.04 @ep80 -> 7.17 best in core phase. encoder_diag at
  epoch 300: mean_z_ks = 0.017 (latent matches the new prior almost
  exactly — the raw-SWD objective converged, unlike the archived v3.8),
  z_mass_ks = 0.752, cycle_mass_ks = 0.527. Run dir keeps best_model.pt,
  best_z_prior.pt, best_reconstruction.pt, best_combined.pt,
  checkpoint_paper_anchor_warmup.pt, checkpoint_paper_core_swae.pt,
  checkpoint_final.pt, train_log.csv, history.json, status.json.
- **2026-08-15 — Session 11 (diagnose Jpsi_new density failure; goal round 1).**
  User: all `pos_*` plots from `outputs/cms_Jpsi_new/Jpsi_newprior_paper_20pct`
  look good, but the mass_density and pT_density plots are catastrophic; find
  why encoder-decoder and decoder-cycle closures fail even though the
  encoder/decoder training itself converged. Diagnosis (artifact-measured from
  `plots/paperstyle_loaded_model_outputs.npz` + source-verified in the loss and
  training code): (1) the vanilla objective is coordinate-level — raw
  per-component MSE + random linear-projection SWD — and has no term for the
  pair mass or Delta_eta/Delta_phi correlation; E(x) matches all eight z
  component distributions (KS <=0.014) and 1000 random linear projections
  (max KS ~0.03) while only 3.1% of E(x) is in the mass window (mass KS 0.760,
  mean 4.65 vs 3.097). The missing two-muon angular correlation (std
  Delta_eta 0.372 vs 0.207, Delta_phi 0.336 vs 0.217) is exactly what makes
  the J/psi mass; d(m)/d(Delta) ~ 13 GeV/rad at pT ~14 GeV. (2) The cycle
  x->z->x therefore has 13.8% in-window mass fraction and std 0.511 GeV while
  per-coordinate residuals are only 0.30-0.68 GeV and pair pT is good
  (KS 0.011). (3) D(z_prior) is never trained directly after epoch 80
  (`tau=rho=nu_d=0`; vanilla eval sets alt_x_loss=0), and the warmup anchor is
  only a cosine direction of muon-1 3-momentum; since only 3% of E(x) is on
  the mass shell, D(z_prior) extrapolates (5.6% in-window, mass std 1.369).
  (4) The low-pT collapse of the xspace pT density ratio is a prior support
  hole: MG5 cards use ptj>=10, so z_prior pair pT min is 9.70 GeV and 0.06%
  below 10 GeV vs data 2.2% below 10 GeV (min 5.18). (5) Checkpoint selection
  is also mass-blind: raw z-SWD ~6.8 dwarfs raw x-MSE ~0.3 in the vanilla
  score, and epoch-300 has slightly better mass closure than chosen epoch-290.
  Wrote `outputs/cms_Jpsi_new/Jpsi_newprior_paper_20pct/density_failure_analysis.md`
  and reproducible `scripts/diagnostics/diagnose_jpsi_new_density.py`; added root cause 5
  to §3.2. Causal probe (round 2): fine-tuned the frozen-decoder model's
  encoder for only 300 Adam steps (lr 1e-4, batch 8192, L=400) with
  `raw_SWD + 10*W1(mass)`; latent mass KS 0.759 -> 0.425, mean 4.64 -> 3.04,
  window fraction 3.0% -> 16.0%, while max component KS only rises 0.018 ->
  0.033. The frozen decoder's cycle mass worsens (KS 0.569 -> 0.748),
  confirming the second half: D is only trained against the old mass-broken
  E(x) distribution. Artifacts:
  `diagnostic_mass_probe_encoder.pt`, `mass_probe_metrics.json`.
  Decoder-only mirror probe (round 3, from untouched best_model, encoder
  frozen, 300 Adam steps): `raw_SWD(x,D(z)) + 10*W1(mass(x),mass(D(z)))`
  improves generator mass KS 0.688 -> 0.391, mean 2.814 -> 3.027, window
  fraction 5.7% -> 27.9%, component KS 0.044 -> 0.023, while cycle mass
  worsens 0.569 -> 0.708 (encoder still mass-broken). Artifacts:
  `diagnostic_mass_probe_decoder.pt`, `decoder_probe_metrics.json`.
  Joint probe (round 3): encoder+decoder trained together for 300 steps with
  `raw_SWD(z,E(x)) + raw_SWD(x,D(z)) + 10*[W1(mass z)+W1(mass x_from_z)]
  + MSE(x,D(E(x)))`. All three mass paths improve simultaneously: z mass KS
  0.759 -> 0.427, cycle mass KS 0.564 -> 0.479, generator mass KS
  0.692 -> 0.398; component and pair-pT closures stay intact. Artifact:
  `diagnostic_mass_probe_joint.pt`, `joint_probe_metrics.json`.
  Together the two probes prove both halves must be trained jointly:
  mass-aware E(x) matching + direct mass-aware D(z_prior) matching. Fix
  directions in the report: enable the existing mass/mass-kin/
  transverse/longitudinal OT terms, add direct D(z_prior) training (`tau>0`),
  close the prior low-pT support hole, and re-scale checkpoint selection.

- **2026-08-15 — Session 12 (prior low-pT hole fixed: ptj 10 -> 5 MG5 rebuild).**
  User: fix the prior first. Reran MG5 3.7.0 with the existing process dirs
  `jpsi_signal_test` / `dy_cont_test` after lowering `ptj` from 10.0 to 5.0
  (signal `run_05`, continuum `run_02`; committed cards
  `scripts/mg5_cards/*_run_card.dat` updated to ptj 5.0). Both runs completed
  normally: signal cross-section 107.2 +- 0.13 pb / 200,000 events;
  continuum 69.49 +- 0.15 pb / 50,000 events. Post-processing: LHE filter
  kept 191,979 signal (96.0%) and 14,445 continuum (28.9%); smeared with
  a=0.0128 (recalibrated on the ptj5 sample, old a=0.0127); muon-pT
  reweighted to the archived v3.9 reference cache and mixed 85/15 ->
  `data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5` (94,880 events; the old
  ptj=10 file is preserved unchanged). Measured (artifact-measured): pass
  rate 94.4%, mass mean/std 3.0972/31.0 MeV (E-based) vs data 3.0943/28.1;
  muon pT median 12.69 vs data 12.57; pair pT median 26.35 vs 26.03;
  **pair pT min 5.52 GeV vs data 5.13 (old prior 9.70)**; fraction 5-10 GeV
  0.88% vs data 2.24% (old 0.06%); joint [p0.5,p99.5] box coverage 97.8%.
  Support hole closed; remaining low-pair-pT density mismatch is a
  reweighting-target limitation (muon-pT marginal only), not an MG5 support
  hole. Runbook updated: docs/jpsi_prior_rebuild_runbook.md. Next step:
  implement the three-term annealed loss + mass guard on top of this prior.

- **2026-08-15 — Session 13 (project cleanup, no training-code changes).**
  Per user request: no mass terms will be added to the next loss; before that,
  the repository was simplified and reorganized. Changes (source-verified):
  * Moved prior tooling to `scripts/prior_build/` and current diagnostics to
    `scripts/diagnostics/`; moved the frozen v3.x diagnostics/report scripts
    and the DoubleElectron plot notebook to `scripts/legacy/`.
  * Deleted dead code: root `py.py` (generated parser table),
    `scripts/feature_ot_loss.py` (one-line re-export),
    `utilityFunctions/split_hdf5data.py` (no references).
  * Removed the legacy `ZLossFactory` and the `z_cycle` training mode from
    `scripts/cms_training.py` (now ~878 lines); unknown loss kinds now raise
    instead of silently falling back. Standard mode is the only supported
    training mode.
  * Removed the unused `OriginalOtusFeatureLossFactory` alias from `scripts/loss.py`.
  * Added `scripts/README.md`, `scripts/legacy/README.md`, `configs/README.md`;
    rewrote `README_MPS.md` to match the current layout; updated all script
    paths in docs/memory/config comments.
  * Ran pyflakes over scripts/tests and fixed all reported unused imports,
    unused locals, and malformed f-strings.
  * Full test suite: 98/98 OK; py_compile clean for all scripts/tests and
    moved tools; moved prior-build/diagnostic CLIs smoke-tested from their new
    locations.
  Next step (not done, per user): implement the three-term annealed loss for
  the next training round without adding mass-related terms.

- **2026-08-15 — Session 14 (three-term cosine loss, no mass terms).**
  User's next-round loss accepted as:
  `L_train = alpha(t)*(SWD(z,E(x)) + SWD(x,D(z))) + lambda_cycle*MSE(x,D(E(x)))`.
  Implemented (source-verified):
  * `scripts/loss.py`: `x_sim_loss` now has a vanilla branch symmetric to
    `z_prior_loss` (raw/standardized 8-vector SWD only). This fixes the old
    trap where enabling tau in vanilla mode silently used the full default
    component loss (~289 instead of ~1).
  * `scripts/cms_training.py`: stage coefficient schedules (`linear`,
    `cosine`, `constant`) with explicit `alpha` and `lambda_cycle` keys;
    `alpha` drives both `lamb` and `tau`. Legacy `lamb`/`tau`/`beta` keys
    remain backward compatible; mixing old and new names in one stage raises.
  * Training evaluates all three raw losses every batch; the scheduled
    coefficients only affect gradients. Logged `train_reference_loss` /
    `eval_reference_loss` = fixed alpha=lambda=1 sum
    `L_z + L_alt_x + L_x`, so changing the cosine schedule does not destroy
    comparability. `train_alt_x_loss_weighted`, `effective_alpha`, and
    `effective_lambda_cycle` are also logged.
  * Validation/checkpoint selection is schedule-independent: it uses fixed
    `loss.selection_score` weights (new config uses 1:1:1).
  * New config `configs/cms_Jpsi_newprior_ptj5_3term_cosine_20pct.yaml`:
    stage 1 = 80 epochs alpha=1, lambda_cycle=1, anchors 50/50; stage 2 =
    220 epochs alpha cosine 1.0 -> 0.25, lambda_cycle=1, anchors off.
    Dry-run passed with `--num-samples 10000`.
  * Tests: 103/103 OK (added schedule/alias/vanilla-x_sim tests). pyflakes
    clean. `scripts/plot_loss.py` now also plots the reference-loss curves.
  Not started: no training run launched.

- **2026-08-16 — Session 15 (Run D implementation).**
  Implemented the Run D loss extensions in `scripts/loss.py` without changing
  the model/data interfaces:
  * `pair_swd_weight` (default 0): optional pair-level sliced-Wasserstein in
    vanilla mode over standardized dimensionless features
    `[log m_ll, log pT_ll, y_ll, cos dphi, sin dphi]`, with configurable std
    floors (`pair_log_m_std_floor` 0.01, `pair_log_pt_std_floor` 0.05,
    `pair_y_std_floor` 0.20). Applied to both `z_prior_loss` and `x_sim_loss`.
    No J/psi mass constant is used, so it can later run on J/psi/Z/Z′.
  * `cycle_mass_huber_weight` / `cycle_mass_huber_delta` (default 0 / 0.02):
    optional per-event cycle term `Huber((m_reco-m_true)/m_true, delta)` added
    to vanilla `x_reco_loss`.
  * New components logged: `z_pair_swd`, `x_pair_swd`,
    `x_reco_mass_huber_raw/weighted`; HistoryLogger fields extended.
  * Run D config created:
    `configs/cms_Jpsi_newprior_ptj5_rund_pairswd_cyclemass_20pct.yaml`
    (pair_swd_weight 0.5, cycle_mass_huber_weight 50.0, delta 0.02, same
    three-term cosine stages). Dry-run passed with `--num-samples 10000`.
  * Tests 106/106 OK; pyflakes clean.
  * Memory file trimmed: superseded §7 F1-F3 roadmap, stale §8 environment
    notes, and outdated §10 quick-reference entries were replaced with current
    status (trim note dated 2026-08-16).
  Not started: no Run D training run launched.


- **2026-08-16 — Session 16 (Run D finished + SOTA exploration).**
  Run D completed. Best epoch 270; test mass W1 0.0258 GeV / KS 0.221.
  Plots written to `plots_best/`, loss curve to `loss_curve.png`, eval sets
  `eval_best/`, `eval_final/`, `eval_reco_best/`, `eval_z_best/`.
  See Session 16 notes in the previous edit for the detailed numbers.

- **2026-08-16 — Session 17 (Tier A / Run E implementation).**
  Created `scripts_sota/` without modifying the Run D pipeline:
  * `ot.py`: 14D cylindrical physics ground cost, debiased Sinkhorn
    divergence, entropic Monge-gap proxy with barycentric projection.
  * `max_swd.py`: adversarially learned slicing directions (max-SW) with
    orthogonality penalty and detached direction ascent.
  * `cylindrical_flow.py`: `CylindricalResidualFlowMap` with `flow_steps`
    stochastic transforms in `(logpT, eta, phi)`, Gaussian core +
    Student-t tail, S^1 phi transport, mass-shell energy reconstruction.
  * `evaluation.py`: C2ST logistic AUC, fixed-z stochasticity, cycle and
    nearest-neighbour pseudo-pair coverage diagnostics.
  * `selection.py`: mass-shape hard gates (W1/KS/window fraction/mean shift)
    and `gated_score` (score=inf when gates fail).
  * `sota_loss.py`: `SotaLossFactory` wraps current J/psi loss and adds
    Sinkhorn + max-SW terms (Monge-gap optional); `sota` config block.
  * `trainer.py`, `run_e.py`: Run E training loop with per-stage noise
    multipliers, cosine LR, gradient clipping, mass-gated checkpointing,
    stage-best restore, and SOTA evaluation outputs.
  * Config: `configs_sota/cms_Jpsi_ptj5_runE_tierA.yaml` (Run D terms +
    Tier A, 60+100+140 stages, full schedule not launched).
  * Paper skeleton: `paper/main.tex` (compiles with local pdflatex),
    `paper/references.bib`, `paper/README.md`.
  * Tests: added `tests/test_sota.py`; full suite 118/118 OK. Run E
    `--dry-run` and `--smoke` on CPU and MPS both complete; evaluation
    report path exercised on a smoke model.
  Not launched: full Run E training.

- **2026-08-16 — Session 18 (Run E smoke ablations + full-config tuning).**
  Six one-epoch/10k smoke ablations completed under
  `outputs/cms_Jpsi_sota/ablations/`:
  control, sinkhorn-off, maxswd-off, flow1, flow4, monge-on.
  Findings (smoke only, not physics conclusions):
  * Monge-gap proxy is currently unstable (train loss up to ~2700, pre-clip
    gradient ~7e7). Keep disabled for the first full Run E; the component
    needs a scale/regularization redesign before an ablation.
  * Sinkhorn at weight 0.5 dominated gradients (pre-clip grad 6k-235k vs
    ~0.3k-8k with Sinkhorn off). Full config reduced to weight 0.10,
    regularization 0.10, max_iter 50, max_batch 512.
  * max-SW on improved smoke mass/latent KS and C2ST; full config reduced to
    weight 0.25, direction lr 5e-4 to avoid direction jitter.
  * flow1 had the best smoke C2ST/cycle coverage, flow4 best eval score but
    worse C2ST/reco KS; keep flow_steps=2 for the first full run as the
    center of the ablation, revisit flow count with 10-epoch probes.
  * Full config now uses eval_every=5 and gate validation subset 8192.
  * Trainer now logs all `latest_components` (Sinkhorn/max-SW/Monge weighted
    components) into `history.json`.
  Full Run E command prepared; not launched.

- **2026-08-16 — Session 19 (power outage + resume support).**
  Full Run E (`runE_full_20260816_213140`) was interrupted by power loss.
  Recoverable state: `last_model.pt` = global epoch 15, stage1 epoch 15/60;
  `history.json` contains rows through epoch 17 but epochs 16-17 have no saved
  weights. No `best_model.pt`/stage-best yet because the mass gates correctly
  failed during early deterministic warmup (epoch15 gates: sim W1 0.0282,
  KS 0.218, window 0.786, mean shift -0.028). Implemented `--resume`:
  run_e.py loads `last_model.pt`, restores model + Adam state + SOTA max-SW
  state when present, truncates history to checkpoint epoch, rebuilds loaders,
  advances cosine scheduler to the correct local epoch, and continues inside
  the same stage. Resume is statistically equivalent but not bitwise
  identical (loaders/RNG streams rebuilt). Tested resume logic end-to-end on a
  /tmp copy; full resume not launched.

- **2026-08-17 — Session 20 (stage-2 instability fix + clean restart).**
  Stage-2 explosion root cause confirmed as unbounded neural-OT scales once
  core noise turns on. Fixes applied:
  * max-SW: p=2 -> p=1, `distance_clamp=25`, direction gradient clip 1.0,
    update_every=5, direction lr 1e-4.
  * Sinkhorn: added `log_scale: true` (`log1p` before weighting), so raw
    Sinkhorn ~1000 maps to ~7 instead of ~100; raw value still logged.
  * stage2 `core_noise_multiplier` 1.0 -> 0.25.
  * `require_stage_gate_pass: true` and trainer now raises if a stage ends
    without ever passing the mass gates.
  * `run_e.py --resume-checkpoint` added; resume now keeps the CLI config
    (so repaired loss settings apply) and skips completed stages correctly.
  * Failed `last_model.pt` preserved as
    `last_model_stage2_exploded_epoch105.pt`.
  Clean restart launched from `best_rune_stage1_deterministic_warmup.pt`
  (epoch 60) into stage2. PID/log: `runE_full_20260816_213140_resume_v5.*`.
  Early stage2 trajectory stable: epoch 1-6 train loss 6.2 -> 12.5,
  max-SW weighted < 0.16, log-scaled Sinkhorn weighted < 0.38, pre-clip
  gradients O(100-1100). No explosion so far.

- **2026-08-17 — Session 21 (Run E final results).**
  Run E completed 300/300 epochs. Best checkpoint global epoch 285
  (`best_model.pt`, stage3 tail-flow polish). Final plots written:
  `plots_paperstyle_final_narrow/`, `plots_paperstyle_final_wide/`,
  `run_e_loss_curve_final.png`. Final narrow-window test metrics:
  * D(z) mass W1 = 0.001047 GeV, KS = 0.01655, mean 3.09510 vs 3.09452,
    std 29.02 vs 28.10 MeV, peak density ratio 0.994.
  * D(E(x)) mass W1 = 0.00360, KS = 0.0502, std 24.36 MeV (slightly
    underdispersed), peak density ratio 1.20.
  * E(x) latent mass KS = 0.0645, std 40.4 vs prior 31.1 MeV; latent
    component KS <= 0.018. Latent mass remains the weakest closed quantity.
  * Pair pT KS x vs D(z) = 0.0834.
  * C2ST AUC = 0.5169 +- 0.0085 (near indistinguishable).
  * Fixed-z decoder stochasticity: median per-event mass width 9.2 MeV.
  * Coverage proxies undercovered: cycle 95% coverage 0.754, pseudo-pair
    0.490 (paired truth unavailable; interpret as diagnostic only).
  * Wide [2.6,3.5] plot: D(z) mass W1 0.108 GeV / KS 0.312 because the
    prior has no sidebands; this is expected support mismatch, not a
    model-divergence signature.
  Conclusion: generator z->x meets the Run E expectations; latent E(x) mass
  width and conditional coverage remain the next targets.

- **2026-08-19 — Session 22 (Run F implementation + launch).**
  Implemented Tier B F1+F2 as a merged bidirectional minibatch-OT flow
  matching / stochastic-interpolant model:
  * `scripts_sota/flow_matching.py`: cylindrical velocity fields for
    decoder z->x and encoder x->z; entropic-OT minibatch coupling (P for
    decoder, P^T for encoder); Brownian-bridge-style noisy linear
    interpolant on non-phi coordinates; S^1 phi transport; Euler inference
    conditioned on per-event eps.
  * `scripts_sota/trainer.py`: `flow_matching_loss` hook and stage
    `flow_weight`; `train_flow_loss` logged in history.
  * `scripts_sota/run_e.py` and plot scripts dispatch on
    `model.class: flow_matching`.
  * Config: `configs_sota/cms_Jpsi_ptj5_runF_flowmatch.yaml`
    (same Tier A losses/gates as Run E; 60+100+140 stages).
  * Tests: `tests/test_flow_matching.py`; full suite 122/122 OK; pyflakes
    clean. Run F smoke (10k, MPS) and full dry-run passed.
  Launched full Run F in background:
  * run: `outputs/cms_Jpsi_sota/runF_full`
  * log: `outputs/cms_Jpsi_sota/runF_full.log`
  * pid: `outputs/cms_Jpsi_sota/runF_full.pid`
  * epoch 1 started normally; train_flow_loss 0.334.
  Not complete at end of session.

- **2026-08-20 — Session 23 (Run F completed + plotted).**
  Run F (bidirectional OT-flow matching) completed 300/300 epochs.
  Best checkpoint global epoch 275 = stage3 ep115
  (`best_model.pt`). Plots: `runF_full/runF_loss_curve.png`,
  `plots_paperstyle_final_narrow/`, `plots_paperstyle_final_wide/`,
  `plots_paperstyle_stage2_best/`.
  Narrow test results:
  * best_model D(z) mass W1 = 0.00537 GeV, KS = 0.0542, std 40.6 vs
    CMS 28.1 MeV, peak density ratio 0.923.
  * stage2-best D(z) mass W1 = 0.00449 GeV, KS = 0.0575, std 33.0 MeV,
    but latent mass KS 0.132.
  * cycle D(E(x)) is good: W1 0.00258, KS 0.0366, std 27.96 MeV.
  * C2ST AUC = 0.5112 +- 0.0097; fixed-z mass stochasticity median
    3.0 MeV (narrower than Run E's 9.2 MeV).
  * Coverage proxies worsened (cycle 95% 0.461, pseudo 0.170),
    consistent with underdispersed conditional draws.
  Verdict: Run F does NOT beat Run E on the primary generator mass
  closure (Run E W1 0.00105 / KS 0.0165). It is slightly better on C2ST
  and cycle closure, but worse on D(z) mass width and conditional
  coverage. Keep Run E as the main result; treat Run F as an ablation /
  future-work direction, or retrain with more integration steps and a
  larger sigma before making claims.

- **2026-08-22 — Run D prepared with the new showered inclusive Z prior.**
  Verified
  `data/cms_dymumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_70_110_1M.hdf5`:
  1,000,000 finite float32 events, post-Pythia8 stable muons with lepton QED
  FSR, merged MG5 0/1-parton CKKW-L production, and 100% pass of the Run D
  Z selection. Against the full-scale Run C CMS Z test split, the new prior
  improves mass KS 0.174 -> 0.086 and pair-pT KS 0.212 -> 0.069 relative to
  the old fixed-order DY+1j prior (pair-pT W1 is worse, 1.74 -> 2.27 GeV,
  because the new showered prior has a harder extreme tail). Added
  `configs_joint/cms_Joint_runD.yaml` inheriting Run C full-scale and changing
  only the Z prior, `scripts_joint/run_d.py`, contract tests, and docs. Dry-run
  and four-stage CPU smoke passed; focused joint tests 11/11.

- **2026-08-22 — Run D completed, plotted, and evaluated on Upsilon.**
  Full-scale Run D completed 432 epochs. The accepted `best_model.pt` is global
  epoch 70 (stage 1 deterministic identity, core/tail noise multipliers 0/0),
  with all J/psi and Z gates passing. Direct z->x test metrics: J/psi mass
  W1/KS = 0.000789 GeV / 0.0138 and pair-pT KS = 0.0675; Z mass W1/KS =
  0.298 GeV / 0.0175 and pair-pT KS = 0.0300. Generated 19 all-element
  artifacts, 22 paper-style plots, and log/linear loss curves under
  `outputs/cms_Joint/Run_D/`.
  Decoded all 1,000,000 events from the matched showered inclusive Upsilon
  0/1j prior and generated raw/decoded peak comparisons plus 19 direct-z->x
  quantitative plots under `Run_D/upsilon_transfer/`. Equal-count Upsilon
  metrics (16,600 per distribution): mass W1/KS = 0.264 GeV / 0.185,
  pair-pT W1/KS = 1.420 GeV / 0.318, linear/MLP C2ST AUC = 0.759/0.802.
  State mean response biases are -28.0/-30.7/-32.4 MeV for 1S/2S/3S. The
  deterministic selected checkpoint preserves overly narrow prior peaks and
  does not improve the inclusive Upsilon mass comparison (raw W1/KS
  0.206/0.174; decoded 0.213/0.179 in [8.5,11.2] GeV). This is recorded as
  `post_unblinding_heldout_transfer`, not a new zero-shot claim. Fixed the
  existing Upsilon plotter's hard-coded Run C title and zero-shot provenance;
  it now derives the run label/scope from decode metadata.

- **2026-08-23 — Run E full-pass epoch implementation.**
  Added `loaders.epoch_definition: full_pass` to the joint trainer and created
  `configs_joint/cms_Joint_runE.yaml` / `scripts_joint/run_e.py`. A Run E epoch
  now deterministically visits every J/psi x/z and Z x/z training row exactly
  once, partitions unequal datasets into the same number of balanced batches,
  and records per-partition event counts plus optimizer updates in history.
  Unequal-cardinality Wasserstein/SWD paths now integrate the complete empirical
  quantile functions instead of truncating the larger batch. Equal-cardinality
  behavior is preserved. Focused correctness/joint/loss tests pass (63 tests,
  one optional skip); full discovery passes all relevant tests but still has
  two pre-existing environment/data errors: Windows h5py temporary-file
  creation and the absent legacy `cms_jpsi_mumu_mg5_8tev_1M.hdf5` fixture.
  CPU dry-run and four-stage full-pass smoke passed without opening Upsilon.
  With Run D's measured partition sizes and batch 24,576, production Run E is
  172 updates per epoch versus Run D's four; the inherited 432 epochs would be
  74,304 updates (43x Run D), so production was not launched automatically.

- **2026-09-01 — Joint Run F live plateau review (distinct from the older SOTA Run F).**
  artifact-measured: Run_F is active around epoch 235/432; stage-3 mean loss
  decreased from 0.8128 (193–202) to 0.7721 (223–234), while J/psi latent mass
  KS remains failing (0.5007 at epoch 232; gate 0.12). All Z gates pass.
  Fixed-checkpoint CPU validation probes show reducing encoder noise improves
  mass W1 but worsens KS because a roughly -8.8 MeV mass bias persists; this
  is not a validated one-setting fix. proposal: bounded continuation to about
  epoch 260, then use sustained validation improvement to decide whether to
  adjust; no automatic monitoring/stop was installed and training is unchanged.
  source-verified + artifact-measured: resume resets the stage-best score;
  saved stage-2 checkpoint is epoch 192 despite a better historical score at
  epoch 112. Also joint_model.py is deleted in the working tree; resolve before
  restarting. Full findings and retained evidence:
  `outputs/cms_Joint/Run_F/diagnostics/plateau_review.md`.
