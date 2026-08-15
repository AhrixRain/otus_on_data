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
  scripts/preflight.py, scripts/verify_v38_gradients.py,
  tests/test_v38_vanilla.py. v3.9 (2026-08-15) added:
  cms_data.filter_theory_prior + theory_prior_selection cache metadata +
  optional muon_pt_max in the muon loader; scripts/prior_kinematics.py,
  scripts/data_kinematics.py (E1 dumps); tests/test_v39_f1.py (10 tests).

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
scripts/prior_kinematics.py + scripts/data_kinematics.py)

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
2026-08-14 via scripts/jpsi_mismatch_check.py + scripts/jpsi_f32_check.py,
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
scripts/zee_health_check.py, cache 9bb63a1e38cf0cd9b9c2 version 1;
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

## 7. Proposed fix roadmap (status: E1 done, F1 implemented as v3.9, running)

Paper-grounded, one-factor-at-a-time:
- **E1 (precondition) — DONE 2026-08-15.** Prior + data kinematics dumped
  (§4.5; scripts/prior_kinematics.py, scripts/data_kinematics.py).
- **F1 (primary) — composition-matched prior.** REVISED by E1: the inclusive
  file is pure 2->1 (pair pT = 0) and unusable; the implemented variant is the
  committed v3.9 pilot: restricted-decoder signal-region scope
  [3.0369, 3.1569] + the signal MG5 prior filtered through the
  trigger-equivalent selection (theory_prior_selection: pT>3, |eta|<2.4, mass
  window — implemented in scripts/cms_data.filter_theory_prior) + a
  muon_pt_max 100 GeV junk-tail guard (optional key in the muon loader).
  Prior survives at 0.60% (5,995 events) — a statistics limitation to watch.
  Config: configs/cms_JpsiDoubleMuons_v3.9_F1_restricted.yaml
  (run_name Jpsi_v3.9_F1_restricted_ptmax100, launched 2026-08-15).
  Residual known mismatch: prior pair-pT median ~10 GeV vs data ~26 GeV.
- **New J/psi prior REBUILT with MadGraph (2026-08-14, goal round 1).**
  data/cms_jpsi_mumu_mg5_8tev_mixed.hdf5 (FDL/zData, 94,880 events = 80,648
  signal + 14,232 continuum, frac-signal 0.85) generated end-to-end ON THIS MAC
  with MG5_aMC 3.7.0 + rebuilt sm_onia(-c_mass) model (recipe:
  docs/sm_onia_rebuild.md; massive muons MM=0.105658; gJpsi=0.01; MASS 443 =
  3.0969; DECAY 443 = 9.29e-05). Components: p p > jpsiv j (200k events,
  55.92 pb) + p p > mu+mu- j with stock sm_mumass (50k events, 36.24 pb),
  cuts ptl 3.0 / etal 2.5 / mmll [2.9,3.3] / ptj 10 / cut_decays=True,
  NNPDF31_lo_as_0130 (lhaid 315200). Post-chain (scripts/): lhe_to_prior_hdf5
  (96.1% signal pass rate), smear_prior (a=0.0127), reweight_prior
  (binned-density muon-pT reweight, max-weight 20), mix_prior (85/15). FINAL
  METRICS vs signal-window data (3,624,454 ev): mass mean 3.09697 vs 3.09428,
  std 30.8 vs 28.1 MeV; muon pT median 13.30 vs 12.57; pair pT median 26.60 vs
  26.03; pass rate 94.3% (was 0.60%); joint support coverage 97.4% (was
  69.8%); continuum 15% (was 0%). Production runbook:
  docs/jpsi_prior_rebuild_runbook.md; cards in scripts/mg5_cards/; unit tests
  tests/test_prior_build.py (7 tests). macOS fixes discovered: brew gcc
  gfortran 16.1.0, conda lhapdf 6.5.6 + PDF grid, DYLD_LIBRARY_PATH /
  -Wl,-rpath for libLHAPDF.dylib (the 'Reason:' survey crash), use_syst=False
  (NNPDF23 errorset missing). Config switch for training: point
  theory_prior_file at the new file (cache auto-invalidates).
- **v3.10 (IMPLEMENTED 2026-08-15, not launched — the user runs it later)** —
  staged + restricted: configs/cms_JpsiDoubleMuons_v3.10_staged_restricted.yaml
  extends the v3.5 staged config with the v3.9 data scope (signal region +
  junk cut + trigger-matched prior) AND the re-weighted checkpoint-selection
  score (x_sim 5.0 / z_prior 1.0 / x_reco 5.0) so the cycle is no longer
  masked (§6.3 finding). Rationale: v3.5's loss is the only objective that
  produced a good generator because its x_sim terms constrain D(z~prior)
  directly (the vanilla hole, §3.2 cause #4). Dry-run validated. THIS IS THE
  RECOMMENDED FIX RUN.
- **F2 — paper anchor warmup:** beta_E=beta_D=50 -> 0 (first ~80 of 300 epochs)
  on the mu- 3-momentum (prevents charge-inversion solutions). Not started.
- **F3 — lambda treatment:** lambda scan {0.1, 1, 10} + upward annealing;
  consider p=1 for the latent SWD (tail-robust); optional gradient clipping as
  a guard. Not started.
- **Fallback — paper's restricted decoder (§6.3.2):** subsumed into the v3.9
  pilot (signal-region scope). The 1_S/P_D(S) reweighting itself is NOT
  implemented — v3.9 uses the plain restricted scope instead.
- Sequencing: v3.9 first as a single controlled change against the frozen v3.8
  config; then F2, F3 separately; never bundle.

## 8. Environment & tooling constraints (from the last DSH sessions)

- **Session 4 (2026-08-15) — conda → .venv.** The project environment was
  converted to a project-local virtual environment `.venv/` (Python 3.12.13
  arm64 via homebrew) holding the `cms` environment, installed from
  `requirements-cms-mps.txt` (uproot/awkward/h5py/numpy/scipy/torch-MPS/
  jupyter etc.). All runbook commands and script docstrings use
  `.venv/bin/python`; `.venv/` is git-ignored. source-verified.
- **Session 5 (2026-08-15) — machine moved; conda cms env available.** On the
  current machine /Users/ahrimarin/Desktop/otus_on_data the `.venv` does not
  exist, but miniforge provides the `cms` conda env:
  /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python (Python 3.10.20,
  numpy 2.2.5, h5py 3.16.0, scipy, torch 2.12.1 MPS available, uproot 5.7.4,
  awkward 2.9.0). Use that interpreter for all runbook commands on this
  machine. source-verified (imports + device report).

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
- As of 2026-08-14 the historical J/psi split caches live at
  outputs/cms_JpsiDoubleMuons/archive/.plot_cache (scripts
  jpsi_mismatch_check/jpsi_f32_check/jpsi_new_prior_comparison point there);
  the new training's caches will be created under outputs/cms_Jpsi_new/.plot_cache.
- Citation style for findings: file:line (code), notebook cell (ipynb),
  document+section (PDFs), full artifact path (plots/NPZ).

## 10. Quick reference

- Main staged config: configs/cms_JpsiDoubleMuons_mps.yaml (run: Jpsi_v3.5).
- Mass ablation: configs/cms_JpsiDoubleMuons_Jpsi_v3.6A_no_explicit_mass.yaml.
- Vanilla-paper: configs/cms_JpsiDoubleMuons_v3.8_vanilla_paper.yaml
  (runbook: docs/Jpsi_v3.8_vanilla_paper_runbook.md; production command NOT
  to be re-run without authorization).
- v3.9 F1-restricted pilot: configs/cms_JpsiDoubleMuons_v3.9_F1_restricted.yaml
  (run_name Jpsi_v3.9_F1_restricted_ptmax100; extends v3.8; signal-region
  scope + trigger-matched prior + muon_pt_max 100).
- v3.10 staged+restricted draft: configs/cms_JpsiDoubleMuons_v3.10_staged_restricted.yaml
  (extends v3.5; validated, not launched).
- v3.9 out-of-scope diagnostic: configs/cms_JpsiDoubleMuons_v3.9_out_of_scope_eval.yaml
  (full-window data fed to the v3.9 model; eval.py's CLI config drives the
  eval data selection — source-verified in cms_model.load_model_from_checkpoint).
- **Runbook for the deferred runs: docs/Jpsi_F1_fix_runbook.md** (exact
  commands for v3.10/v3.9 training, evaluation, baselines, guardrails).
- E1 kinematics dumps: scripts/prior_kinematics.py, scripts/data_kinematics.py
  (see §4.5 for the measured numbers).
- Verdict tables: scripts/eval_verdict.py --eval-dir <run>/eval (W1/KS +
  shape stats per path; baselines in §4.3b).
- Generator-quality check: scripts/sample_generator.py (decodes the full
  filtered prior K times with fresh noise; needed for v3.9 whose z_test is
  only 600 events vs 100k for v3.5/v3.8).
- CMS-vs-MG5 mismatch diagnostics (2026-08-14): scripts/jpsi_mismatch_check.py
  (composition/support/pass-rate asymmetry; §4.7) and scripts/jpsi_f32_check.py
  (float32 z-mass noise; §4.7). Both read the version-3 .plot_cache, no ROOT
  rescan. Z->ee counterpart: scripts/zee_health_check.py (mass/kinematics/
  support/barrel-endcap scale split; §4.8).
- Data vs priors comparison figure (2026-08-14): scripts/jpsi_new_prior_comparison.py
  -> experiments/cms_Jpsi_ee/jpsi_new_prior_vs_cms_mumu.png (3x2 panels: mass /
  mass zoom / muon pT / pair pT / |eta| / summary table; CMS signal-window data
  vs old 1M delta prior vs new 94,880 mixed prior, training-exact x-space mass
  convention). Same-folder sibling of the old jpsi_prior_vs_cms_mumu.png.
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
  E1 facts. Wrote scripts/prior_kinematics.py, scripts/data_kinematics.py,
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
  does not eliminate it. Wrote scripts/jpsi_mismatch_check.py and
  scripts/jpsi_f32_check.py; findings in §4.7. Then, at the user's request,
  ran the same methodology on the Z->ee pair: healthy on every dimension
  (prior is a real 4-GeV-wide lineshape at the pole, kinematics matched at
  the few-% level, 93.4% support coverage, 100% pass rate, converged v4/v5
  runs). One real caveat found: the skim's endcap electron energy scale is
  +1.9% high (endcap-only mode 92.94 vs pole 91.19; barrel-only 91.18) —
  a skim-level 2012 ECAL calibration artifact, absorbable, not an OTUS bug.
  Wrote scripts/zee_health_check.py; findings in §4.8.
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
  scripts/jpsi_new_prior_comparison.py renders a 3x2 panel figure to
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
  98/98 OK, and scripts/jpsi_new_prior_comparison.py re-run end-to-end
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
