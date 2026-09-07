# Persistent Memory — OTUS on CMS Open Data

> The project's durable record. **Read §1 first; it says where things stand
> today.** Everything below it is reference material you can reach for as
> needed. Last restructured 2026-09-04 (Session 24) — content preserved, order
> and navigation rewritten; the pre-restructure file is `memory.md.bak.20260904`.

| § | Section | Read it when |
|---|---|---|
| 0 | How to use this file | Before writing to it |
| 1 | **Status at a glance** | **First, every session** |
| 2 | What this project is | You are new to the project |
| 3 | The three experimental programs | A run name is ambiguous |
| 4 | Established findings | You need the diagnosis so far |
| 5 | Key measured numbers | You need a number to compare against |
| 6 | Open questions | Choosing what to do next |
| 7 | Current plan and next experiment | Starting work |
| 8 | Paper reading notes | Relating the work to arXiv:2101.08944 |
| 9 | Where things live | Looking for a file or a command |
| 10 | Environment, conventions and guardrails | Before running anything |
| 11 | Session log | Reconstructing how something happened |

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
- **Keep the working tree honest.** This file is tracked as of commit 530dfee;
  commit your edits along with the work they describe.
- Never delete the sections; extend them. Trim only with a dated note.
- **Name runs by program.** "Run E" alone is ambiguous — there is a Run E in
  both the SOTA program and the joint program. Always write "SOTA Run E" or
  "joint Run E". See §3.

## 1. Status at a glance

**2026-09-07 update — joint Run E final-checkpoint Upsilon test reproduced.**
artifact-measured: legacy-prior decoding with last_model.pt (epoch 432,
noise 1/1) gives matched mass W1 identity/model/floor
0.23668/0.09698/0.01295 GeV, gauge 0.376, versus best_model gauge 1.183.
But pair-pT W1 worsens to 1.71378 (identity 1.36373, floor 0.11627,
gauge 1.281); window retention drops from 98.36% to 87.87%, and the decoded
state widths become 480–537 MeV, washing out the peaks. This is an inclusive
mass improvement, not demonstrated Upsilon response closure. Report:
`outputs/cms_Joint/Run_E/upsilon_transfer_legacy_last_model_20260907/REPORT.md`.

**2026-09-07 update — joint Run E legacy Upsilon test reproduced.**
artifact-measured: a fresh one-million-event decode of the legacy prior with
the selected deterministic epoch-65 checkpoint reproduces the earlier mass
metrics to numerical precision. Matched 8,300-event mass W1
identity/model/floor = 0.23668/0.27764/0.01295 GeV, gauge 1.183; pair-pT KS
0.33217/0.22000/0.01423, gauge 0.647. Momentum improves, mass degrades.
Report: `outputs/cms_Joint/Run_E/upsilon_transfer_legacy_20260907/REPORT.md`.

**2026-09-07 update — joint unifiedP1 Upsilon evaluation completed.**
artifact-measured: the frozen epoch-432 fallback checkpoint was evaluated on
both the legacy prior and 17,802 correctly materialized new unified-prior
events. On matched 8,300-event comparisons the new prior's mass W1 increases
from identity 0.18097 to decoded 0.28268 GeV, with floor 0.01295 and gauge
1.605; legacy increases from 0.23668 to 0.35483, gauge 1.528. The dominant
0–5 GeV pair-pT bin degrades; smaller higher-pT bins have mixed results.
artifact-measured: the checkpoint contains hard gates and a 1e6 failure
penalty despite the current config requesting nonblocking 35/65 selection.
This is an evaluation of the saved fallback, not validation of the intended
unifiedP1 training policy. Full report:
`outputs/cms_Joint/unifiedP1/upsilon_transfer_comparison/REPORT.md`.

*Updated 2026-09-04 (Session 24).*

**Best accepted result: joint Run E** (`outputs/cms_Joint/Run_E/`), one shared
mass-blind response model closing J/psi and Z simultaneously.

| region | direction | mass W1 [GeV] | mass KS | pair-pT KS | C2ST (MLP) |
|---|---|---|---|---|---|
| J/psi | z->x simulate | 0.00088 | 0.017 | 0.052 | 0.62 |
| J/psi | x->z unfold | 0.00177 | 0.037 | 0.022 | 0.65 |
| Z | z->x simulate | 0.086 | 0.013 | 0.006 | 0.51 |
| Z | x->z unfold | 0.053 | 0.007 | 0.005 | 0.51 |

**Read that table with two caveats.**
1. The selected checkpoint is **stage-1 deterministic** (`global_epoch` 65,
   noise multipliers 0/0) — as it also was for joint Run D. The stochastic
   stages have never won selection, so the accepted model is close to a
   deterministic map, not a demonstrated stochastic detector response.
2. Its J/psi prior had been hand-smeared to 26.2 MeV against 28.1 MeV data
   (§5.9), so the encoder only had to remove ~7% of the mass width. Whether
   the model can actually deconvolve resolution is **untested**.

**Held-out transfer is the honest weak point.** Decoding the Upsilon prior with
frozen joint Run E gives mass W1 0.28 GeV / KS 0.19, C2ST 0.81, and per-state
mass biases of -28 to -53 MeV. The model does not yet generalize to an unseen
resonance. This is `post_unblinding_heldout_transfer`, not a zero-shot claim.

**Nothing is training.** M1 (`AB_narrow_split`) completed 2026-09-05 with the
hypothesis falsified: 0 of 35 gate-passing validations, J/psi latent gauge
minimum 0.972, never below the 0.898 benchmark. The decoder was unaffected
(gauge 0.098) and the Z region unfolded to 0.052. Details in section 7.5; the
next branch is M2 or M5. Historic context follows.

**Step 3 is DONE (2026-09-07, Session 30). The per-event test has been run on
our own model, and it is the clearest result the programme has.** Section 7
step 3 had been the next action since Session 25. `configs_joint/cms_Joint_ppzee.yaml`
now trains our Run E architecture unpaired on the upstream OTUS ppzee benchmark
in under 16 minutes and scores it against each event's withheld truth partner.
artifact-measured on all 160,000 held-out pairs, `residual_rms_vs_identity`:

| | |
|---|---|
| oracle floor - best correction fitted ON the pairing | 0.8099 |
| **our encoder** | **0.9843** |
| identity, hand back the detector event | 1.0000 |
| upstream OTUS encoder, same 160,000 events | 3.3304 |

Read those four numbers together, never one alone.
1. **Our encoder is 3.4x closer to the truth partner than the upstream one**,
   because ours is a residual flow bounded near the identity and theirs is an
   unconstrained MLP. Section 7.4's claim that the per-event failure is "a
   property of this family of objectives" is therefore **too strong**; the
   upstream 3.33 is substantially architectural.
2. **It still did not invert the response.** The encoder removed 85% of the
   detector's global mass bias and made the per-event scatter 2.0% *worse*;
   subtracting a single constant would score 0.9647, better than the model.
   The whole gain is a global scale correction.
3. **Only ~19% of ppzee's resolution is recoverable at all** (that is the 0.81
   oracle), so "below 1.0" is a much weaker statement than it sounds. All four
   leakage checks pass (`outputs/cms_Joint/ppzee/paired_leakage_audit.json`).
4. **The decoder is the programme's strongest result**: it manufactures the
   entire recoil spectrum from a truth with identically zero pair pT - pair-pT
   gauge 0.000, mass gauge 0.058, C2ST 0.510.

Full detail, the pre-registered predictions and the bench recommendation are in
section 11, Session 30. Standing claim, now measured where paired truth exists:
**an OT objective on marginals buys a global scale correction, not a per-event
inverse.**

**Superseded status line:** joint `AB_narrow_split` (M1), launched 2026-09-04 23:16,
~133 s/epoch, so stage A (72 epochs) lands about 02:00 and the full 160 epochs
about 05:15 on 2026-09-05. It is the narrow-prior arm with the encoder and
decoder noise schedules split: encoder deterministic, decoder at 1.0 from epoch
one. Read-out is `latent_mass_ks_vs_identity`; the completed narrow arm never
went below **0.898** at any of its 35 validations, so a sustained excursion
below that is the signal. Details in section 7.5.

**Historic, for contrast:** joint Run F was stopped at global epoch 238/432 and
removed by the user on 2026-09-04; it had **0 gate-passing validations out of
51**, failing `latent_mass_ks` at every one. Its only controlled change was the
14.7 MeV showered J/psi prior.

**ANSWERED 2026-09-04 (Session 25) — the A/B ran and the answer is yes.**
The J/psi results depended on the pre-smeared prior. Smeared arm: 34/35
gate-passing validations; narrow arm: 0/35; only the prior file differed.
And the smeared arm's pass is not evidence of unfolding: the trivial identity
map (z~ = x, no model at all) passes every strict J/psi latent target on the
smeared prior and fails all of them on the narrow one. Full numbers in §7.2.
Read the Run D/E J/psi rows above as distribution matching against a
pseudo-truth prior that had already been given the detector resolution —
not as demonstrated deconvolution.

**Known blockers / debts.**
- The paper skeleton `paper/main.tex` is still the single-region SOTA Tier A
  draft and predates the entire joint program.
- Encoder and decoder share one noise schedule, which is physically backwards
  (the decoder should add resolution, the encoder remove it). Splitting them is
  the leading method fix after the A/B — see §6.
- `scripts_joint/_retired_launchers/` is pending deletion.

## 2. What this project is

- **Goal.** Apply **OTUS** (Optimal-Transport-based Unfolding and Simulation, a
  sliced-Wasserstein autoencoder) to **CMS 2012 Open Data dimuon resonances**,
  learning a detector response between MadGraph5 truth-level four-vectors (z)
  and CMS detector-level four-vectors (x) **without per-event paired
  simulation**. The decoder z->x simulates; the encoder x->z unfolds.
- **Why it is hard here.** The paper's Z->ee case had a prior and data of
  identical composition. J/psi does not: the CMS window is signal plus
  continuum, the truth prior is a near-delta mass shell, and the resonance is a
  narrow nonlinear function of a two-particle angular correlation that no
  coordinate-level loss sees. §4 is the accumulated diagnosis.
- **Current scientific target.** One shared, mass-blind response model that
  closes more than one resonance at once (J/psi + Z), with Upsilon held out as
  a transfer test. That is the joint program, §3.
- **Machine.** Windows desktop `desktop-otag01s`,
  `C:\Users\AhrixMarin\Desktop\otus`, Anaconda `cms` env, CUDA GPU (12 GB).
  Earlier work ran on macOS/MPS; session entries before 2026-08-21 refer to
  that machine. Machine history: `/Users/liziqing/Desktop/Codex/otus_on_data`
  -> `/Users/ahrimarin/Desktop/otus_on_data` (moved 2026-08-14, same content)
  -> the current Windows path.
- **Synced reference mirror** (macOS era; treat `sources/` as strictly
  read-only): `/Users/liziqing/.codex/.chatgpt-projects/g-p-6a3a231c9cb48191a8fbb0a03f979b09`.
  Held `OTUS_paper.pdf`, the same file as `docs/2101.08944v2.pdf`.
- **Paper.** `docs/2101.08944v2.pdf` = arXiv:2101.08944v2, "Learning to
  Simulate High Energy Particle Collisions from Unlabeled Data" (Howard, Mandt,
  Whiteson, Yang). MD5 3c99a7f95e70c97a2152377c9370571a. Reading notes in §8.
- **Upstream OTUS repo:** https://github.com/yiboyang/otus (reference only;
  network use forbidden unless separately authorized).
- **SWAE reference:** arXiv:1804.01947 (Kolouri et al.).
- **Sister studies:** successful Z->e+e- OTUS (reference PDFs) and the upstream
  paper experiments ppzee / ppttbar.

## 3. The three experimental programs

Three distinct programs have run in this repo. **Run letters repeat across
them** — there is a Run D, E and F in both the SOTA and the joint program — so
always qualify the name. This ambiguity has already caused confusion in the
record (the 2026-09-01 plateau review had to disambiguate in its title).

| Program | Runs | Outputs | Configs | Status |
|---|---|---|---|---|
| **A. Single-region J/psi** | v1 - v3.10 | `outputs/cms_JpsiDoubleMuons/archive/` | `configs/archive/` | Closed; superseded |
| **B. New-prior + SOTA** | paper_20pct, three-term, Run D, SOTA Run E/F/G0 | `outputs/cms_Jpsi_new/`, `outputs/cms_Jpsi_sota/` | `configs/`, `configs_sota/` | Closed; SOTA Run E is its best result |
| **C. Joint multi-region** | joint Run A - F | `outputs/cms_Joint/` | `configs_joint/` | **Active** |

**Program A — single-region J/psi (v1 - v3.10).** The original attempt: OTUS on
the CMS J/psi window against a signal-only, delta-like MG5 prior. Produced an
excellent generator and a broken encoder, then the v3.8 literal-paper objective
diverged. Its value now is diagnostic: §4.2 root causes and §5.3 / §5.4 / §5.6
come from here.

**Program B — rebuilt prior and the neural-OT upgrade.** Rebuilt the MG5 prior
(massive muons, smearing, pT reweighting, continuum admixture, ptj 10 -> 5),
then escalated the objective: paper-vanilla -> three-term cosine -> Run D
(pair-level SWD + relative-mass cycle Huber) -> SOTA Run E "Tier A" (physics-cost
Sinkhorn, adversarial max-SW slicing, cylindrical residual flow, hard mass
gates) -> SOTA Run F (bidirectional OT flow matching, did not beat Run E).
SOTA Run E remains the best single-region result: D(z) mass W1 0.00105 / KS
0.0165, C2ST 0.517.

**Program C — joint multi-region (current).** One shared encoder/decoder trained
on J/psi and Z at once, with no explicit parent-mass conditioning and no
resonance label, Upsilon held out. Runs A -> F are a chain of single controlled
changes (Z prior species, training profile, full-scale splits, showered Z prior,
full-pass epochs, showered J/psi prior). Joint Run E is the current best; joint
Run F failed and was removed.

## 4. Established findings

### 4.1 What "failure" means here
1. **Early-run failure** (runs A/B, v1-v3, pre-v3.5): cycle masses of 2.49 GeV
   (run A) and 1.68 GeV (run B) vs the 3.04 GeV data mean; run A corrupted the
   8-vector while satisfying the mass marginal. reported-not-reproduced
   (documented in configs/cms_JpsiDoubleMuons_mps.yaml comments; artifacts deleted).
2. **Residual failure at HEAD (v3.5):** generator z->x is excellent; the
   reconstruction cycle and the encoder's latent sharpness remain compromised.
3. **Ablation fragility:** v3.6A (no explicit mass supervision) degrades the
   generator ~12x.
4. **v3.8 literal-paper objective diverges** (see §5.4).

### 4.2 Ranked root causes
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
   stays small (§5.4 trajectory).
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

### 4.3 What works
- **v3.5 generator (z->x simulation) is a genuine success** at the mass level:
  W1 = 0.0047 GeV, KS = 0.012 on 50k test events; pred histogram nearly
  indistinguishable from truth (peak-bin 3055 vs 3213; frac-in-resonance 0.497
  vs 0.497). artifact-measured.
- v3.5 unfolding mean is pinned to 3.0953 vs prior 3.0969 (offset -1.6 MeV);
  the remaining defect is width (24 vs ~6 MeV) and cycle mass peak (3.03-3.06
  vs 3.095).

### 4.4 Known-suspect verdicts (current source)
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

## 5. Key measured numbers

Ordered by section number. Data and prior characterisation: §5.1, §5.2, §5.5,
§5.7, §5.8, §5.9. Run results: §5.3, §5.3b, §5.4, §5.6.

### 5.1 CMS x_test (full v3.8 eval set; artifact-measured)
- n = 717,760; mass mean 3.0442, std 0.1853, mode bin 3.095 GeV,
  frac in [3.0369, 3.1569] = 0.505.
- Cut flow (experiments/cms_Jpsi_ee/jpsi2mumu_peak.ipynb saved outputs):
  148,124,900 muons (pT>2, |eta|<2.4) -> 52,998,579 events >=2 muons ->
  83,393,605 OS pairs -> **7,177,454 OS pairs in [2.6,3.5]**; 2,803,269 SS pairs
  in window; data peak 3.0925 GeV (5 MeV bins); window mean 3.0439, median 3.0816.

### 5.2 Prior z_test (artifact-measured)
- Full (v3.8): n = 100,000; mass mean 3.09692, std 0.00571, frac-in-resonance 0.999.
- 50k capped (older evals): mean 3.09685, std 0.01047 (tail-dominated; core is a delta).
- Stored energies are not exactly on-shell (source-verified,
  scripts/loss.py comment); z-space mass uses stored-E authority, x-space
  uses the stable p-based formula.

### 5.3 v3.5/v3.6A/candidates (artifact-measured, 50k/5k test sets as noted)
| Checkpoint | z->x W1 [GeV] | z->x KS | cycle KS | cycle fracR | E(x) mass sigma [MeV] |
|---|---|---|---|---|---|
| v3.5 stage1 | 0.0656 | 0.222 | 0.715 | — | — |
| v3.5 stage2 | 0.0255 | 0.127 | 0.403 | 0.568 | 24.1 |
| v3.5 best | **0.0047** | **0.012** | 0.544 | 0.369 | 24.1 |
| v3.5 final | 0.0054 | 0.012 | 0.528 | 0.584 | 24.1 |
| v3.6A best | 0.0582 | 0.141 | 0.270 | — | 44.4 |
| encoder candidates (5k) | 0.088-0.506 | 0.19-0.29 | — | 0.55-0.94 | 44-95 |
| candidateB stage3 ep20/100/best (5k) | ~0.21/0.20/0.21 | — | — | 0.141/0.846/0.705 | 53.7 (frozen) |

### 5.3b Fresh same-harness baselines (2026-08-15, eval.py, mass range
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

### 5.4 v3.8 vanilla-paper run (artifact-measured)
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

### 5.5 E1 kinematics dump (2026-08-15; artifact-measured via
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
  (matches §5.1's muon count; the "(pT>2, |eta|<2.4)" label there was
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

### 5.6 v3.9 F1-restricted pilot — early trajectory (artifact-measured,
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

### 5.7 CMS-vs-MG5 mismatch re-check, J/psi ~3.09 GeV focus (artifact-measured
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

### 5.8 Z->ee health check (artifact-measured 2026-08-14 via
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

### 5.9 Joint-series prior mass width vs CMS (artifact-measured 2026-09-04,
computed from the region caches under `outputs/cms_Joint/.region_cache/jpsi/`,
z_test / x_test splits, p-based mass with m_mu = 0.105658)

| sample | source | n | pair-mass std | pair pT median | muon pT median |
|---|---|---|---|---|---|
| Run D/E prior | `cms_jpsi_mumu_mg5_8tev_mixed_ptj5` (cache 319d7736) | 8,958 | **26.2 MeV** | 26.30 | 12.68 |
| CMS data x_test | Run2012BC DoubleMuParked | 362,446 | **28.1 MeV** | 26.00 | 12.53 |
| Run F prior | `..._0j1j_..._reweighted` (cache b65a524d) | 99,997 | **14.7 MeV** | 26.10 | 12.40 |

Consequence (hypothesis, high confidence, and the reason for the A/B below):
the Run D/E prior had been hand-smeared (a = 0.0128, Session 12) until its mass
width nearly equalled the data's, so the encoder had to remove only ~7% of the
mass width — the mass map was close to an identity. Both Run D and Run E
duly selected a **stage-1 deterministic** checkpoint (noise multipliers 0/0) as
best, and their J/psi latent gates passed easily (latent_mass_ks 0.038 / 0.034).
Run F's prior is 1.9x narrower, requiring genuine resolution deconvolution; its
encoder removed only ~21% of the width (latent_mass_width_rel_error 0.509 at
epoch 232) and it failed `latent_mass_ks` at **all 51** of its validations
(0 gate-passing epochs, vs 33 for Run D and 75 for Run E). Run F changed the
generator, shower, reweighting, event count and width at once, so the width
attribution is untested — see the A/B in §7.

### 5.10 Two facts that constrain the method program (2026-09-04)

**The joint loss is already mass-aware, and the conditioning is already
mass-blind.** source-verified from `outputs/cms_Joint/Run_E/config.resolved.json`:
the loss carries `pair_mass_w1 = 1`, `mass_kin_swd = 0.5`,
`delta_eta_w1 = delta_phi_w1 = 0.25`, `physics_swd = 0.5`,
`x_reco_physics_w1 = 1` (with `mass_w1` and `resonance_mass_w1` at 0), while
`model.condition_features` lists 13 features with `log_pair_mass` deliberately
absent. So the encoder fails the latent mass gate on an honest prior *while the
objective penalises the pair mass directly*. This rules out the cheapest
remaining hypothesis — "add a mass term to the loss" — which was Program B's
fix and is already carried into Program C. The remaining failure is about the
shape of the map (a point estimate standing in for a posterior) and/or the
shared encoder/decoder noise, not about the loss being blind to the resonance.

**Paired truth data is already on disk and unused.** source-verified from
`PaperPlots/ppzeePaperPlots.ipynb`: `data/ppzee_test.hdf5` and
`data/ppttbar_test.hdf5` expose `z_data` and `x_data` as **event-by-event
pairs** (the upstream OTUS paper's MadGraph+Pythia+Delphes datasets). Running
the unpaired joint pipeline on them and then using the withheld pairing gives a
direct per-event unfolding closure test — the only way to answer "how do you
know the unfolding is right without truth pairs", and the way to turn the
undercovered coverage proxies (cycle 0.754, pseudo-pair 0.490 for SOTA Run E)
into interpretable numbers. proposal: schedule this alongside the method work,
not after it.

## 6. Open questions (ranked)

Items are referenced elsewhere as §6.1, §6.2 ... by their number in this
list. Answered questions are struck through and kept, not deleted.

1. ~~Kinematic support of the priors~~ **ANSWERED 2026-08-15**: see §5.5. The
   signal prior is boosted+recoil (not pure 2->1), passes the trigger-matched
   selection at 0.60%; the inclusive prior is pure 2->1 and passes at 0.00%.
2. ~~v3.8 divergence mechanism~~ **REFINED 2026-08-15**: the "divergence" was
   a transient epoch-290 spike (latent grad 73.8k) on an otherwise-plateaued
   run; the catastrophic generator came from a converged-but-degenerate
   solution where D(z~prior) is never trained (§5.4). The v3.9 F1-restricted
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
   hence identical pT>2 and pT>3 pair counts (§5.5).

6. ~~Does the joint model's J/psi mass closure depend on a pre-smeared
   prior?~~ **ANSWERED 2026-09-04** (§7.2, artifact-measured): yes, entirely.
   Smeared 34/35 gate passes, narrow 0/35, single-variable change. The
   follow-on questions it opened are items 8 and 9.
7. **Should the encoder and decoder have separate noise schedules?** OPEN.
   They currently share one, which is physically backwards: the decoder adds
   detector resolution, the encoder should remove it. The 2026-09-01 plateau
   probe showed that simply reducing encoder noise at fixed weights improves
   latent mass W1 but worsens KS, because a ~-8.8 MeV mass bias persists — so
   this needs a retrained comparison, not an inference-time switch.
8. ~~Is the stage-1 deterministic phase structurally unable to solve the
   narrow-prior case?~~ **FALSIFIED 2026-09-05, M1 complete, see section 7.5**:
   the completed narrow arm discarded the mass width under a fully
   deterministic decoder, which this hypothesis says should be impossible, and
   the split-schedule rerun has not beaten the identity map at any of its first
   13 validations. Original statement follows (§7.2). Stage 1
   runs with `core_noise_multiplier` and `tail_noise_multiplier` both 0
   (source-verified, `config.resolved.json`), so the decoder has no stochastic
   source with which to re-add resolution. Under a cycle-MSE term the encoder
   therefore cannot discard the mass width without making reconstruction
   impossible, so the physically correct solution is unreachable for the first
   72 epochs by construction. Joint Runs D and E and the smeared arm all
   selected inside that phase; they were never penalised for it because their
   prior already carried the resolution.
9. **Do the current gates have power to detect unfolding?** **ANSWERED
   2026-09-07 (Session 30): no, and now there is a direct demonstration.**
   artifact-measured (§7.2): on the smeared prior the identity map scores
   latent mass KS 0.0359 / W1 0.00219 / width_rel 0.071 against targets of
   0.04 / 0.003 / 0.1 — it passes all three. A gate that a no-op passes
   cannot certify the physics claim. The ppzee run closes this: the encoder
   improved the latent mass marginal to a gauge of 0.588 — real, measurable
   marginal work — while its **per-event** scatter against the withheld truth
   partner got 2.0% *worse*. Marginal improvement and per-event inversion came
   apart in the same checkpoint, on the same events. The `*_vs_identity`
   gauges of §7.3 are still necessary and still not sufficient; only paired
   truth settles it, and only ppzee/ppttbar have it.

10. **Is the per-event failure a property of the objective or of the
    architecture?** NEW, OPEN, 2026-09-07. Session 30 measured our residual-flow
    encoder at 0.9843 and the upstream MLP encoder at 3.3304 on the same 160,000
    events under the same objective family, so at least 3.4x of the upstream
    per-event miss is architectural. What neither reaches is the 0.81 oracle.
    The test that separates the two explanations is M2: a posterior encoder with
    the same residual parameterisation. If it also stalls near 0.98, the
    objective is the binding constraint.

## 7. Current plan and next experiment

*Updated 2026-09-04 (Session 25), after the §7.1 A/B completed (§7.2).*

**Next actions in order. Steps 1-3 cost no GPU time and should happen before
any training starts.**

1. ~~Report the identity baseline alongside every gate number.~~ **DONE
   2026-09-04 (Session 26), see §7.3.**
   §7.2 shows a no-op passes the smeared-prior latent targets. Until each
   metric is printed next to what the identity map scores on the same pair,
   no gate number carries a claim. Cheap: pure numpy over the region caches,
   no model involved.
2. ~~Add at least one metric the identity map fails by construction.~~
   **DONE 2026-09-04 (Session 26), see §7.3.** The chosen form is
   `*_vs_identity`, which is 1.0 for a no-op by construction; the C2ST AUC and
   the paired closure test are kept as the two independent cross-checks rather
   than as alternatives.
3. ~~**Run the paired closure test on the upstream paired datasets**~~
   **DONE 2026-09-07 (Session 30), see §11 Session 30 and the §1 summary.**
   The ppzee loader branch exists (`scripts_joint/paired_data.py`, additive;
   the CMS muon path is byte-identical, proven by Run E's contract hash). Our
   encoder scores `residual_rms_vs_identity` **0.9843** against an identity of
   1.0, an oracle floor of 0.81 and the upstream encoder's 3.33, with all four
   leakage checks clean. It removes a global mass bias and does not invert the
   response per event. ppzee is now the recommended development bench for M5
   and, with the restriction in §11 Session 30 item 8, for M2.
4. **Split the encoder and decoder noise schedules and rerun the narrow arm**
   (§6 items 7 and 8). Code and config are in place and unrun:
   `configs_joint/cms_Joint_abNarrowSplit.yaml`, launched with
   `python scripts_joint/run_joint.py --run abNarrowSplit --device cuda`
   (preflight with `--dry-run` first). Specific test: decoder stochastic from epoch 0,
   encoder deterministic, everything else as `cms_Joint_abNarrow.yaml`. If the
   stage-1 hypothesis is right, the narrow arm's latent mass KS should drop
   below the identity baseline of 0.370, which the completed run never did
   (it ended at 0.415, worse than doing nothing). One ~6.5 h run answers it.
5. **A genuine stochastic inverse for the encoder** — an amortized posterior
   with a likelihood/consistency term, or the paper's restricted-decoder
   reweighting (Eqs. 17-19, §8) — rather than more OT terms on marginals.
   Conditioned on step 4: attempt this only if splitting the noise schedules
   fails to recover the narrow arm.
6. **Rewrite `paper/main.tex`** around what is defensible: the z->x simulation
   direction, plus the A/B as a controlled negative result on the x->z
   inverse. The current skeleton is the single-region SOTA Tier A draft and
   predates Program C.

**Superseded (Program B chain, completed).** Kept so old references resolve:

- Prior rebuilt with ptj=5 -> `data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5`
  (old ptj=10 file retained as provenance).
- Three-term cosine baseline
  `alpha(t)*[SWD(z,E(x)) + SWD(x,D(z))] + lambda_cycle*MSE(x,D(E(x)))`
  (`configs/cms_Jpsi_newprior_ptj5_3term_cosine_20pct.yaml`; completed).
- Run D extension: pair-level SWD over
  `[log m, log pT, y, cos dphi, sin dphi]` plus cycle relative-mass Huber
  (`configs/cms_Jpsi_newprior_ptj5_rund_pairswd_cyclemass_20pct.yaml`).
- All historical v3.x configs live in `configs/archive/`; frozen diagnostics
  in `scripts/legacy/`.

### 7.1 Controlled prior-width A/B (staged and run 2026-09-04)

The open question after Run F is whether the joint model's J/psi mass results
depend on the prior having been pre-smeared to the data's width (§5.9). Run F
changed five things at once, so it cannot answer it.

Design (`scripts_joint/build_ab_priors.py`): both arms are derived from the same
source file `cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_..._reweighted.hdf5`
and the same events. Arm `narrow` keeps the source width (~14.7 MeV); arm
`smeared` applies the calibrated Gaussian pT smear from
`scripts/prior_build/smear_prior.py` to reach ~26.2 MeV, the Run D/E width. Each
arm then has the training selection applied independently, and the smearing
amplitude is bisected against the width measured **after** selection. The
builder aborts unless the achieved width is within 2 MeV of target and the
muon-pT, pair-pT and |eta| medians agree within 2% between arms, and writes
`data/cms_jpsi_ab_manifest.json` with both sha256s.

First attempt (2026-09-04, corrected same session): the arms were event-matched
— an event kept only if it passed the window before *and* after smearing. The
guard caught it. Because the window is only 120 MeV wide, that requirement
discards the tail events carrying the width in both arms: the smeared arm
truncated to 23.23 MeV (target 26.2, refused) and the narrow arm fell from
14.7 to 12.79 MeV, deforming the quantity under test. Replaced by per-arm
selection with post-selection calibration.

artifact-measured (2026-09-04, verified with pure numpy against the cached
Run F prior events, `outputs/cms_Joint/.region_cache/jpsi/selected_split_b65a524d*.npz`,
999,957 events): the corrected procedure calibrates to a = 0.01257 and yields
narrow 999,957 events / 14.70 MeV and smeared 942,775 events / 26.22 MeV with
5.72% window spill-out; muon-pT median 12.41 vs 12.42 and pair-pT median 26.06
vs 26.06. Two independent confirmations that this reproduces how the Run D/E
prior was built: Session 12 recorded a = 0.0128 for that prior (here 0.01257)
and a ~5.7% smearing-induced spill-out (here 5.72%). The post-selection width
ceiling is ~34 MeV, so 26.2 MeV is comfortably reachable.

Configs `configs_joint/cms_Joint_abNarrow.yaml` / `cms_Joint_abSmeared.yaml`
inherit Run E and change only the J/psi prior path plus a bounded schedule
(stage 1 = 72 epochs, stage 2 = 88; stages 3-4 disabled). Run E selected at
stage-1 epoch 65 and Run F was already failing inside stage 1, so the signal
should appear early.

Read-out: `joint_selection.regions.jpsi.gates.latent_mass_ks` and
`latent_mass_width_rel_error` per validation in each `history.json`.
- smeared passes, narrow fails -> the Run D/E mass closure was carried by the
  pre-smeared prior; the latent-width failure is the real physics result and
  the method needs a genuine stochastic-inverse, not more OT terms on marginals.
- both pass -> width was not the cause; Run F's regression is attributable to
  the other four changes (generator/shower/reweighting/event count).
- both fail -> the bounded schedule or the event-matched selection changed
  something else; check the manifest before drawing any conclusion.

### 7.2 A/B result (artifact-measured 2026-09-04, Session 25)

Both arms completed on the Windows GPU host: `outputs/cms_Joint/AB_narrow/`
(6.49 h wall, 160 epochs, 35 validations) and `outputs/cms_Joint/AB_smeared/`
(4.22 h wall, same schedule). source-verified: `diff` of the two
`config.resolved.json` files differs only in `theory_prior_file`, `run_name`,
`run_label` and the contract hash. artifact-measured from
`data/cms_jpsi_ab_manifest.json`: the arms agree to <0.1% in muon-pT median
(12.407 vs 12.413 GeV), pair-pT median (26.055 vs 26.055 GeV) and |eta| median
(0.8381 vs 0.8378); they differ only in mass width, 14.70 vs 26.25 MeV.

**Headline.** J/psi region gate passes: **narrow 0/35, smeared 34/35.** The Z
control region passed 35/35 in both arms, so nothing outside the J/psi prior
moved. This is the "smeared passes, narrow fails" branch of the §7.1 read-out.

**But the smeared arm did not unfold.** artifact-measured (pure numpy over the
region caches, `selected_split_636b42e5*.npz` and `*_4eb219e4*.npz`, z_test vs
x_test, equal-count subsets):

| latent x->z, J/psi | narrow arm | smeared arm | gate / target |
|---|---|---|---|
| identity map (z~ = x) mass KS | 0.3702 | **0.0359** | 0.12 / 0.04 |
| identity map mass W1 [GeV] | 0.01736 | **0.00219** | 0.015 / 0.003 |
| identity map width rel. error | 0.910 | **0.071** | 0.5 / 0.1 |
| trained model mass KS (final eval) | 0.4148 | 0.0250 | 0.12 / 0.04 |
| trained model C2ST AUC (MLP) | 0.8515 | 0.5344 | 0.5 = indistinguishable |
| encoded mass std [MeV] | 56.98 | 27.99 | prior 14.72 / 26.18; data 28.05 |

Three readings follow, in increasing order of consequence.

1. **On the smeared prior the identity map passes every strict target.** The
   trained model (KS 0.0250) improves on doing nothing (KS 0.0359) by a margin
   far smaller than the gate width, and its encoded mass std, 27.99 MeV, sits
   on the *data* width (28.05) rather than the prior width (26.18). The smeared
   arm's 34/35 is a measurement of how close the prior was to the data, not of
   what the encoder learned.
2. **On the narrow prior the trained model is worse than doing nothing**
   (KS 0.4148 vs identity 0.3702; C2ST 0.85 = easily separable). The encoded
   mass distribution is close to flat across the 120 MeV window; see
   `outputs/cms_Joint/AB_narrow/plots_paperstyle/jpsi/paperstyle_zspace_mass_density_ratio.png`.
   Note the target it was asked to hit: artifact-measured, the narrow prior has
   81% of its events within +-5 MeV of 3.0969 GeV and an IQR-based width of
   **0.16 MeV** — a near-delta mass shell plus a ~17% continuum pedestal,
   which is what puts its std at 14.7 MeV. The smeared prior's IQR-based width
   is 27.99 MeV against the data's 30.03 MeV. Smearing the prior did not make
   the inverse problem easier; it removed the inverse problem.
3. **The z->x (simulation) direction is the part that survives.** Narrow arm
   z->x mass KS 0.1825 against an identity baseline of 0.3702 — a real
   improvement, still failing the 0.12 gate; smeared arm 0.0137. The decoder is
   doing work in both arms. The encoder is not.

**Checkpoint selection reproduced the Run D/E pathology exactly.** The smeared
arm selected `global_epoch` 35, stage `runA_stage1_deterministic_identity`
(noise multipliers 0/0) — as joint Runs D (epoch 65) and E did. The narrow arm
had no gate-passing epoch at all and fell back to the final checkpoint;
`outputs/cms_Joint/AB_narrow/global_gate_fallback.txt` records that
`best_model.pt` is **not** an accepted result.

**Mechanistic hypothesis (high confidence, testable, now §6 item 8).** Stage 1
sets both noise multipliers to 0, so the decoder is deterministic and has no
source of resolution to re-add. A cycle-MSE term then forbids the encoder from
discarding the mass width, because the discarded width cannot be regenerated.
The physically correct map is therefore unreachable during the first 72 epochs
by construction, and Runs D, E and the smeared arm were never penalised for it
only because their priors already carried the resolution. This is the first
concrete, falsifiable account of the narrowing failure that has recurred since
v3.5, and §7 step 4 is the experiment that decides it.

**What may and may not be claimed in the paper as of today.** May: the joint
mass-blind decoder simulates J/psi and Z simultaneously from truth-level
priors, with a controlled ablation showing the encoder direction fails when the
prior is not pre-smeared. May not: that the model unfolds detector resolution
for J/psi. The Run D/E J/psi unfolding numbers are closure against a
pseudo-truth prior that had already been given the resolution.

### 7.3 Identity and floor references, implemented (2026-09-04, Session 26)

source-verified, new code:

* `scripts_joint/identity_reference.py` — NumPy only, no torch import, so
  finished runs can be re-scored offline and the statistics are unit-testable
  without a GPU. Defines, per direction:
  `{d}_identity_*` (the metric when the map is the identity, i.e. the model
  output replaced by its own input), `{d}_floor_*` (two independent draws of
  the target distribution at the same sample size), and the gauge
  `{d}_*_vs_identity = (model - floor) / (identity - floor)`. 1.0 = no better
  than the input, 0.0 = at the finite-sample floor, > 1 = worse than doing
  nothing. Also `{d}_*_headroom = (identity - floor) / floor`, a property of
  the dataset that says whether the comparison can resolve anything at all.
* The gauge is lower-is-better and non-negative on purpose, so it drops into
  the existing `abs(value) <= threshold` gate machinery in
  `score_joint_metrics` with **no change to the selection logic**.
* The floor uses two disjoint draws of exactly n events when the pool allows
  it, otherwise halves rescaled by sqrt(m/n); `{d}_floor_disjoint` records
  which branch ran. The A/B test splits fall in the rescaled branch.
* Deliberately no cycle gauge: the identity cycle is x -> x, whose metrics are
  identically zero, so cycle gates have no power against a no-op and a ratio
  there divides by zero.
* Wired into `joint_metrics.evaluate_region` (every validation) and into
  `run_joint.py`'s final report as `regions.<name>.identity_reference`.
  `scripts_joint/identity_baseline.py` reconstructs the same numbers for runs
  that finished before the change, reproducing each run's exact per-epoch
  validation seed rather than approximating it.
* Tests: `tests/test_identity_reference.py` (13) and
  `tests/test_paired_closure.py` (7), both NumPy-only; 20/20 pass. The load
  bearing ones are "an identity model gauges to exactly 1.0" and "a shuffled
  prediction matches the marginal but fails per event".

**Retrofit of the completed A/B** (artifact-measured 2026-09-04, final
evaluation geometry, x -> z, J/psi):

| arm | model KS | identity KS | floor KS | vs_identity | headroom |
|---|---|---|---|---|---|
| narrow | 0.4148 | 0.3706 | 0.0043 | **1.12** | 86.1 |
| smeared | 0.0250 | 0.0363 | 0.0042 | **0.65** | 7.6 |

and the Z control region, same runs: model KS 0.0058-0.0063 against identity
0.0885 and floor 0.0061, **vs_identity 0.000-0.001**, headroom 13.4.

Three things follow that the raw numbers could not show.

1. **The Z region is a genuine success.** The encoder closes essentially the
   entire identity-to-floor gap on Z, in both arms. That is real unfolding of
   the Z mass marginal, and it is the first defensible positive result in the
   joint program. The physical difference from J/psi is that the Z truth prior
   has real width (Gamma = 2.5 GeV) while the J/psi truth prior is a near-delta
   (IQR width 0.16 MeV, §7.2), so the Z task is a deconvolution the model can
   express and the J/psi task is not.
2. **The smeared J/psi arm did partial work, not none.** Per-validation gauge
   0.19-0.84 through stage 1, degrading past 1.0 during stage 2 while every
   gate still read True. So "the model learned an identity in mass" (§7.2) is
   too strong; the accurate statement is that it closed some of a gap that had
   been made small by construction, and the gate could not tell the difference.
3. **The narrow J/psi arm never beat the input on shape.** Its KS gauge is at
   or above 1.0 at every one of the 35 validations. Its W1 gauge does fall to
   0.23-0.60, so the encoder does transport mass toward the prior in mean
   position while making the shape worse. Partial transport, wrong shape.

### 7.4 Paired closure: scorer built, upstream reference measured (2026-09-04)

`scripts_joint/paired_closure.py` scores a map against each event's withheld
truth partner instead of against a histogram. The headline statistic is

    residual_rms_vs_identity = rms(mass(pred) - mass(z_true))
                             / rms(mass(x_input) - mass(z_true))

whose denominator is the detector resolution itself, so < 1 means resolution
was inverted event by event and >= 1 means it was not.

artifact-measured, from `experiments/ppzee/otus_results-dataset=ppzee_test.npz`
(the repository's copy of the upstream authors' saved results on their own
ppzee benchmark; 160,000 paired events, arrays `z`, `x`, `x_encoded`,
`z_decoded`, `x_reconstructed`, all [N, 8] and index-aligned — per-event
correlation between x and z columns is 0.977-0.991, which is what confirms the
pairing):

| quantity | value |
|---|---|
| detector resolution, rms(mass(x) - mass(z)) | 2.65 GeV |
| upstream encoder, rms(mass(E(x)) - mass(z)) | **8.83 GeV** |
| residual_rms_vs_identity | **3.33** |
| per-event mass correlation with truth | 0.679 (identity: 0.964) |
| latent mass std: E(x) / prior | 11.87 / 9.33 GeV (over-dispersed by 27%) |
| cycle D(E(x)) vs the x it came from | rms 1.03 GeV, correlation 0.994 |

Read this carefully and fairly. OTUS's published claim is distributional, not
per-event, so a per-event miss is not a broken promise. What the number does
establish is that **the upstream method, on its own benchmark, does not invert
the detector response event by event** — its encoder is 3.3x further from the
truth partner than the detector-level event already was. Our J/psi failure is
therefore not a CMS-specific or a J/psi-specific artefact; it is a property of
the method that a marginal metric cannot see. That reframes the whole program:
the question is not "why does our encoder fail where the paper's works", it is
"what has to be added to the objective for any encoder in this family to
invert a response".

**CORRECTION, 2026-09-07 (Session 30). The last three sentences above are too
strong and should be read with this.** Our own encoder, trained on this same
benchmark under the same objective family, scores **0.9843**, not 3.33 — a
factor of 3.4 better per event. So the upstream 3.33 is substantially a
property of its *architecture* (an unconstrained MLP) rather than of the
objective alone; ours is a residual flow initialised near the identity with
bounded `mean_residual_limits` and structurally cannot wander that far from its
input. What survives, and is the claim to quote, is weaker and better
supported: **an OT objective on marginals buys a global scale correction and
does not buy a per-event inverse.** Our 0.9843 is entirely accounted for by
removing 85% of a global mass bias, with the per-event scatter 2.0% worse than
the identity's, against an oracle floor of 0.81. Full detail in §11 Session 30.

artifact-measured, secondary: the saved `z_rep`/`z_decoded_rep`/`x_rep` arrays
(100 events x 100 repeats) carry a spread across repeats of ~3e-5, i.e. none.
Either the upstream decoder is effectively deterministic at evaluation or those
draws were taken with a fixed seed; the archive cannot distinguish the two.
Worth resolving before quoting it, but it points the same way as our own
finding that every selected checkpoint has been a deterministic one.

**What remains before our model can be scored this way.** `joint_data.py`
accepts only the CMS muon channel (`resolve_joint_config` raises on any other
`data.channel`), so a ppzee region needs a loader branch that reads the
`FDL` / `ROL` datasets out of `data/ppzee.hdf5` and emits the six split arrays;
the pairing is then withheld from training and used only by the scorer. The
scorer is dataset-agnostic and tested; the loader branch is the open work item.

### 7.5 M1: split noise schedules — COMPLETE, hypothesis falsified

`outputs/cms_Joint/AB_narrow_split/`, launched with
`run_joint.py --run abNarrowSplit --device cuda`. source-verified from
`config.resolved.json`: prior `cms_jpsi_ab_narrow.hdf5`; only
`split_stageA_stochastic_decoder` (72 epochs, encoder core 0.0, decoder core
1.0) and `split_stageB_stochastic_encoder` (88 epochs, encoder 0.1->1.0 cosine,
decoder 1.0) enabled; `latent_mass_ks_vs_identity` and
`latent_mass_w1_gev_vs_identity` gating at 0.5 in both regions. This is the
first run in which the split schedules and the identity gauges execute at all.

artifact-measured at the first two validations (epochs 1 and 5), against the
completed narrow arm at the same epochs:

| metric, J/psi | split ep1 | narrow ep1 | split ep5 | narrow ep5 |
|---|---|---|---|---|
| latent_mass_ks | 0.3893 | 0.3818 | 0.3839 | 0.3905 |
| latent_mass_ks_vs_identity | 1.048 | (1.027) | 1.048 | (1.066) |
| direct_mass_width_rel_error | 0.6843 | 0.0935 | 0.1884 | 0.6648 |
| cycle_mass_width_rel_error | 0.8781 | 0.3262 | 0.4554 | 0.7137 |

Two readings, both provisional:

1. **The intervention is real and visible on the decoder side.** The direct and
   cycle width errors differ from the deterministic run by factors of several
   at both validations, which is what turning the decoder's noise on at epoch 1
   should do. The knob is connected.
2. **The encoder side shows nothing yet, and should not.** In the completed
   narrow arm the latent width error stayed near 0.9 until roughly epoch 40 and
   only fell to 0.03 around epochs 60-72. The discriminating window is stage
   A's second half, not its first ten epochs. Do not read the epoch-1-to-5
   agreement as a null result.

**The number to beat.** artifact-measured via `identity_baseline.py` on the
completed narrow arm: its J/psi latent KS gauge had minimum **0.898**, median
1.091, maximum 1.538, and was below 0.5 at **0 of 35** validations. So the
split arm has to hold below ~0.9, sustained, to mean anything, and below 0.5 to
pass the gate.

**Interim trend through epoch 60 (artifact-measured, 13 validations, run still
going).** The J/psi latent KS gauge by epoch: 1.048, 1.048, 1.010, 0.994,
1.059, 1.063, 1.047, 1.036, 1.077, 1.033, 1.080, **1.400**, **1.510**. Minimum
0.994 at epoch 15; first-five mean 1.032, last-five mean 1.220. It has not gone
below the completed narrow arm's benchmark of 0.898 at any validation, 0 of 13
validations passed all gates, and the last two are the worst of the run.
Meanwhile the latent width relative error falls exactly as it did before
(0.886 -> 0.749 -> 0.357 -> 0.126) and the W1 gauge improves to 0.401. Same
pathology as section 7.2 reading 3: the encoder transports mass toward the
prior in position and width while the shape gets worse.

Against the deterministic run at matched epochs the split arm is **behind**, not
ahead: at epoch 60, latent KS 0.5507 vs 0.3896 and width error 0.126 vs 0.053.
Its decoder side is also worse at matched epochs (direct mass KS ~0.036 vs
~0.019, direct width error 0.105-0.148 vs 0.005-0.035, base loss 1.035 vs
0.834), which is the cost of a stochastic decoder from epoch 1 and was
expected; what was not expected is that the encoder got nothing in return.

**The mechanism in section 6 item 8 is contradicted, and the contradiction was
already in data I had.** That hypothesis said a deterministic decoder
*forbids* the encoder from discarding mass width, because discarded width
cannot be regenerated under a cycle-MSE term. But the completed narrow arm,
with both noise multipliers at 0, reached latent width relative errors of
0.034-0.068 at epochs 60-72 (section 7.2) - it discarded the width perfectly
well. The claim was too strong when it was written and I should have caught it
against those numbers. What neither arm has ever achieved is the *shape*: the
KS gauge has never gone below 0.898 in 48 validations across the two runs.

So the reachability explanation is looking dead, and the failure is where
section 7.4 pointed: a point-estimate encoder standing in for a posterior. That
is section 7 step 5 (M2), not a schedule fix.

**Caveats before calling it.** Stage A has 12 epochs left and stage B (88
epochs, encoder noise ramping 0.1 -> 1.0) has not started. Stage B is the half
that changes the encoder itself, so the run should be allowed to finish before
the result is recorded as final. The epoch 45-60 window is also noisy in both
arms (the deterministic run ran 0.449, 0.414, 0.565, 0.390 over the same
stretch), so two adverse points are not yet a trend on their own - it is the
*absence of any point below 0.9 in 13 validations* that carries the weight.

**FINAL RESULT (artifact-measured, 160 epochs, 35 validations, completed
2026-09-05 ~05:15).** M1 is answered: **splitting the schedules changed nothing
for the encoder.**

| J/psi, final evaluation | split arm | narrow arm |
|---|---|---|
| x -> z mass KS | 0.3626 | 0.4148 |
| **latent vs_identity** | **0.983** | 1.121 |
| x -> z C2ST (MLP) | 0.842 | 0.852 |
| encoded mass std | 25.51 MeV | 56.98 MeV |
| z -> x mass KS | 0.0384 | 0.0190 |
| **direct vs_identity** | **0.098** | - |
| z -> x C2ST (MLP) | 0.581 | 0.746 |

Per-validation J/psi KS gauge across the run: minimum **0.972** (epoch 117),
median 1.080, maximum 1.883. Below 1.0 at **2 of 35** validations, below the
completed narrow arm's best of 0.898 at **0**, below the 0.5 gate at **0**.
Zero gate-passing validations; `global_gate_fallback.txt` again records that
`best_model.pt` is not an accepted result.

**Verdict.** A stochastic decoder from epoch 1 with a deterministic encoder does
not let the encoder invert the response. Together with the observation that the
fully deterministic arm discarded the mass width perfectly well (section 7.2),
the reachability hypothesis of section 6 item 8 is **falsified**, not merely
unsupported. The failure is where sections 7.6 and 7.4 place it: the encoder
noise floor (2.19 MeV against a 0.16 MeV target) and a point-estimate map that
cannot express a mixture.

**What the run did confirm, and it is not nothing.** The decoder is unaffected:
direct gauge 0.098 and C2ST 0.581 on the honest prior, holding flat while the
encoder wandered between 0.97 and 1.88. And the Z region unfolds essentially
perfectly throughout - final latent gauge **0.052**, C2ST 0.513. One
architecture, one objective, two regions: it works where the truth prior has
physical width and fails where the truth prior is a delta. That contrast is now
measured twice, on two different schedules.

**Programme consequence.** M1 is closed. The next branch is M2 (posterior
encoder) or M5 (decoder as forward simulator), and section 7.7 argues the
decoder-first route is already earning its keep.

**A gate note that is not a defect.** The Z region fails its new
`*_vs_identity` gate at epoch 5 (gauge 0.799 > 0.5). That is expected early:
on the completed narrow arm the Z gauge ran 0.86 / 0.75 / 0.68 / 0.77 over the
first four validations and only settled below 0.5 from about epoch 20, ending
below 0.5 at 29 of 35. The 0.5 threshold is reachable in validation geometry;
it is simply not reached in the first fifteen epochs.

### 7.6 Structural diagnosis: why a near-delta latent is unreachable (2026-09-04)

Written after AB_narrow_split reached stage B epoch 40/88 with no encoder-side
improvement. This supersedes the reachability story in section 6 item 8.

**The observation that fixes the diagnosis.** artifact-measured, split run epoch
70 (encoder deterministic, decoder stochastic): `latent_mass_width_rel_error`
= **0.016** with `latent_mass_ks` = **0.4760** and gauge 1.292. The encoder
matched the prior's mass width to 1.6% and is still worse than the identity map
on shape. The failure is not "cannot narrow". It is **cannot build a mixture**:
the prior is 81% inside a 0.16 MeV core plus a 19% continuum pedestal, and the
encoder emits a single smooth hump whose variance happens to match.

**Reason 1, an inequality: a stochastic encoder cannot reach this prior.**
source-verified `scripts_sota/cylindrical_flow.py`: the flow adds
`nu * core_sigma * eps` to each cylindrical coordinate, with
`core_sigma_floors = 0.001` in ln pT. With
m^2 = 2 m_mu^2 + 2(mT1 mT2 cosh(deta) - pT1 pT2 cos(dphi)), independent noise of
width sigma on each ln pT gives sigma_m/m = sigma/sqrt(2), so

    sigma_m  >=  0.001 / sqrt(2) * 3096.9 MeV  =  2.19 MeV

against a target core of 0.16 MeV - a factor of 14, before the eta and phi
noise is counted. **No weight and no amount of training closes that.** Stage B
ramps encoder noise 0.1 -> 1.0 and the width error duly went 0.016 -> 0.18-0.51.
Consequence: for any region whose prior core is below a few MeV the encoder
must stay deterministic. This also explains the 2026-09-01 plateau probe.

**Reason 2: the mass is not a coordinate.** The six flow outputs are
(ln pT, eta, phi) per muon; the mass is a curved function of all six. Turning a
30 MeV hump into a 0.16 MeV core needs the Jacobian to contract ~190x along one
curved direction while staying near unity in the other five, from a residual
map whose head initialises at 1e-4 and whose outputs pass through a bounded
tanh. The architecture is near-identity by construction.

**Reason 3: the objective pays for the compromise we observe.** beta = lambda =
1 in both stages. The SWD term wants E to destroy the per-event mass; the
cycle MSE wants E to keep it so D can rebuild x. A partial squeeze is the
compromise, and a smooth hump of matching width is a cheap minimum for a
sliced-Wasserstein term computed on random 1-D projections.

**Ordered response** (details and figures in the companion artifact "Why the
Encoder Can't Make a Delta"):

1. proposal: never ramp encoder noise on a delta-like region. Config only.
2. proposal: add `latent_mass_core_fraction` (fraction within +-5 MeV of the
   peak) to the metric block with the same identity/floor references. Prior
   0.809, CMS data and identity map 0.133. One line of numpy, and it is the
   number that separates a mixture from a hump. Would have exposed this months
   ago.
3. proposal: reparametrise the latent as (m, pT, y, phi, cos theta*, phi*) and
   give the encoder a two-component mixture head over m with a per-event
   weight, so the delta is expressible by construction. Mass-blindness survives
   only if the component parameters are learned from the prior sample and
   3.0969 is never typed in.
4. proposal: amortised posterior q(z|x) with a proper scoring rule and an
   entropy term, keeping the OT term as the marginal constraint. Without the
   entropy term nothing makes q a mixture. This is M2.
5. proposal: or abandon the inverse and reweight with the decoder as a forward
   simulator (M5), justified by section 7.4.

**Physics-versus-pattern tests that a moment-matcher fails.** proposal:
(a) fit the decoder's induced spread against pT and eta and compare with the
published CMS muon resolution sigma(pT)/pT = a (+) b*pT - a learned response
that reproduces the instrument's curve is physics, a flat or arbitrary one is
not; (b) the same decoder must give the same curve on the Upsilon region, since
a response belongs to the detector and not to the resonance; (c) per-event
closure on ppzee (section 7.4); (d) core fraction and its vs-identity gauge.

### 7.7 The decoder is not contaminated by the failing encoder (2026-09-04)

The question: if the cycle term forces D to turn a wrong E(x) back into a
correct x, is the encoder's failure being passed downstream into the decoder?
artifact-measured across all three completed/running arms:

| run | latent KS mean | direct KS mean | corr(latent, direct) |
|---|---|---|---|
| AB_narrow (35 val) | 0.4222 | 0.0320 | **-0.139** |
| AB_narrow_split (28 val) | 0.4463 | 0.0417 | **-0.205** |
| AB_smeared (35 val) | 0.0439 | 0.0299 | +0.833 |

On the honest narrow prior the two directions are **uncorrelated**, and if
anything slightly anti-correlated. The strong positive correlation on the
smeared arm is not evidence against this: there E(x) is near-identity and
z is near-x, so both metrics are measuring nearly the same comparison.

The vs-identity gauges make it sharper still. Across the whole split run the
J/psi **direct** gauge sits at 0.065-0.090 while the **latent** gauge runs
1.04-1.51. The decoder closes 91-93% of the identity-to-floor gap and holds
that flat from epoch 20 to epoch 127, including through the epochs where the
encoder degrades from 1.05 to 1.51. Best direct mass KS on the honest prior:
**0.0190** (AB_narrow, epoch 40) against a strict target of 0.03 and a gate of
0.12.

**Conclusion (artifact-measured):** the z->x direction already meets the strict
target on the honest 14.7 MeV prior, and its quality is independent of the
encoder's failure. The cycle is not poisoning the decoder in outcome. The
mechanism to watch is still real - D is trained partly on E(x), whose
distribution differs from p_z in shape - so a decoder-first program should
decouple them explicitly rather than rely on this staying true.

**The narrow prior is an asset for the decoder and a liability only for the
encoder.** z -> x asks the model to ADD resolution: a well-posed, one-to-many
map that noise can express. x -> z asks it to REMOVE resolution onto a
near-delta: ill-posed, and subject to the 2.19 MeV inequality of section 7.6.
The A/B result therefore reads two ways - it invalidates the unfolding claim
and it validates the honest prior as the correct input to a simulator.

### 7.8 What the reference notebook has that the production pipeline lost

source-verified, `reference/jpsi_otus.ipynb` (127 cells, **0 stored outputs**,
data at `pp_jpsi_mumu_full_aligned.npy` which is NOT in `data/`). Its results
therefore cannot be verified from the repository, and "both directions worked"
should not be treated as established until the identity baseline is applied to
whatever numbers it produced. Its encoder is subject to the same 2.19 MeV
inequality: `core_sigma_floors` is 0.0010 in ln pT there too, and core noise is
on from its stage 2 onward.

Three design elements in it are strictly better than what the joint pipeline
runs today, and are worth porting regardless of whether its results hold:

1. **A shape-sensitive mass objective.** It carries `mass_quantile_band_w1`
   = 1.25 over **32 quantile edges** (dense near the tails: 0.000, 0.005,
   0.010, 0.020, 0.035 ...), `mass_tail_cdf` = 0.50 over 11 tail quantiles,
   quantile-interval width matching, and fixed-bin occupancy. source-verified:
   `scripts/loss.py` supports **none** of these keys. The joint program asks
   for the mass only through `pair_mass_w1` (a plain W1) and `mass_kin_swd` -
   transport- and moment-like terms that a smooth hump of the right variance
   satisfies cheaply. This is the most plausible single explanation for
   "matches the width to 1.6%, fails the shape" in section 7.6, and it was
   available in-house the whole time.

   **Correction, same session, after the user pushed back.** Calling this "the
   answer" was too strong. Every term in this objective - W1, SWD, quantile
   bands, tail CDF - is a **marginal** statistic: it constrains the pushforward
   measure and says nothing about which x maps to which z. Finer quantile bands
   specify the target histogram more precisely; they do not add a different
   *kind* of constraint. On the x -> z side the target is our own MC, so
   tightening it is an invitation to the degenerate solution "ignore x, emit
   the prior's histogram", which scores perfectly on every marginal term and is
   worthless - precisely the failure the identity gauge and the paired closure
   test exist to catch. Adopt the quantile terms only with those instruments
   watching, and prefer, as the physics-carrying term, a constraint on the
   RESPONSE rather than on the histogram: penalise the model when its implied
   sigma(pT, eta) departs from a smooth two-parameter tracker form
   a (+) b*pT. That has few enough parameters that it cannot memorise, and it
   states a physical fact rather than fitting a shape. proposal.
2. **The latent and direct weights are annealed in opposite directions.**
   lambda_z 2.0 -> 1.0 -> 1.0 -> 0.75 -> 0.5 while tau_x 0.0 -> 0.5 -> 1.0 ->
   1.5. The joint pipeline holds lambda = 1 constant and moves tau 0.25 -> 1.
   The reference deliberately buys the latent alignment first, while the
   encoder is deterministic and a delta is still reachable, then trades it for
   detector-space fidelity.
3. **Per-stage `freeze_encoder` / `freeze_decoder` flags**, wired through
   `set_trainable(model, encoder=..., decoder=...)`. The joint trainer has no
   equivalent. This is the mechanism a decoder-first curriculum needs.

Its cells 124-126 also already implement the decoder-stochasticity test
proposed in section 7.6: fix each truth event, decode it repeatedly, and check
the spread is nonzero and kinematically sensible.

### 7.9 Are the three priors deltas? (artifact-measured 2026-09-04)

Measured with `prior_stats.py` on the cached A/B splits (J/psi, Z) and read
from the retained Upsilon peak fit. Robust width is IQR/1.349, which is the
quantity that distinguishes a spike from a bump; the standard deviation of the
J/psi prior is a composition artifact and says nothing about the line.

| region | prior robust width | physical width | CMS resolution | prior / data |
|---|---|---|---|---|
| J/psi | **0.16 MeV** | Gamma = 92.9 keV | 30.1 MeV | 0.005 |
| Upsilon(1S) | **~110 MeV** (see below) | Gamma = 54 keV | 84.4 MeV | 1.3 |
| Z | **2130 MeV** | Gamma = 2495 MeV | 3134 MeV | 0.68 |

**J/psi is a delta, Z is not, and the difference is physics.** The Z's width is
its own Breit-Wigner decay width - 2.5 GeV of real, physical spread that exists
before any detector. The J/psi's 92.9 keV is invisible next to anything. So the
Z encoder is asked to remove ~32% of a width that is mostly physical, and the
J/psi encoder is asked to remove 99.5% of a width that is entirely
instrumental. Same objective, same architecture, two completely different
problems.

This also unifies section 7.6 with the one success we have. The 2.19 MeV
encoder-noise floor is **0.1%** of the Z target and **1400%** of the J/psi
target. One inequality explains both why Z unfolds to the noise floor
(gauge 0.001) and why J/psi never beats the identity (gauge >= 0.898).

**The Upsilon prior is not a spike, and that is a problem.** artifact-measured
from `outputs/cms_Joint/Upsilon_zero_shot/0j1j_full_retry/preliminary_prior_vs_cms/peak_summary.csv`:
the per-component prior widths are **110.6 / 166.7 / 192.2 MeV** for
1S / 2S / 3S, against physical widths of 54 / 32 / 20 **keV** - three to four
orders of magnitude too wide. They are also **wider than the CMS resolution
this test is supposed to measure** (84.4 / 84.3 / 80.3 MeV).

hypothesis, high confidence: the Upsilon prior was hand-smeared like the
Program B J/psi prior. A smearing amplitude a = 0.0128 gives
sigma_m = a*m/sqrt(2) = 86 MeV at 9.46 GeV, the right order. The measured
widths grow faster with mass than that predicts (110 -> 167 -> 192 against
86 -> 91 -> 94), so this needs confirming with `prior_stats.py` on the actual
file before it is asserted.

**Consequence.** The proposed Upsilon experiment - freeze the response, decode
a prior spiked at the three resonance masses, and check the model reproduces
the CMS peak widths - is the right test and is currently **not runnable**,
because the prior already contains more width than the detector adds. It
reproduces the Runs D/E confound exactly one level up. The Upsilon prior has to
be regenerated unsmeared before that test means anything, which is another item
for the MG5 VM (docs/mg5_vm_runbook.md).

Note also `comparison_summary.json`, which records honestly that the prior's
mixture fractions are "CMS-fit-derived; not strict zero-shot", and that Upsilon
is `excluded_post_unblinding`. If a genuinely blind prediction is wanted for
the paper, psi(2S) at 3.686 GeV is in the same DoubleMuParked data, has never
been used, and tests interpolation just above J/psi. Keep it sealed.

### 7.10 Unified prior rebuild, proposed (2026-09-04)

proposal, nothing run. Full method in `docs/prior_generation_method.md`; region
specs in `configs_priors/` (`_common.yaml` plus jpsi, psi2s, upsilon, z).

The claim the method is built to support: *every number in a prior is either a
PDG constant, a generator setting fixed by a documented insensitivity test, or
a declared analysis parameter carried as a systematic; no shape in the prior
was fitted to the data it will be compared against.*

Load-bearing decisions:

* **One process for all four regions**, `p p > mu+ mu-` plus
  `p p > mu+ mu- j`, CKKW-L merged. Not 2->1 (zero ME pair pT) and not
  1-jet-with-a-cut (the old ptj=5 prior had 0% of events below 5 GeV in pair pT
  against 78.5% in CMS). Pair pT is a **conditioning feature of the response**,
  so getting its spectrum wrong trains the wrong response.
* **The method cannot predict the signal-to-continuum rate** and does not
  pretend to: quarkonium production through an effective vector q qbar -> V is
  not physical, so the coupling is arbitrary. Therefore components are shipped
  **labelled and unmixed** (`FDL/component_id`), and the mixture fraction
  becomes a declared analysis parameter fitted from a sideband and varied as a
  systematic. **A fraction is a scalar; a smear is a shape** - the first is an
  auditable one-number systematic, the second destroys the object under test.
  `mix_prior.py` and `smear_prior.py` are retained for provenance only.
* **No reweighting.** `reweight_prior.py` corrects pair pT toward data, which
  is a data-derived correction on a conditioning variable - the most damaging
  place for one. Measure the residual mismatch and report it; do not correct it.
* **The merging scale is set by an insensitivity scan**, not chosen: generate
  at TMS, 0.5x and 2x, require agreement in pair pT and acceptance, carry the
  spread as a systematic. Starting point max(10 GeV, M/4). At 3 GeV this is
  genuinely awkward; if no scale passes, that is a finding, not a number to
  force.
* **One fiducial rule**, replacing four hand-chosen windows:
  pT(mu) > 3, |eta| < 2.4, and window = M +- max(3*Gamma, 5*sigma_res) with
  sigma_res = 0.01*M. The 1% is the published CMS figure and matches our own
  measurements (0.97% at J/psi, 0.89% at Upsilon(1S)). Resulting windows:
  J/psi 2.942-3.252, psi(2S) 3.502-3.870, Upsilon 8.987-10.873, Z 83.70-98.68.
  **This breaks comparability with every existing run and cache** - the J/psi
  window widens from +-60 MeV, the Z narrows from 70-110. That is the intended
  cost of one clean rebuild of all four regions together.
* **Acceptance test that would have caught the current Upsilon prior**: every
  signal component must have a robust width consistent with Gamma_PDG, i.e.
  below ~1 MeV. Anything wider means something smeared it and the file is
  rejected.
* **psi(2S) is built with the others and then sealed.** Upsilon is already
  unblinded, so psi(2S) at 3.686 GeV is the only genuinely blind prediction
  left, and it tests interpolation just above the J/psi.

## 8. Paper (2101.08944v2) reading notes — what matters for this project

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

## 9. Where things live

### 9.1 Quick reference (current)

- Configs:
  - completed baseline: `configs/cms_Jpsi_newprior_paper_20pct.yaml`
  - completed three-term cosine: `configs/cms_Jpsi_newprior_ptj5_3term_cosine_20pct.yaml`
  - Run D candidate: `configs/cms_Jpsi_newprior_ptj5_rund_pairswd_cyclemass_20pct.yaml`
  - all old v3.x configs: `configs/archive/`
- Prior files (**corrected 2026-09-04, artifact-measured from a `data/` listing:
  two files this section used to claim are on disk are NOT present on the
  Windows host**):
  - active J/psi (Program B): `data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5` (ptj=5)
  - active J/psi (Program C, and the source of both A/B arms):
    `data/cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_3p0369_3p1569_1M_reweighted.hdf5`
  - A/B arms: `data/cms_jpsi_ab_narrow.hdf5`, `data/cms_jpsi_ab_smeared.hdf5`,
    with `data/cms_jpsi_ab_manifest.json`
  - **MISSING**: `data/cms_jpsi_mumu_mg5_8tev_mixed.hdf5` (ptj=10 provenance)
    and `data/cms_jpsi_mumu_mg5_8tev_1M.hdf5` (original delta prior). The
    second one is what makes
    `tests/test_v39_f1.TestV39Config.test_cache_metadata_records_prior_selection`
    unrunnable; that test now skips with the file name rather than erroring.
    Both are presumably on the old macOS machine.

- **Generation provenance gap for the 0j1j CKKW-L priors (2026-09-04).**
  `scripts/mg5_cards/` holds only the ptj=5 signal and continuum proc/run
  cards. There is no MG5 process or run card, no Pythia8 CKKW-L merging
  configuration, and no driver script in version control for ANY
  `mg5py8_ckkwl_8tev_inclusive_0j1j` file — which is the family that now carries
  the J/psi Program C prior, both A/B arms, the Z region prior and the Upsilon
  prior. `scripts/prior_build/` (lhe_to_prior_hdf5, smear_prior,
  reweight_prior, mix_prior) covers everything downstream of the LHE files, so
  the gap is exactly the generator step. This is the single largest
  reproducibility hole in the repository.

- **UPDATE 2026-09-05 (Session 28): the generator step is now recorded; the gap
  has moved downstream.** The cards for all seven 0j1j samples are in
  `scripts/mg5_cards/0j1j/` with `SOURCE.md` (absolute source path, sha256, mtime
  per file) and a rewritten `versions.txt`. Recovered source-verified: MG5_aMC
  **3.7.0**, Pythia8 **8.317**, LHAPDF **6.5.6**, PDF **NNPDF31_lo_as_0130 /
  lhaid 315200 / member 0**, CKKW-L **kT-Durham** merging at **TMS = 5.0 GeV**
  (`Merging:doKTMerging = on`, MLM off), three UFO models
  (`sm_onia`, `sm_mumass`, `sm_upsilon_family`, all `-c_mass`), and every
  per-region cut, seed and cross-section. So the claim immediately above — "no MG5
  process card, no run card, no Pythia8 CKKW-L merging configuration ... exists in
  version control" — **is no longer true**. Do not cite it as current.
  **What replaced it is two gaps one step later, both newly found:** the
  HepMC->HDF5 converter that actually built the priors was never committed
  (`lhe_to_prior_hdf5.py` reads LHE, but the priors came from showered HepMC), and
  the 4-D per-component reweighter that built the 1M file was never committed
  either (`reweight_prior.py` is a different, 1-D program). The `--ref-cache`
  argument is recovered (`selected_split_0ed04817d72bb34f84ed.npz`, sha256
  `ffb18e03...`) but the file is gone — 172 candidate `.npz` files hashed, none
  matches. So the sentence "`scripts/prior_build/` covers everything downstream of
  the LHE files" is also wrong: it covers a path the priors did not take. Full
  detail in the Session 28 entry in §11 and in `docs/mg5_vm_runbook.md` §7.
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

### 9.2 Joint program layout (current, after the Session 24 restructure)

- `scripts_joint/run_joint.py` — the single training entry point.
  `--run <ID>` resolves `configs_joint/cms_Joint_<ID>.yaml` (`--run E`,
  `--run C_fullScale`, `--run abNarrow`); `--config <path>` still works.
- `scripts_joint/joint_model.py`, `joint_trainer.py`, `joint_data.py`,
  `joint_metrics.py` — model, stage schedule + gated selection + resume,
  region/split contract, gate metrics.
- `scripts_joint/make_plots.py`, `dashboard.py`, `upsilon_transfer_test.py` —
  run-agnostic tooling, each requires `--run-dir`.
- `scripts_joint/upsilon/` — the Upsilon decode/compare/evaluate scripts.
  These used to live under `outputs/cms_Joint/Upsilon_zero_shot/`; executable
  code no longer lives under `outputs/`.
- `scripts_joint/build_ab_priors.py` — builds the §7.1 A/B prior pair.
- `scripts_joint/_retired_launchers/` — the old per-run `run_a.py` .. `run_f.py`
  shims, superseded by `--run`. Pending deletion.

### 9.3 Historical repository state (Program A / B era; superseded)

Kept for provenance. Paths and machine references below are the macOS era.

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
  (diverged, see §5.4), Jpsi_v3.9_F1_restricted (aborted pre-cut warmup,
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

## 10. Environment, conventions and guardrails

### 10.1 Environment and tooling

**Current machine (Windows, since 2026-08-21).** artifact-measured from
`outputs/cms_Joint/Run_F/provenance.json` and the 2026-09-04 session:

- Windows 10 (`desktop-otag01s`), repo at `C:\Users\AhrixMarin\Desktop\otus`.
- Conda env `cms`: Python 3.10.20 (Anaconda), torch 2.12.0+cu126, numpy 2.2.5,
  h5py, scipy, uproot, awkward, matplotlib. Activate with
  `(D:\Miniconda\shell\condabin\conda-hook.ps1) ; (conda activate cms)`.
- CUDA GPU with 11.99 GB total; runs cap the allocator via
  `cuda_memory_limit_gb` (Run F requested 11.0 GB).
- Joint full-pass epochs cost roughly 110 s each at batch 24,576.
- Windows OpenMP: `KMP_DUPLICATE_LIB_OK=TRUE` is set by the launchers; see
  `scripts_joint/README.md` for the Error #15 notes.

**Historical (macOS/MPS era, before 2026-08-21).** Kept for reading old
session entries:

- Interpreter `/opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python`
  (Python 3.10.20, torch 2.12.1 MPS, numpy 2.2.5). `.venv` instructions in
  README_MPS.md were the portable alternative.
- MG5_aMC 3.7.0 at `~/MG5_aMC_v3_7_0` with the macOS/LHAPDF fixes documented
  in `docs/jpsi_prior_rebuild_runbook.md`. **The prior-generation toolchain
  lives on that machine, not this one** — rebuilding a prior means going back
  to it, or to the Linux MG5 box referenced in `docs/sm_onia_rebuild.md`.
- Pyflakes is installed in the cms env (`python -m pyflakes ...`).
- Tests: `python -m unittest discover -s tests`.
- Full training runs must be explicitly authorized; prefer preflight/dry-run.

### 10.2 Conventions and guardrails

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

## 11. Session log

> **All dates and times in this file are local, America/Los_Angeles
> (PDT, UTC-7)** — the timezone of the Windows GPU host and of the
> generation VM. Corrected 2026-09-04: entries for Sessions 25 and 26 were
> first written from a UTC clock and dated 2026-09-05; they are the same
> working day as Session 27 and now read 2026-09-04. When comparing a
> timestamp across machines, compare epoch seconds, not formatted strings.

Preserved in the order originally recorded. Two known irregularities, left
uncorrected because the underlying dates cannot be verified: some 2026-08-14
entries appear after 2026-08-15 entries, and the four joint-program entries
before Session 24 were never assigned session numbers.

| # | Program | Outcome |
|---|---|---|
| 1 | A | Read-only failure investigation; run matrix rebuilt from surviving NPZs |
| 2 | A | Read the paper in full; proposed the F1-F3 fix roadmap |
| 3 | meta | Created this file |
| 4 | meta | conda -> .venv environment conversion (macOS era) |
| 5 | A | E1 kinematics; v3.9 F1-restricted pilot launched, then terminated on request |
| 6 | A | CMS-vs-MG5 mismatch quantified; Z->ee verified healthy as the contrast case |
| 7 | B | MG5 on macOS fixed; J/psi prior rebuilt with massive muons + smearing + reweighting |
| 8 | B | New-prior vs CMS comparison figure |
| 9 | B | Archived all Program A configs and outputs |
| 10 | B | Paper-faithful training on the new prior, 20% data; converged |
| 11 | B | Diagnosed the density failure: the vanilla objective is mass/correlation-blind |
| 12 | B | Prior low-pT support hole closed (ptj 10 -> 5) |
| 13 | meta | Repository cleanup and reorganisation |
| 14 | B | Three-term cosine loss implemented (no mass terms) |
| 15 | B | Run D loss extensions (pair SWD, cycle mass Huber) |
| 16 | B | Run D completed; SOTA exploration begun |
| 17 | B | SOTA Tier A / Run E implemented |
| 18 | B | Six smoke ablations; full Run E config tuned |
| 19 | B | Power outage; `--resume` implemented |
| 20 | B | Stage-2 instability fixed; clean restart |
| 21 | B | **SOTA Run E final results** — best single-region result |
| 22 | B | SOTA Run F (flow matching) implemented and launched |
| 23 | B | SOTA Run F completed; does not beat Run E |
| — | C | Joint Run D prepared with the showered inclusive Z prior |
| — | C | Joint Run D completed; first Upsilon transfer evaluation |
| — | C | Joint Run E full-pass epoch implementation |
| — | C | Joint Run F plateau review; latent mass gate failing |
| 24 | C | Repo repair, joint restructure, prior-width A/B built |
| 25 | C | **Prior-width A/B completed and read out** — smeared 34/35, narrow 0/35 |
| 26 | C | Identity/floor gauges implemented; paired-closure scorer built; noise schedules split |
| 27 | meta | MG5 generation VM: toolchain installed and verified end to end; `versions.txt` written |
| 28 | meta | **VM rebuilt on 24.04; 0j1j provenance recovered and the toolchain matched exactly** — cards for all 7 samples in the tree; MG5 3.7.0 / Pythia8 8.317 / LHAPDF 6.5.6 / NNPDF31 315200 all installed and verified; CKKW-L kT settled; smoke test shows the previous VM's 17σ offset was entirely its wrong PDF; `--ref-cache` found but its file is gone; two uncommitted downstream tools discovered |
| 29 | meta | Unified priors built: J/psi, Z, Upsilon (see the Session 29 entry) |
| 30 | C | **Step 3 done: the per-event closure test, run on our own model** — ppzee loader branch (CMS path byte-identical, proven by Run E's contract hash); `residual_rms_vs_identity` **0.9843** against identity 1.0, an oracle floor of 0.81 and the upstream encoder's 3.33; all four leakage checks clean; the gain is a global mass-bias removal, not a per-event inverse; decoder gauge 0.000 on pair pT from a zero-recoil truth |


- **2026-08-14/15 — Session 1 (read-only failure investigation).** Recorded git
  state; read AGENTS.md + all instructions; complete file inventory of repo and
  reference mirror; read all core scripts, configs, tests, reference PDFs
  (ghostscript), and notebooks (cells + outputs); built the run matrix from the
  surviving mass_histograms.npz (custom Node NPZ parser); measured W1/KS/
  peak statistics for every recoverable run; audited caches, configs, numerics
  (stable mass formula), losses, stochastic evaluation, and Z-vs-J/psi
  differences; delivered the full diagnostic report (sections A-M). Findings
  distilled into §4, §5, §6. Marked the session goal complete.
- **2026-08-15 — Session 2 (paper reading + fix design).** Discovered new
  docs/ (paper + v3.8 runbook), commit a640077, and the executed
  Jpsi_v3.8_vanilla_paper_all_data_seed0 run; read arXiv:2101.08944v2 in
  full (main text + supplementary statistics + lambda/beta ablations); measured
  v3.8's divergent results (§5.4); proposed the paper-grounded fix roadmap
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
  metrics recorded in §9) and finalized the fix implementation: v3.10
  config gained the re-weighted checkpoint-selection score (§6.3 finding),
  dry-run validated; wrote docs/Jpsi_F1_fix_runbook.md with all run/eval
  commands; committed. No training runs in progress — the user runs the fix
  later.
- **2026-08-15 — Session 5 (E1 kinematics + v3.9 F1-restricted pilot).** Found
  the committed-but-unrun v3.9 F1 design (commit 48fb0ca31: filter_theory_prior
  + v3.9 config). On this machine the `cms` conda env is the interpreter
  (§10). Ran E1: prior kinematics (both MG5 files), CMS trigger floor, and
  signal-region pair kinematics; all numbers in §5.5; answered §6.1 and §6.5.
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
  scripts/diagnostics/jpsi_f32_check.py; findings in §5.7. Then, at the user's request,
  ran the same methodology on the Z->ee pair: healthy on every dimension
  (prior is a real 4-GeV-wide lineshape at the pole, kinematics matched at
  the few-% level, 93.4% support coverage, 100% pass rate, converged v4/v5
  runs). One real caveat found: the skim's endcap electron energy scale is
  +1.9% high (endcap-only mode 92.94 vs pole 91.19; barrel-only 91.18) —
  a skim-level 2012 ECAL calibration artifact, absorbable, not an OTUS bug.
  Wrote scripts/diagnostics/zee_health_check.py; findings in §5.8.
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
  to §4.2. Causal probe (round 2): fine-tuned the frozen-decoder model's
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
  * Memory file trimmed: superseded §7 F1-F3 roadmap, stale §10 environment
    notes, and outdated §9 quick-reference entries were replaced with current
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

- **2026-09-04 — Session 24 (repo repair, joint restructure, prior-width A/B prepared).**
  Context: the user removed Run F after it failed on the chosen parameters and
  asked for a cleanup, bug fixes, and the controlled experiment.
  * source-verified: commit `4f7b242` ("new start") deleted more than the Run F
    artifacts. `scripts_joint/joint_model.py` (196 lines) and the four Upsilon
    evaluation scripts under `outputs/cms_Joint/Upsilon_zero_shot/`
    (`decode_prior`, `evaluate_z_to_x`, `compare_prior_cms`,
    `compare_retry_metrics`, 1,609 lines) went with it, leaving `scripts_joint/`
    unimportable. All five recovered from `530dfee`; the Upsilon scripts now
    live in `scripts_joint/upsilon/` instead of under `outputs/`.
  * source-verified: four defects fixed in `scripts_joint/joint_trainer.py`.
    (a) `stage_best_score` was reset to `math.inf` on resume, so a later worse
    epoch overwrote a better stage-best — this is the mechanism behind the
    Session-23 observation that stage-2 best was epoch 192/3.498 while history
    held epoch 112/3.276. It is now reconstructed from history when resuming
    inside a stage. (b) `global_best_score` was only reconstructed from
    gate-passing rows, so with `hard_gates: false` a resume could overwrite
    `best_model.pt` with a worse checkpoint; it now honors the flag and only
    trusts a historical score when `best_model.pt` still exists. (c) resuming
    exactly at a stage boundary skipped the "restore this stage's best weights"
    step that the non-resume path performs, so the next stage started from
    last-epoch weights; the resume path now mirrors it. (d) `hard_gates` was
    re-read from config inside the epoch loop; hoisted.
  * source-verified: the dashboard drew the penalized `selection_score`
    (+1e6 on any gate failure) and `raw_worst_region_score` (~4) on one shared
    axis, flattening the raw curve. Split into a raw-score chart and a
    gate-pass chart.
  * Repo hygiene: added `.gitattributes` (`text=auto eol=lf`, CRLF for
    `.ps1`/`.bat`, binary for data/checkpoint types). This collapsed the
    161 phantom-modified files — CRLF churn from the macOS -> Windows move with
    no `core.autocrlf` — leaving only real changes in `git status`.
  * Restructure: the seven `run_a.py` .. `run_f.py` shims are superseded by
    `run_joint.py --run <ID>` (resolves `configs_joint/cms_Joint_<ID>.yaml`);
    they are parked in `scripts_joint/_retired_launchers/` pending deletion.
    `run_f_dashboard/plots/upsilon_test.py` became run-agnostic
    `dashboard.py` / `make_plots.py` / `upsilon_transfer_test.py` with required
    `--run-dir`. Four duplicate `import os` statements removed.
  * Prepared but NOT run: `scripts_joint/build_ab_priors.py` plus
    `configs_joint/cms_Joint_abNarrow.yaml` and `cms_Joint_abSmeared.yaml`
    (see §7). Nothing was trained this session; the Windows GPU host runs it.
  * Blocker for the user: a stale `.git/index.lock` blocks every git write in
    the repo and could not be removed from this session (no delete permission).
    Delete it before committing.

- **2026-09-04 — Session 25 (prior-width A/B read-out).** The user ran both arms
  on the Windows GPU host between 2026-09-04 01:40 and 2026-09-04 18:36 and
  asked what the result means. Nothing was trained, moved or deleted from this
  session; the analysis is read-only over `history.json`,
  `joint_evaluation.json`, `config.resolved.json`, `provenance.json`,
  `data/cms_jpsi_ab_manifest.json` and the two region caches.
  * artifact-measured: J/psi gate passes narrow 0/35, smeared 34/35; Z region
    35/35 in both. Configs differ only in the prior file (source-verified diff).
  * artifact-measured: the identity map z~ = x passes every strict J/psi latent
    target on the smeared prior (KS 0.0359, W1 0.00219, width_rel 0.071) and
    fails all three on the narrow prior (0.3702 / 0.01736 / 0.910). The trained
    smeared model beats identity by less than the gate width; the trained
    narrow model is worse than identity (KS 0.4148).
  * artifact-measured: narrow prior IQR-based width 0.16 MeV with 81% of events
    within +-5 MeV of the J/psi mass (near-delta + ~17% continuum pedestal);
    smeared prior 27.99 MeV vs CMS data 30.03 MeV.
  * Conclusions and the revised plan are in §7.2 and §7; new open questions
    are §6 items 8 and 9. §1 and §6 item 6 updated.

- **2026-09-04 — Session 26 (identity gauges, paired closure, split noise).**
  Implementation session following §7.2. No training was started and no
  existing output, checkpoint or data file was modified.
  * New: `scripts_joint/identity_reference.py`, `identity_baseline.py`,
    `paired_closure.py`, `tests/test_identity_reference.py`,
    `tests/test_paired_closure.py`,
    `configs_joint/cms_Joint_abNarrowSplit.yaml`. Modified:
    `joint_metrics.py`, `joint_model.py`, `joint_trainer.py`, `run_joint.py`,
    `scripts_joint/README.md`. Full rationale in §7.3 and §7.4.
  * source-verified: `joint_metrics` now imports its 1-D statistics and the
    equal-count subsetting from `identity_reference` instead of defining them,
    so the offline re-scorer and the trainer cannot drift on sampling
    geometry. `ks_distance` / `wasserstein_1d` are re-exported, keeping
    `plot_joint_run.py`'s import working.
  * source-verified: encoder and decoder noise schedules are now independent.
    `JointDimuonAutoencoder.set_component_noise_multipliers` sets them
    separately; `set_noise_multipliers` remains and delegates, so
    `restore_joint_checkpoint` and every pre-existing checkpoint behave
    unchanged. Stages read `encoder_core_noise_multiplier`,
    `encoder_tail_noise_multiplier`, `decoder_core_noise_multiplier`,
    `decoder_tail_noise_multiplier`, each defaulting to the shared
    `core_noise_multiplier` / `tail_noise_multiplier`, so every existing config
    reproduces exactly. All four resolved values are written to each history
    row and to the checkpoint payload.
  * source-verified: the split contract hash covers data identity only
    (`joint_data.build_joint_split_manifest` excludes execution state and
    never hashes `stages`), so the new stage keys cannot invalidate a resume.
  * Verification actually performed this session: 20 NumPy unit tests pass;
    every touched module byte-compiles; `cms_Joint_abNarrowSplit.yaml` resolves
    through `load_config` with the expected stage list, prior file and gates;
    `identity_baseline.py` reproduces the A/B retrofit table; the paired-closure
    scorer reproduces the upstream ppzee numbers. NOT performed: anything
    requiring torch or a GPU. Before the next run:
    `python -m pytest tests/` and
    `python scripts_joint/run_joint.py --run abNarrowSplit --device cuda --dry-run`
    on the Windows host.
  * Retained artifact: `outputs/cms_Joint/paired_closure_upstream_ppzee.json`.
  * Later in the same session, for the MG5 VM: new `CLAUDE.md` at the repo root
    (auto-loaded by Claude Code, so any session anywhere inherits the read-first
    / update-last / label-evidence / no-unattended-training conventions),
    `docs/mg5_vm_runbook.md`, `scripts/prior_build/prior_stats.py` and
    `scripts/prior_build/reference/jpsi_0j1j_reweighted_selected.json`.
  * artifact-measured: `prior_stats.py` reproduces the A/B narrow arm from the
    cached splits (999,957 events) as mass mean 3.09550 GeV, std 14.6997 MeV,
    IQR width 0.1645 MeV, muon-pT median 12.4074, pair-pT median 26.0556,
    |eta| median 0.83813 — agreeing with the manifest to 0.005 MeV, which
    validates both the tool and the shipped reference JSON.
  * source-verified: two mass conventions coexist and must not be mixed.
    `build_ab_priors.py` (and therefore every A/B number and the reference
    JSON) uses massive-muon momentum-based mass, `stable=True`;
    `cms_data.filter_theory_prior` applies its mass window with stored
    energies and massless daughters. For the 0j1j family they happen to agree
    to 0.0001 MeV, but that is a property of that file, not a rule.

- **2026-09-04 — Session 27 (MG5 generation VM, runbook sections 3-6 only).**
  Executed `docs/mg5_vm_runbook.md` sections 3 and 6 on the Linux generation VM
  (`ahrimarin-virtual-machine`). No physics sample was generated: the 0j1j
  process and run cards are still unrecovered, and generating without them
  would produce a sample that cannot be attributed. Nothing under `data/` or
  `outputs/` was touched and no git command was run.
  * source-verified, installed and confirmed by running: Ubuntu 22.04.5 LTS,
    gcc/g++/gfortran 11.4.0, GNU Make 4.3, Python 3.10.12, git 2.34.1.
    MG5_aMC **3.7.3** (launchpad tarball, sha256 5964645515a1...769b) unpacked
    to `~/MG5_aMC` on the VM's **local disk**, never onto `/mnt/hgfs`.
    Through MG5 so the interfaces are wired: LHAPDF **6.5.5**, HepMC **2.06.09**,
    Pythia8 **8.317**, MG5aMC_PY8_interface **1.3**. All four reported
    "successfully installed in ./HEPTools". Paths persisted into
    `~/MG5_aMC/input/mg5_configuration.txt` with `set ... --save`.
  * source-verified: `install lhapdf6` also pulled two PDF sets,
    NNPDF23_lo_as_0130_qed (lhaid 247000, LO, alphas(MZ)=0.130003, 101 members)
    and NNPDF23_nlo_as_0119_qed (lhaid 244800). Neither is asserted to be the
    set the 0j1j priors used; that is still unknown.
  * artifact-measured, end-to-end verification run `~/mg5work/smoke_dy`
    (`p p > mu+ mu-`, 8 TeV, 1000 events, `pdlabel = lhapdf`, `lhaid = 247000`):
    cross-section 554.8 +- 3.237 pb; 1000 `<event>` records in
    `unweighted_events.lhe.gz`; 1000 `E` records in
    `tag_1_pythia8_events.hepmc.gz` with 1057 particles in the first event,
    against ~6 at parton level — so the shower genuinely ran. This is a
    toolchain smoke test and is **not** a physics sample; it has no relation to
    any file in `data/`.
  * Written: `scripts/mg5_cards/0j1j/versions.txt` (new directory), recording
    every version above plus the PDF set name and member id, the compilers, the
    OS, the install commands, and the verification-run numbers. Uncommitted —
    the repo has a stale `.git/index.lock` and this session ran no git commands.
  * source-verified: FastJet is **not** installed; MG5 falls back to its bundled
    fjcore. Section 6 does not require it, but if a recovered 0j1j run card
    needs a real FastJet the environment must be rebuilt and `versions.txt`
    regenerated.
  * Runbook section 6.1 (added to the runbook mid-session) — the two items
    that are not VMware GUI actions were done. source-verified:
    `automatic_html_opening = False` is persisted at
    `~/MG5_aMC/input/mg5_configuration.txt` line 116, so runs no longer open
    Firefox on the VM desktop. Note the `--save` flag alone does **not** persist
    this option; it needs `set automatic_html_opening False` followed by
    `save options`. The stray Firefox the smoke run opened has been closed.
  * artifact-measured, guest clock (runbook section 9): NTP service active,
    "System clock synchronized: yes", and at epoch 1788591162 the guest agreed
    with both the Cloudflare and the Google HTTP `Date` header to **0 seconds**.
    The clock is correct.
  * **The dating inconsistency in this repo is a timezone artefact, not a wrong
    clock — do not "fix" it by editing dates.** The VM is
    `America/Los_Angeles` (PDT, UTC-7), so epoch 1788591162 is simultaneously
    2026-09-04 23:52 local and 2026-09-05 06:52 UTC. That is why Session 26
    appears as 2026-09-04 in the runbook header and as 2026-09-05 in sections
    7.2-7.4 here: same instant, two conventions. This session's entries are
    dated **2026-09-04 local**. proposal: state the convention once, near the
    top of this file, and record epoch seconds alongside any date that has to
    be compared across the host, the VM and the share.
  * Still open, unchanged by this session: the 0j1j process card, run card and
    Pythia8 CKKW-L merging configuration, and the `reweight_prior.py
    --ref-cache` argument, all pending recovery from the old macOS machine
    (runbook sections 0 and 7, memory.md 9.1). The runbook's section 4/5 items
    (Claude Code CLI, shared folder) were already in place. The **snapshot was
    not taken** — runbook 6.1 requires it with the VM powered off, which is a
    VMware GUI action outside this session. The `sm_onia-c_mass` UFO model is
    also still absent from `models/`; it must come from the old machine.

- **2026-09-05 — Session 28 (MG5 generation VM rebuilt on Ubuntu 24.04; 0j1j
  provenance recovered; toolchain rebuilt to match the original).** Executed
  `docs/vm_rebuild_2404_prompt.md` on the reinstalled generation VM. **No physics
  sample was generated.** The only generation was the step-7 toolchain smoke test
  (2 × 1000 Drell-Yan events), which has no relation to any file in `data/`.
  Nothing under `data/`, `outputs/` or any checkpoint was modified, and no git
  command was run.

  * **Why the OS moved.** The 22.04 VM of Session 27 no longer exists; it was
    reinstalled as Ubuntu 24.04 LTS because 22.04 had a networking and
    shared-folder fault. Everything Session 27 installed (MG5 3.7.3, LHAPDF
    6.5.5, Pythia8 8.317, HepMC 2.06.09) went with it.
  * source-verified, this VM: Ubuntu 24.04.4 LTS, kernel 7.0.0-31-generic,
    hostname `ahrimarin-VMware-Virtual-Platform`, Python 3.12.3, 8 vCPU,
    15 GiB RAM, 126 GiB root volume (109 GiB free — below the runbook's
    150-200 GB spec), open-vm-tools 2:13.0.10.
  * artifact-measured: **both 22.04 faults are gone.** Network clean —
    `launchpad.net` HTTP 200 in 0.776 s, `lhapdf.hepforge.org` HTTP/2 301, DNS
    via systemd-resolved. Share mounted and writable (wrote and deleted a
    scratch file under `docs/`). Guest clock **0 s** against both the Cloudflare
    and Google HTTP `Date` headers at epoch 1788634683, NTP active.
  * source-verified, and a **deviation from runbook §5**: `/etc/fstab` has no
    hgfs entry and none was added. open-vm-tools mounts the share at boot itself
    — this boot started at epoch 1788634242 and the journal shows
    `vmtoolsd[5577]: /usr/bin/vmhgfs-fuse: 0 - HGFS FUSE client enabled` at
    11:53:11 of the same boot; systemd tracks it as `mnt-hgfs.mount` with an
    empty `FragmentPath`. An fstab line would race that mount. No reboot was
    performed to test persistence — a reboot kills the CLI session doing the
    testing — so the evidence is this boot, which was itself a fresh boot.
  * **The one hard constraint on unattended work: `sudo` on this VM requires a
    password** (`sudo -n true` → "sudo: a password is required"), so a CLI session
    cannot install packages itself. The user ran the apt command in a native
    terminal; a line-wrap meant only its first line reached apt, so
    `python3-venv`, `libbz2-dev` and `git` are still absent. **Nothing needed
    them** — MG5, LHAPDF and Pythia8 require none, and with no `pip` present PEP
    668 was never actually hit. Once `build-essential`/`gfortran`/`python3-dev`
    landed, prompt steps 4, 5, 6 and 7 all completed.
  * source-verified: gcc/g++/gfortran **13.3.0**, GNU Make 4.3 — the expected 13.x
    generation, against 11.4.0 on 22.04.
  * source-verified: **no `-fallow-argument-mismatch` was needed.** Both LHAPDF
    6.5.6 and Pythia8 8.317 built clean under gfortran 13.3.0. The anticipated
    24.04 build problem did not materialise.

  ### The toolchain, installed and matched (prompt steps 4-7)

  Every component now **matches the recovered original**, which the 22.04 build
  did not. source-verified on this VM unless noted.

  * **MG5_aMC 3.7.0.** The URL in the rebuild prompt
    (`.../3.0/3.7.x/+download/MG5_aMC_v3.7.0.tar.gz`) returns **HTTP 404** —
    3.7.0 is published under the **3.6.x** series, not 3.7.x. Correct URL:
    `https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.7.0.tar.gz`.
    sha256 **computed here** on the 31,786,016-byte download:
    `b151dee0a46bfd625959ca0202aa5f3a26ed5492a0fb98e1f3c164c860947870`.
    artifact-measured and the strongest single check of the session:
    `sha256(~/MG5_aMC/VERSION)` = `d8521e7a…fb5e`, **byte-identical to
    `<OLD_TREE>/VERSION`** — the launchpad tarball is the same 3.7.0 release the
    original installation used. Unpacked to `~/MG5_aMC` on local disk.
  * **MG5 3.7.0 runs under Python 3.12.3.** `import model sm` succeeds and all
    three recovered models import. The prompt's contingency (install python3.11
    alongside) was not needed and was not proposed. Note `./bin/mg5_aMC --version`
    does not exist in 3.7.0 — it errors with "no such option".
  * **Pythia8 8.317 — matched, and not by luck.** MG5 3.7.0 pins the installer to
    `HEPToolsInstaller_V168`, and `<OLD_TREE>/HEPTools/HEPToolsInstallers/VERSION`
    is exactly **168**. So `install pythia8` under 3.7.0 reproduces the original's
    Pythia by construction. This explains why Session 27 also got 8.317 and closes
    a question that build could not answer. (`--pythia8_tarball=` exists as a pin
    if ever needed.) HepMC **2.06.09** and PY8 interface **1.3** came with it, both
    matched; zlib 1.3.2.
  * **LHAPDF 6.5.6 — matched, but only by building it.** `help install` shows the
    MG5 installer takes **no version pin** for lhapdf6 (only `--force`,
    `--keep_source`) and hardcodes `LHAPDF-6.5.5.tar.gz`. So `install lhapdf6`
    gives 6.5.5 — exactly how the 22.04 build landed there. Route taken: ran
    `install lhapdf6` (6.5.5, kept but unused), then built 6.5.6 from source into
    `~/MG5_aMC/HEPTools/lhapdf6_py3_656` and repointed MG5 with
    `set lhapdf …` + `save options`.
    **24.04 gotcha:** LHAPDF's `configure` aborts with "Cannot find python in your
    system path" because 24.04 ships only `python3`. Fix without sudo:
    `./configure … PYTHON=/usr/bin/python3`.
  * **PDF NNPDF31_lo_as_0130, lhaid 315200, member 0 — matched and verified, not
    assumed.** `SetIndex: 315200`, `DataVersion: 1`, 101 members, `OrderQCD: 0`,
    `AlphaS_MZ: 0.1300000` read from the installed `.info`. This is the single most
    important correction to the previous build, which was on NNPDF2.3.
  * Ordering mattered: LHAPDF was repointed to 6.5.6 **before** installing Pythia8,
    so `HEPTools/pythia8/Makefile.inc` links `--with-lhapdf6=…/lhapdf6_py3_656`.
  * source-verified, cosmetic bug worth not chasing later:
    `input/mg5_configuration.txt` line 97 reads `hepmc_path = …/MG5_aMC/None` — the
    installer wrote the literal string "None". The build used the correct path and
    the shower emits valid HepMC, so nothing is broken. `fastjet = None`, matching
    the original.
  * **`auto_update = 0` is a new and load-bearing setting.** On startup MG5 3.7.0
    offers "New Version of MG5 available! Do you want to update your current
    version?" with a 60-second timeout. Accepting it would silently destroy the
    3.7.0 pin this whole exercise depends on. Also `automatic_html_opening = False`
    — and the Session 27 trap is confirmed again: MG5 explicitly printed that
    `save options` was required, so `--save` alone does not persist it.
  * **Models installed and numerically verified.** The three UFO models were copied
    **from the repo** (`scripts/mg5_cards/0j1j/models/`) into `~/MG5_aMC/models/`,
    so what was tested is the committed deliverable. Values, not just exit codes:
    `mdl_MJPSI = 3.0969`, `mdl_WJPSI = 9.29e-05`, `mdl_gJpsi = 0.01`,
    `mdl_MM = 0.105658` (the restrict-card value, **not** the 0.10566 default in
    `parameters.py`), `jpsiv` self-conjugate; Υ(1S) 9.4603 / 5.402e-05,
    Υ(2S) 10.0234 / 3.198e-05, Υ(3S) 10.3551 / 2.032e-05, all g = 0.01;
    `sm_mumass` MM 0.105658, MC 1.55.

  ### Toolchain smoke test — the PDF was the whole story

  artifact-measured. **Not a physics sample; no relation to any file in `data/`.**
  Two runs of `p p > mu+ mu-`, 1000 events, 8 TeV, MG5-default cuts
  (ptl 10, etal 2.5, drll 0.4), same seed 90501, in `~/mg5work/smoke_dy`.
  They differ in **one** variable, the PDF (CLAUDE.md §3).

  | run | lhaid | cross-section |
  |---|---|---|
  | run_01 | 315200 (NNPDF3.1) | **643.7 ± 4.012 pb** |
  | run_02 | 247000 (NNPDF2.3) | **556.9 ± 3.425 pb** |
  | 22.04 VM | 247000 (NNPDF2.3) | 554.8 ± 3.237 pb (MG5 3.7.3, gcc 11.4.0) |

  * run_01 vs run_02: +86.8 pb, combined error 5.275 pb → **16.5σ**
  * run_02 vs 22.04: +2.1 pb, combined error 4.713 pb → **0.4σ**
  * run_01 vs 22.04: +88.9 pb, combined error 5.155 pb → **17.2σ**

  **Conclusion: the PDF family change accounts for essentially the entire 17.2σ
  difference.** Held at the same PDF, this 24.04 / MG5 3.7.0 / gcc 13.3.0 build
  agrees with the 22.04 / MG5 3.7.3 / gcc 11.4.0 build to 0.4σ — statistically
  indistinguishable. So the MG5 downgrade and the compiler move are *measured* to
  be irrelevant at this sample size, and the previous VM's 554.8 pb was low for
  exactly one reason: NNPDF2.3 where the original used NNPDF3.1. Nothing was tuned
  to make any of these agree. run_01 also gave 1000 `<event>` records in the LHE,
  1000 `E` records in the HepMC, and 1718 particles in the first event against ~6
  at parton level, so the shower genuinely ran.

  ### What was recovered from the old macOS tree

  The tree is at `/mnt/hgfs/MG5_aMC_v3_7_0/MG5_aMC_v3_7_0` (doubled path — the
  zip unpacked one level deep; the sibling `__MACOSX/` is 14,727 AppleDouble
  `._*` files with no content). Read-only throughout; nothing was written to it
  or executed from it.

  * source-verified: **MG5_aMC 3.7.0**, from `VERSION` ("version = 3.7.0",
    sha256 `d8521e7a…fb5e`) and independently from the `<MGVersion>` block of all
    14 run banners. The directory name agreed but was not treated as evidence.
    **Session 27's 3.7.3 was the wrong target.**
  * source-verified: **Pythia8 8.317** from the banner of a log that actually ran
    (`z_mumu_inclusive_0j1j_8tev/Events/production_ckkwl_1m_a/tag_1_pythia8.log`).
    Session 27's `install pythia8` happened to pull exactly the right version.
  * source-verified: **LHAPDF 6.5.6**, PDF set **NNPDF31_lo_as_0130**, **lhaid
    315200**, **member #0**, from
    `jpsi_inclusive_0j1j_8tev/SubProcesses/P0_ccx_jpsiv_jpsiv_mupmum/G1/production_600k_log.txt`
    line 3 — tied to the production run, not to the tree. The same three lines
    appear in all five 0j1j production directories.
    **Session 27's NNPDF23_lo_as_0130_qed / 247000 was the wrong PDF family** —
    NNPDF2.3 vs 3.1. That is a physics error, not a version nit: it moves the
    parton luminosities and hence the pair-pT spectrum, which is a conditioning
    feature of the response (§7.10). Its 554.8 pb smoke number is not comparable
    to anything the original produced.
  * source-verified: MG5aMC_PY8_interface **1.3**, HepMC **2.06.09**,
    HEPToolsInstallers **168**. LHAPDF was **not** in `HEPTools` at all — the
    original pointed at a conda env
    (`lhapdf_py3 = /opt/homebrew/.../envs/cms/bin/lhapdf-config`).
  * source-verified: **no FastJet was configured** — no `fastjet` line in the
    original `input/mg5_configuration.txt`, so MG5 used bundled fjcore. Session
    27's open question on this is closed: fjcore is correct. `run_mode` and
    `nb_core` were never pinned either.
  * source-verified: **the CKKW-L question is settled**, from the *resolved*
    Pythia configuration MG5 handed to Pythia (`tag_1_pythia8.cmd`):
    `Merging:doKTMerging = on`, `Merging:TMS = 5.0` (MG5 translated
    `ktdurham = 5.0`), `Merging:Process = guess`, `nJetMax = 1`,
    `Dparameter = 0.4`, `nQuarksMerge = 4`, and `JetMatching:setMad = off` with
    the MLM lines commented out. ME side: `ickkw = 0`, `xqcut = -1.0`,
    `ptlund = -1.0`. **MLM was not used.** Note the hand-written
    `pythia8_card.dat` says `Merging:TMS = -1.0` — that is a placeholder, not the
    scale; read the `.cmd` file.
  * The Upsilon directories are named `*_matched_*` but their production runs are
    CKKW-L. The only MLM artifact in the tree is one abandoned 20k smoke
    (`upsilon_continuum_matched_0j1j_8tev/Events/smoke_matched_20k`,
    `ickkw = 1`, `xqcut = 3.0`, integrated weight 2.119e9 pb). The name is a
    leftover, not a description.
  * source-verified: **three UFO models, not one** — `sm_onia-c_mass` (J/psi
    signal), `sm_mumass-c_mass` (J/psi continuum, Z, Upsilon continuum),
    `sm_upsilon_family-c_mass` (Upsilon 1S/2S/3S). All three copied into the
    repo. `docs/sm_onia_rebuild.md` is confirmed correct, with one trap:
    `parameters.py` has `MM = 0.10566` but `restrict_c_mass.dat` sets
    `MM = 0.105658`, and the restrict card is what reached every run.
    Upsilon family: 1S 9.46030 GeV / 54.02 keV, 2S 10.0234 / 31.98 keV,
    3S 10.3551 / 20.32 keV, all with g = 0.01 like gJpsi.
  * source-verified: `Cards/run_card.dat` and the banner's `MGRunCard` are
    **identical except `iseed`**, which MG5 resets to 0 in `Cards/` after a run.
    Verified by diffing both for all seven samples. **The banner is the authority
    for the seed; `Cards/` is trustworthy for everything else.**
  * source-verified, per-region cuts (identical everywhere else: 8 TeV,
    `ptj = 5.0`, `etaj = 5.0`, `maxjetflavor = 4`,
    `dynamical_scale_choice = -1`, `bwcutoff = 15.0`, `use_syst = False`):

    | region | model | ptl | etal | drjl | mass window |
    |---|---|---|---|---|---|
    | J/psi signal | sm_onia | 3.0 | 2.5 | 0.4 | 2.9-3.3 |
    | J/psi continuum | sm_mumass | 3.0 | 2.5 | 0.4 | 2.9-3.3 |
    | Z / DY | sm_mumass | **20.0** | **2.6** | **0.0** | **65-120** |
    | Upsilon 1S/2S/3S | sm_upsilon_family | 3.0 | 2.4 | 0.4 | 8.5-11.5 |
    | Upsilon continuum | sm_mumass | 3.0 | 2.4 | 0.4 | 8.5-11.5 |

  * source-verified, runs/seeds/cross-sections. The **J/psi 1M prior is two runs,
    one seed each**: signal `production_600k` (ME iseed 33002, PY8 seed 330102,
    107.40636 pb) and continuum `production_400k` (34002 / 340102,
    69.6727414317805 pb). The **Z prior is four runs**, 3.2M events generated:
    `1m_a` (24001/240101, 883.139913 pb), `1m_b` (24002/240102, 882.94432),
    `1m_c` (24003/240103, 883.179881), `200k_d` (24004/240104, 884.0002489999999)
    — a 0.12% spread across chunks, a useful internal consistency check.
    Upsilon: 1S 300k (13001, 2794.494 pb), 2S 100k (13002, 4724.0091),
    3S 80k (13003, 7425.3753), continuum 1m_a (13004, 791.8371264152) and
    400k_b (13304, 792.1509788390001).
  * source-verified and important: **the Upsilon showers were never seeded.**
    Their `pythia8_card.dat` has no `Random:setSeed`/`Random:seed`, and no
    `Random:*` line appears in their `.cmd` or in their logs' changed-settings
    table. So the Upsilon shower stage is not reproducible even in principle from
    the recorded cards. The J/psi and Z showers are seeded.

  ### The `--ref-cache` — found, and it does not help as much as hoped

  artifact-measured, read out of the HDF5 attributes of
  `data/cms_jpsi_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_3p0369_3p1569_1M_reweighted.hdf5`
  (no h5py on the VM; HDF5 stores string attributes uncompressed in the object
  header, so they were read from the raw bytes).

  * `reference_file` =
    `/Users/ahrimarin/Desktop/otus_on_data/outputs/cms_JpsiDoubleMuons/archive/.plot_cache/selected_split_0ed04817d72bb34f84ed.npz`,
    `reference_file_sha256` =
    `ffb18e03f5af1f2aa527f32abf61caeb76a06d0ca7cd82a9ecf0b833180e19e3`.
    The `reference_note` records it was itself a fallback: the run wanted
    `outputs/cms_Joint/.region_cache/jpsi/selected_split_319d77365faebcd0ac9a.npz`,
    found it absent, and substituted the full CMS J/psi cache.
  * artifact-measured: **that file is gone.** `outputs/cms_JpsiDoubleMuons/archive/.plot_cache/`
    does not exist; only an empty `archive/` remains. All **172** `.npz` files
    under `outputs/` and `data/` were hashed and **none** matches `ffb18e03…`.
  * The `319d77…` file *does* exist now (118,850,382 B, sha256 `0b85a5d8…`) and is
    a valid ref-cache shape (`x_train/x_val/x_test/z_train/z_val/z_test`), but it
    is a different file by hash, and its JSON sidecar keys its fingerprint on
    `data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5` — the **ptj=5 Program B prior**,
    not the 0j1j one. Substituting it would silently change the reweighting.

  ### Two uncommitted tools — a gap that was not previously recorded

  * **The priors were built from showered HepMC, not from LHE.** The 500k J/psi
    prior records `source_hepmc` naming eight files
    (`…/production_600k/PY8_parallelization/split_{0..3}/events.hepmc` and the
    same for `production_400k`), each with a sha256, and
    `event_level = "post-parton-shower stable muons with lepton QED FSR"`.
    `scripts/prior_build/lhe_to_prior_hdf5.py` takes `--lhe` and writes an
    `lhe_file` attribute the shipped prior does not have, and **no file in the
    repo reads HepMC at all**. So a HepMC->HDF5 converter existed and was never
    committed. Those eight HepMC files are also gone — there is no
    `PY8_parallelization` directory and no `events.hepmc` anywhere in the tree.
  * **`reweight_prior.py` is not the program that made the 1M prior.** The repo's
    script is 1-D on muon pT (401 log-pT bins, `gaussian_filter1d`, default max
    weight 20). The shipped prior records
    `reweight_method = "4D histogram density ratio: log10(lead muon pT),
    log10(sublead muon pT), log10(pair pT), max |eta|; per-component; clipped;
    sampled with replacement"`, `reweight_bins = {lead_pt 80, sublead_pt 60,
    pair_pt 60, max_eta 12}`, `max_weight_after_clip = 200.0`, and attributes the
    script never writes. Two different programs; the 4-D one is not in the repo.
  * **Consequence: runbook §7 Stage B cannot pass as written, and the runbook's
    instruction was itself wrong** — it names `reweight_prior.py`. Even with the
    right cache, running it would produce a file that is not the 1M prior.
    Stage A also needs redefining, since it is specified against an LHE path the
    priors did not take.

  ### What the surviving artifacts do support

  * artifact-measured: the 500k prior on disk hashes to exactly
    `1620f6bbfd9881243d792c7bf2a3db2a71902e3e59ffcaeb06500e8f1d99e1cf`, the
    `source_file_sha256` the 1M prior recorded for its input. **That link is
    verified intact.**
  * artifact-measured: the 1M prior's self-recorded validation block —
    mass mean 3.09550142288208 GeV, mass std 0.014705080538988113 GeV,
    muon pT median 12.40745735168457, pair pT median 26.055429458618164,
    minimum pair pT 5.2037034034729, pair-pT 5-10 GeV fraction 0.022954,
    n 1,000,000 — matches the runbook §7 Stage B reference table exactly. Also
    `component_counts = {"jpsi": 850000, "continuum": 150000}` (an 85/15
    mixture), `artificial_smearing_applied = false`,
    `upsilon_tuning_used = false`, produced 2026-08-31T21:58:19Z.
  * artifact-measured: the **J/psi LHE files survive** on the share
    (`production_600k` 104,505,851 B; `production_400k` 57,645,083 B) together
    with `tag_1_pythia8.cmd` and `run_shower.sh`. The Z and all Upsilon samples
    retain **both** LHE and showered HepMC (~10.2 GB for the Z alone). So the
    J/psi arm — the one carrying the Program C prior and both A/B arms — is
    exactly the arm whose showered output is missing.
  * Event bookkeeping from the 500k prior: jpsi 581,216 processed -> 436,815 after
    fiducial; continuum 398,845 -> 91,699. Sum 528,514, which is the "500k" in
    the filename. Processed < requested because CKKW-L vetoes events below the
    merging scale before writing.

  ### Deliverables written this session

  * `scripts/mg5_cards/0j1j/` — 120 files. Per-sample subdirectories
    `jpsi_signal/`, `jpsi_continuum/`, `dy/`, `upsilon/{upsilon1s,upsilon2s,upsilon3s,continuum}/`,
    each with `proc_card_mg5.dat`, `run_card.dat`, `param_card.dat`,
    `pythia8_card.dat`, the run banner, `tag_1_pythia8.cmd` and `run_shower.sh`;
    plus `dy/additional_chunks/` and `upsilon/continuum/additional_chunks/` for
    the remaining chunk banners, and `models/{sm_onia,sm_mumass,sm_upsilon_family}/`
    (UFO `*.py` + all `restrict_*.dat`; `py3_model.pkl` and `__pycache__`
    deliberately excluded as interpreter-specific caches).
  * `scripts/mg5_cards/0j1j/SOURCE.md` — absolute source path, sha256 and mtime
    for every copied file, plus the sha256 of banners read but not copied, and the
    surviving-event-file inventory.
  * `scripts/mg5_cards/0j1j/versions.txt` — rewritten, 730 lines, four separated
    sections (`[this VM]`, `[previous VM … SUPERSEDED]` kept verbatim,
    `[recovered from the original toolchain]`, `[deltas]`) plus
    `[what this file does not establish]`.
  * `docs/mg5_vm_runbook.md` — updated §0, §2, §3, §5, §6 (new §6.0 pin table and
    §6.2 models), §7 (Stage B rewritten) and §10 (checklist marked up). 22.04 text
    struck through, not deleted.
  * Also written on the VM's local disk (not in the repo): `~/MG5_aMC` (the
    verified 3.7.0 install) and `~/mg5work/smoke_dy` (the two smoke runs). Nothing
    was generated onto `/mnt/hgfs`.
  * **Nothing committed.** The repo has a stale `.git/index.lock`, `git` is not even
    installed on this VM, and this session ran no git command.
  * **Still outstanding for a human:** take the powered-off VMware snapshot — the
    toolchain has now verified, so this is the moment it is worth taking. Suggested
    name `toolchain-verified-2026-09-05`, description MG5 3.7.0 / Pythia8 8.317 /
    LHAPDF 6.5.6 / HepMC 2.06.09 / NNPDF31_lo_as_0130 315200.

  ### Assessment

  hypothesis: **this VM can at best *regenerate* the 0j1j priors, not *reproduce*
  them** — the materially weaker claim runbook §7 says must be stated as such in
  the paper. Note what this verdict now does *not* rest on: the toolchain is no
  longer the problem. Every component matches the original exactly (MG5 3.7.0,
  Pythia8 8.317, LHAPDF 6.5.6, HepMC 2.06.09, PY8 interface 1.3, installer V168,
  NNPDF31_lo_as_0130/315200/0), the models import with verified numbers, and the
  smoke test shows the residual OS-layer differences — MG5 3.7.3→3.7.0 and
  gcc 11.4→13.3 — are worth 0.4σ, i.e. nothing. The generator stage is in good
  shape: fully specified, with the J/psi and Z LHE files and shower seeds both
  surviving.

  What blocks *reproduction* is now entirely downstream of the generator, and all
  three reasons are missing artifacts rather than mismatched software:
  (1) the HepMC→HDF5 converter was never committed, and its eight `events.hepmc`
  inputs are gone — their sha256s are recorded, so any reimplementation can be
  checked against the 500k prior but not against the intermediates;
  (2) the 4-D per-component reweighter was never committed either, so the *method*
  is recoverable from the recorded attributes but the code is not;
  (3) the reweighting reference cache `ffb18e03…` is gone and is not on either
  machine (172 `.npz` files hashed).
  Plus the platform itself: the original Fortran was built by an Apple toolchain on
  arm64, so no Linux build is a byte-level reproduction — though the smoke test
  suggests that difference is small where it can be measured at all.
  To move to *reproduction* would need the `ffb18e03…` cache from a backup of the
  macOS machine, both converters recovered rather than reimplemented, and — for the
  Upsilon specifically — a shower seed that was never recorded in the first place.

---

### Session 29 — unified priors built: J/psi, Z, Upsilon (2026-09-06)

Ran `docs/unified_prior_cli_prompt_autonomous.md` Stages 2-6 unattended on the
Ubuntu 24.04 generation VM. Wall time **2 h 10 min** total; Stages 4+5 used
**1.6 h** of a 36 h budget. Zero run failures. Full narrative in
`docs/unified_prior_run_log.md`; this entry records what a future session must
know. Labels per CLAUDE.md section 2.

#### Delivered

* `scripts/prior_build/hepmc_to_prior_hdf5.py` — the HepMC->HDF5 converter that
  §9.1 recorded as never committed. Streams HepMC2 ASCII, supports `born` /
  `bare` / `dressed`, writes `FDL/zData` `[N,8] float32`, `FDL/component_id`
  `[N] int8` and `FDL/weight`. 23 unit tests, all pass (`unittest`,
  `tests/test_hepmc_to_prior_hdf5.py`, synthetic fixture, no sample needed).
* `scripts/prior_build/finalise_prior.py`, `prior_gates.py`,
  `merging_scan_report.py`; `scripts/diagnostics/pairpt_regime_scan.py`;
  `scripts/mg5_cards/unified/` (21 generated cards + `make_cards.py` + `SOURCE.md`).
* Three priors in `data/priors/`, all gate-passing, labelled and **unmixed**:

  | region | file | events | components | TMS |
  |---|---|---|---|---|
  | Z | `z_unified_bare_tms22p8.hdf5` | 697,180 | continuum | 22.8 |
  | J/psi | `jpsi_unified_bare_tms10.hdf5` | 1,736,351 | jpsi 737,262 + continuum 999,089 | 10.0 |
  | Upsilon | `upsilon_unified_bare_tms10.hdf5` | 3,295,815 | 1S/2S/3S/continuum | 10.0 |

  No smearing, no reweighting, no mixing. `versions.txt` has an appended
  `[unified prior production - 2026-09-06, this VM]` section.

#### Four findings that change what §5, §7.9 and §7.2 say

**1. artifact-measured — the 4-D reweighting was far more destructive than
recorded, and it is now quantified exactly.** The 1M reweighted J/psi prior was
sampled with replacement from the 500k file, so the realised weights are
recoverable by matching rows: **1,000,000 of 1,000,000 rows byte-match a 500k
row**, and reweighting the source by the recovered multiplicities reproduces the
1M pair-pT median to 26.0554 vs 26.0554. Consequences: weights 0 to **62**;
**55.05% of source events given weight zero**; **Kish effective sample size
63,037 = 12.61%** of the source; top 5% of events supply 56% of the 1M rows; mean
weight rises by a factor **97** from the 6-8 GeV pair-pT bin to the 30-40 GeV bin.
**Every J/psi result in joint Runs D, E and F rests on a 1,000,000-row file with
the statistical power of ~63,000 independent events, built by a data-derived
deformation of a conditioning variable of the response.**

**2. artifact-measured — §7.9's "the Upsilon prior was hand-smeared" hypothesis is
FALSIFIED, and the 110-192 MeV numbers are standard deviations, not widths.**
Robust widths (IQR/1.349) of the shipped Upsilon signal components are
**0.317 / 0.353 / 0.423 MeV**, the same order as a correct bare-truth sample
converted from recovered HepMC (0.648 MeV) and two orders below a genuine smear
(A/B smeared arm 28.03 MeV, Program B 29.34 MeV). The 110.6/166.7/192.2 MeV
figures reproduce the **std** column exactly; the std of a bare-muon truth line is
dominated by the QED FSR radiative tail, not by resolution. **So the Upsilon
prior is not smeared, the acceptance gate in method book §11 would have passed it,
and the failure of the held-out Upsilon transfer (§1: mass W1 0.28 GeV, per-state
biases -28 to -53 MeV) no longer has the explanation §7.9 offered. That failure is
unexplained again.**

**3. artifact-measured — the truth variant is `bare`, VERIFIED.** The original
J/psi LHE files were reshowered with the recovered cards and seeds (330102 /
340102) and converted under all three variants. Against the shipped 500k prior's
signal component: mass KS **born 62.5 sigma, bare 1.39 sigma, dressed 7.72 sigma**;
the radiative tail below the peak is reproduced bin by bin to ~1% by `bare`, is
**identically zero** for `born`, and is ~30% too small for `dressed`. The
continuum component excludes `born` independently on kinematics (pair-pT KS 33.2
sigma, muon-pT 42.8 sigma). Event bookkeeping agrees with the shipped cutflow to
0.05%. This also validates the new converter: it reproduces a file made by a lost
program on a different OS to sub-sigma on pair pT and muon pT.

**4. artifact-measured — the merging-scale insensitivity scan FAILS in every
region, and catastrophically for the J/psi.** Spread across scan points:

| region | scan points | acceptance spread | pair-pT median spread |
|---|---|---|---|
| J/psi | 5 / 10 / 20 | **48.69%** | **92.85%** (11.1 -> 15.9 -> 28.2 GeV) |
| Z | 5 / 11.4 / 22.8 / 45.6 | 1.47% | 10.25% |
| Upsilon | 5 / 10 / 20 | 1.70% | 14.40% |

The J/psi's fiducial pair-pT floor is 5.14 GeV, so at TMS 10 and 20 the *entire*
fiducial sample sits below the merging scale, where the 1-jet matrix element has
been cut away and the recoil comes from the shower. **For the J/psi the merging
scale does not perturb the pair-pT spectrum, it sets it.** This falsifies method
book §5.1's second hypothesis, which predicted J/psi would be the *least*
sensitive region; it is the most, by a factor of six.

The scan's central assumption was separately **confirmed**: generating the
Upsilon *signal* at 0.5x and 2x TMS gives a response matching the continuum's to
1.3% on the KS and 1.8% on the median shift, so `d(spectrum)/d(TMS)` is common and
the continuum-only scan design is justified.

#### The trap this leaves behind — read before touching TMS

artifact-measured: gate 5 gives J/psi prior pair-pT median **15.87** against CMS
**26.03** (ratio 0.610, KS 374 sigma), measured and **not corrected**. The scan
shows TMS = 20 gives 28.2. **So a TMS between 10 and 20 would reproduce the CMS
pair-pT spectrum almost exactly, and choosing it would be a data-derived tuning of
a generator setting — the same act as the deleted reweighting, but recorded as
"the merging scale" instead of as "a correction" and therefore far harder to
detect later.** TMS is fixed here at the rule value `max(10, M/4)` by the run's
decision rule and was **not** chosen by looking at the data. Other gate-5 ratios:
Z 0.941, Upsilon 1.989 (the Upsilon CMS reference `Ymumu.csv` is a *different
trigger* — muons to 0.713 GeV, pair-pT median 1.45 GeV — and is not comparable to
the DoubleMuParked cache).

#### Two gate definitions that were wrong as written

* **Method book §11 gate 2 is unsatisfiable at float32.** `m^2 = E^2 - p^2` is a
  cancellation, so the `[N,8] float32` schema amplifies ~1e-7 precision on E into
  ~1e-3 on m in the boosted tail. artifact-measured on the new Z prior: stored
  deviation 1.4975e-03 against a float32 round-off floor of **1.4975e-03**, ratio
  **1.000**. Every shipped prior shows it too (J/psi 500k **4.13e-03**), so a
  stored-file gate at 1e-6 would reject every prior the project has ever used. The
  gate is now taken from the converter's float64 check on the HepMC records
  (0.000e+00 for all seven components); the stored value is reported beside its
  floor. No threshold was changed.
* **Gate 1's threshold had to be calibrated, not derived from Gamma.** Set to
  **5.0 MeV** on the robust width: passes the widest correct bare file measured
  (0.648 MeV) by 7.7x, rejects the smallest genuine smear measured (28.03 MeV) by
  5.6x.

#### Also settled

* CKKW-L samples are **not unweighted**: the nominal weight takes 3664 distinct
  values in 17,746 events. The original converter dropped these weights, so every
  shipped prior is the unweighted distribution — an undeclared approximation worth
  **+2.2%** on the Upsilon pair-pT median. The new converter stores them in
  `FDL/weight` and does **not** apply them.
* Method book §6.1's `ptj = TMS` hole was looked for at Stage 1 and **not found**
  (Z control rms pull 0.87, blind-window mean pull +0.48). `ptj = 2.5` is
  insurance, not a repair.
* `Merging:Process = guess` resolves to `Les Houches User Process(es)` code 9999,
  **2 -> 2** for the J/psi signal and **2 -> 1** for the Z continuum.
* The Pythia card MG5 *reads* (`pythia8_card.dat`) is not the one it *writes*
  (`tag_1_pythia8.cmd`); feeding the resolved card back aborts the run. The input
  card carries `Merging:TMS = -1.0`, so `ktdurham` alone sets the merging scale.

#### Blocking debt — these priors are NOT trainable yet

`cms_data.load_theory_prior_z` returns `z_data[:, :8]` and never reads
`FDL/component_id`. The components are shipped labelled and unmixed by design, so
**applying a mixture fraction at load time has to be built before any of these
three files can be trained on.** That code does not exist and was out of scope for
this run. Until it does, `data/priors/*.hdf5` cannot replace the current priors in
any config.

Second item: `data/priors/_compat_probe.hdf5` (sha256 `d44a8528…c4b6`) was written
with h5py 3.16.0 / HDF5 2.0.0 and must be opened once from the Windows `cms`
environment before any Stage 6 file is trusted there. Superblock version is 0 for
both it and the shipped macOS-written priors, so the risk looks low, but that is an
inference from a header byte, not a read test.

### Session 30 — ppzee paired closure: step 3 of section 7 (2026-09-07)

Program C. Executes `docs/step3_ppzee_closure_prompt.md`. Sections below are
written in the order they were produced; the **predictions were recorded
before section 4 was run**, so the outcome cannot be retrofitted.

#### 1. ppzee layout — established numerically, not assumed

artifact-measured 2026-09-07 (`outputs/ppzee_layout.json`,
`outputs/ppzee_pairing.json`) and source-verified from
`dataGenerationCode/ppzeeDataGenCode/FinalData/FinalData_ppzee.py` and
`utilityFunctions/configs.py`.

| item | result |
|---|---|
| `ROL[:, 0:8]` is the detector partner of `FDL` | **yes.** per-column rho 0.9761-0.9906; shuffled control -0.001 to +0.002. Reproduces section 7.4's 0.977-0.991. |
| truth pair pT | **identically zero.** rho(FDL px1, px2) = rho(py1, py2) = -1.0000 exactly; max pair pT = 0. Reco pair pT median 8.82 GeV. |
| mass convention | both numerically massless. \|m\|/E per particle: FDL median 3.6e-06 (max 1.4e-05), ROL median 1.3e-08 (max 3.3e-08), against m_e/E = 8.8e-06 at the median E of 58 GeV. ROL is massless *by construction* (`make4Vector(..., m=0)`). |
| train/test overlap | **none.** 331,699 and 160,000 rows, all distinct within each file, **0** shared rows by blake2b over `FDL` and over `FDL`+`ROL`. |
| `ROL[:, 8:12]` | **the Delphes MET**, as a massless four-vector built from its stored (pT, eta, phi): `E^2 - p^2 = 0` to 1.2e-15 relative, pT median 7.7 GeV, \|eta\| > 3 in 81% of events, present in every event, does not balance the pair pT. source-verified in the generator (`metMsk = a == 99`) and in `utilityFunctions/configs.py`: `'ppzee': {'z_dim': 8, 'x_dim': 12}  # x_dim = 12 includes MET, this is removed in experiments`. Dropped from `x` because `z` carries no MET partner to score it against — identified and excluded, not assumed ignorable. |

**The zero-pair-pT fact must be stated whenever the closure number is quoted.**
The truth is LO 2 -> 1 with no recoil, so every unit of the reco pair pT was
manufactured downstream by the shower and Delphes. That is the configuration
`docs/unified_prior_method.md` section 4 argues is wrong for a response study.
It does not invalidate the per-event measurement; it changes what it means.

**One further fact, not asked for and load-bearing.** artifact-measured: the
upstream results archive `experiments/ppzee/otus_results-dataset=ppzee_test.npz`
IS `data/ppzee_test.hdf5`, row-for-row — `max |z_archive - FDL_test| = 3.0e-05`
and `max |x_archive - ROL_test[:, 0:8]| = 2.9e-05` in stored row order, i.e. a
float32 round-trip and nothing else. So the upstream 3.33 and any number we
measure on this test split are computed on **the same 160,000 events**, and the
comparison is like-for-like rather than merely same-dataset.

#### 2. Upstream reference reproduced

artifact-measured 2026-09-07, `paired_closure.py --results-npz`, all four
numbers as recorded in section 7.4:

| quantity | expected | measured |
|---|---|---|
| detector resolution, rms(mass(x) - mass(z)) | 2.65 GeV | **2.6515** |
| upstream encoder, rms(mass(E(x)) - mass(z)) | 8.83 GeV | **8.8305** |
| `residual_rms_vs_identity` | 3.33 | **3.3304** |
| per-event mass correlation with truth | 0.679 (identity 0.964) | **0.6787** (identity **0.9641**) |

The scorer and the archive are unchanged since Session 26.

#### 3. Predictions, recorded before section 4 was run

hypothesis, 2026-09-07, pre-registered per section 5 of the prompt. Our model's
`residual_rms_vs_identity` on the ppzee test split will be read as one of:

* **near 3.33** — our architecture behaves like the upstream one on the
  upstream benchmark. Confirms section 7.4's reframing: the failure is a
  property of this family of objectives, not of the CMS J/psi setting, and
  ppzee becomes the development bench for M2 and M5.
* **materially worse than 3.33** — our implementation differs from upstream in
  a way that matters, and finding it comes before any new model branch.
* **below 1.0** — nobody in this literature has inverted a response per event.
  Treat it as pairing leakage until disproved: re-check that `pair_index` never
  entered training, that the splits contain no duplicated rows, and that the
  scorer is not being handed the truth it is scoring against. Only after all
  three come back clean is it a result.

hypothesis, additional, same date: **near 3.33 is the expected outcome**, on
the grounds that section 7.7 measured our decoder to be healthy and our encoder
to fail in exactly the way section 7.4 measured the upstream encoder to fail,
and the two encoders are point estimates trained against marginal OT terms.

#### 4. The loader branch (source-verified)

New file `scripts_joint/paired_data.py` plus three additive branches in
`scripts_joint/joint_data.py` (`resolve_joint_config`, `region_data_config`,
`load_joint_regions`) and a `pair_indices` argument on
`build_joint_split_manifest`. `cms_data.py` was **not** touched: its
`_pipeline_semantic_fingerprint` hashes that whole file, so editing it would
have changed every CMS region cache key and every joint contract hash.

**The prime directive, and how it is enforced.** Three mechanisms, in
increasing order of how hard they are to fool:

1. `x` and `z` are drawn through independent permutations of the source file
   (seeds `seed`, `seed+1`, and `seed+2`/`seed+3` for the held-out file),
   exactly as `cms_data.split_unpaired` does for the CMS regions.
2. The loader asserts the six `SPLIT_KEYS` arrays are the only thing returned;
   the row indices are a separate return value that goes into the manifest.
3. A **measured** guard with a positive control: max per-column
   `|rho|(x_train, z_train)` must sit at the finite-sample floor while the same
   statistic on the unshuffled source reads ~0.99. artifact-measured on the run:
   **0.0060 against a source of 0.9907**.

One defect found and fixed during test: a flat 0.05 threshold on that
correlation is wrong at small n, because two genuinely *independent* draws of
100 rows correlate at ~0.26 by chance. `leakage_threshold(n)` is now
`max(0.05, 6/sqrt(n-3))`, which keeps the power where it matters (a leak reads
~0.98, i.e. 9.7 sigma at n=100 and 340 sigma at n=120,000).

**Evidence the CMS muon path is unchanged.** Three independent checks:

* the full suite went 203 -> 228 tests (the 25 new ones are all in
  `tests/test_paired_data.py`), still `OK (skipped=2)`; a name-by-name diff of
  the verbose output shows **no pre-existing test changed status**;
* rebuilding **Run E's** split manifest with the new code reproduces its
  recorded `contract_sha256` **89b6130547eaa9...d918a** exactly, and the identity
  document compares equal;
* `pair_indices` is `{}` for a CMS run and the string `pair_index` does not
  appear anywhere in a CMS manifest, so the two `purpose`/`locked_test_policy`
  strings and the region blocks are byte-identical to before.

`configs_joint/cms_Joint_ppzee.yaml` is a single-region **bench, not a result**.
It is deliberately not `extends: cms_Joint_runE.yaml` - `cms_data._deep_merge`
has no key-deletion operator, so an inheriting config cannot drop the inherited
jpsi and z regions; every model, loss, loader, selection and evaluation value is
instead a verbatim copy of the resolved Run E config, with four stated
differences (one paired region, electron daughter masses, a bounded 60+60 epoch
schedule, and rescaled gates).

**A pre-existing defect found by attribution, not fixed here.** `--run ppzee
--smoke` dies with `RuntimeError: Stage ... produced no validation checkpoint`.
artifact-measured: `--run abNarrowSplit --smoke` fails identically, so this is
**not** the paired branch. Mechanism: `--smoke` leaves ~100 validation events,
`identity_reference._floor_pair` then puts the floor at or above the identity,
`gauge()` returns `+inf` by design, `score_joint_metrics` maps that to `inf`,
and `joint_trainer.py`'s `if score < stage_best_score` never fires when both
are `inf`. Any config gating a `*_vs_identity` metric is affected. Workaround:
`--smoke --num-samples 20000`, which completes. Both smoke runs are reported
below as section 4 of the prompt asks.

| command | result |
|---|---|
| `--run ppzee --device cuda --dry-run` | passes; splits 800/100/1000, leakage guard 0.0964 <= 0.2125 |
| `--run ppzee --device cuda --smoke` | **fails**, pre-existing gauge/floor defect above |
| `--run ppzee --device cuda --smoke --num-samples 20000` | passes end to end |

#### 5. How far below 1.0 could ANY map get? The oracle bound

artifact-measured 2026-09-07, `outputs/ppzee_oracle_bounds.json` and
`outputs/ppzee_oracle_bounds_nonparametric.json`. These are **oracles**: fitted
on `ppzee.hdf5` *using the pairing*, evaluated out of sample on
`ppzee_test.hdf5`. No honest encoder can beat them.

| map | rms [GeV] | ratio to identity |
|---|---|---|
| identity, hand back x | 2.6515 | 1.0000 |
| oracle: best constant mass offset (+0.687 GeV) | 2.5578 | **0.9647** |
| oracle: best affine in mass(x) | 2.4762 | 0.9339 |
| oracle: ridge on 14 x-kinematic features | 2.4393 | 0.9200 |
| oracle: non-parametric correction from mass(x), 200 quantile bins | 2.1474 | **0.8099** |
| oracle: same, with abs(eta1), abs(eta2) added (6,400 cells) | 2.3484 | 0.8857 |

**This is the reference the prompt's section 5 was missing, and it changes the
reading of the whole test.** The reachable range is **[0.81, 1.0]**, not
[0, 1.0]: only about 19% of the ppzee detector resolution is predictable from
`x` at all, the rest being irreducible stochastic smearing. So "below 1.0" is
a far weaker statement than it sounds, and the genuinely impossible regime - the
one that would be decisive evidence of leakage - is below about 0.8.

Two construction notes, because the first attempt was wrong. Binning `mass(x)`
into equal-count bins and predicting the bin **mean** of `mass(z)` scores 1.27
to 2.25, i.e. *worse* than the identity. That is not a bug in the data: with a
heavy tail the top equal-count bin spans 114 to 827 GeV, `mass(x)` is nowhere
near constant inside it, and the conditional-mean inequality does not apply to
a coarsening. Predicting the **correction** `E[mass(z) - mass(x) | cell]` on top
of the identity is the right construction and is what the table reports. Adding
more conditioning variables past `mass(x)` makes the out-of-sample number worse,
which is cell sparsity, not information - so 0.81 is close to the practical
limit for a mass-only map.

#### 6. THE RESULT

`outputs/cms_Joint/ppzee/` - 120 epochs (60 deterministic + 60 with the encoder
noise ramped), 120,000 training events, 7.7 + ~8 min wall on the RTX 4080
Laptop at ~8 s/epoch, scored on all **160,000** rows of `ppzee_test.hdf5`,
which is the same event set the upstream 3.33 was measured on.

artifact-measured, `outputs/cms_Joint/ppzee/paired_closure.json`:

| | `residual_rms_vs_identity` |
|---|---|
| oracle floor (best correction fitted ON the pairing) | 0.8099 |
| **our model, stage-2 checkpoint (selected)** | **0.9843** |
| our model, stage-1 checkpoint | 0.9808 |
| identity, hand back the detector event | 1.0000 |
| **upstream OTUS encoder, same 160,000 events** | **3.3304** |

**Our encoder is 3.4x closer to the truth partner than the upstream one.** The
pre-registered "near 3.33" prediction is **falsified**, and so is the reasoning
behind it.

**But the decomposition says it did not invert the response.** artifact-measured,
`outputs/cms_Joint/ppzee/paired_closure_decomposition.json`, on the mass
residual `mass(pred) - mass(z_true)`:

| | mean [GeV] | std [GeV] | rms [GeV] | ratio |
|---|---|---|---|---|
| identity | -0.6988 | 2.5578 | 2.6515 | 1.0000 |
| model | **-0.1036** | **2.6078** | 2.6099 | 0.9843 |

The encoder removed **85.2%** of the detector's global mass bias and made the
per-event **scatter 2.0% worse**. Removing the bias alone, changing nothing
else, would score 0.9647 - *better* than the model. So the entire per-event gain
is a global constant, and on the resolution itself, which is the thing a
per-event inverse has to invert, the encoder is slightly negative. It captures
**8.3%** of the available headroom, against **18.6%** for subtracting one number.

It is not the identity in disguise: `displacement_vs_resolution` = 0.416, i.e.
it moves events by 0.42 detector resolutions. It moves them, mostly sideways.
Per four-vector column against the identity: transverse (px, py) **0.33**, a
real improvement; longitudinal (pz, E) **1.19 / 1.14**, worse.

**Marginals from the same run** (`joint_evaluation.json`), which tell the other
half of the story:

| direction | mass KS | gauge | pair-pT KS | gauge | C2ST (linear) |
|---|---|---|---|---|---|
| z -> x simulate | 0.0154 | **0.058** | 0.0068 | **0.000** | **0.510** |
| x -> z unfold | 0.0989 | 0.588 | 1.0000 | 1.000 | **1.000** |

The decoder is the strongest result in the joint programme to date: it
manufactures the entire recoil spectrum from a truth with *identically zero*
pair pT (pair-pT gauge 0.000, C2ST 0.510 = indistinguishable). The encoder's
x -> z C2ST of exactly 1.0 is **not** a model failure and must not be quoted as
one: the truth pair pT is a delta at zero, so any encoder output with non-zero
pair pT is separated by a linear classifier on one coordinate. That is the
dataset, not the map. `cms_Joint_ppzee.yaml` therefore excludes
`latent_pair_pt_ks` from its gates, and the reason was written into the config
before the run.

Zero gate-passing validations out of 24; `global_gate_fallback.txt` records that
`best_model.pt` is not an accepted result. The blocking gate is
`latent_mass_ks_vs_identity` at 0.588 against a 0.5 threshold.

#### 7. Which of the three pre-registered readings holds

The **third** fired on its face - 0.9843 is below 1.0 - so the leakage protocol
was run in full. artifact-measured, `outputs/cms_Joint/ppzee/paired_leakage_audit.json`,
**all four checks pass**:

| check | result |
|---|---|
| 0. the pair index describes the arrays the trainer received | PASS - all six splits reproduce the manifest's own sha256 fingerprints, which were written from the arrays the trainer was handed |
| 1. the pairing never entered training | PASS - max abs(rho)(x_train, z_train) 0.0060 / 0.0136 / 0.0033 for train/val/test, against a source positive control of 0.9907 |
| 2. no duplicated rows | PASS - 0 shared rows for every pair of splits, on x and on z |
| 3. the scorer was not handed the truth | PASS - rms(pred - truth) 8.79, and the scored x is bit-identical to the split x |

**But the third reading's premise is wrong, and the oracle table above is why.**
It was written as "nobody in this literature has inverted a response per event",
which treats any number below 1.0 as extraordinary. On this dataset a 1.6%
improvement is worth 8.3% of a headroom that only extends to 0.81, and it is
fully accounted for by removing a global mass bias. Nothing extraordinary
happened. The honest summary is a **fourth** reading the prompt did not list:

> **materially better than 3.33 and still not an inverse.** Our encoder beats
> the upstream one on this benchmark by 3.4x, and the reason is architectural,
> not objective: ours is a residual flow initialised near the identity with
> bounded `mean_residual_limits`, so it structurally cannot wander 3.3
> resolutions away from its input, while the upstream encoder is an
> unconstrained MLP. Section 7.4's conclusion that the per-event failure "is a
> property of this family of objectives" is therefore **too strong** - the
> upstream 3.33 is substantially a property of its architecture. What survives,
> and is now measured on the paper's own benchmark with the paper's own truth
> partners, is the weaker and more useful claim: **an OT objective on marginals
> buys a global scale correction and does not buy a per-event inverse.** That
> is exactly what section 7.6 argued from the CMS side, now confirmed where
> paired truth exists.

Also worth recording: the deterministic stage-1 checkpoint (0.9808) beats the
stochastic stage-2 one (0.9843) per event while losing on the marginal
(mass KS 0.102 vs 0.099, W1 0.799 vs 0.730). The same trade-off as sections 7.3
and 7.5, now visible per event.

#### 8. Recommendation: does ppzee become the bench for M2 and M5?

**Yes for M5, yes for M2 with one restriction.** The case for it is confirmed
by measurement: 120 epochs on 120,000 events took **under 16 minutes** on this
host and scoring takes about two more, against 16-24 hours for a CMS run that
can only produce a marginal answer. It carries a published number, its truth
partners make the per-event test possible at all, and this session's oracle
bound now gives it a *floor* as well as a ceiling, so a method change can be
placed on a scale rather than merely ranked. The decoder direction is fully
usable as-is and already scores 0.058 / 0.000 / 0.510, so **M5 (decoder as
forward simulator) should be developed here**.

The restriction is on M2. This truth is LO 2 -> 1 with identically zero pair pT,
so the x -> z marginal is unmatchable by construction: latent pair-pT KS is
pinned at 1.000 and the x -> z C2ST at exactly 1.000 whatever the encoder does.
An amortised posterior is precisely the thing whose latent marginal one wants to
read, so **M2 must be scored on ppzee by `residual_rms_vs_identity`, the mass
gauge, and posterior calibration (pull and coverage, which `paired_closure.py`
already computes from `z_pred_draws`) - never by the latent pair-pT or the
x -> z C2ST.** With that restriction the discriminating range is 0.81 to 1.0,
which is narrow but real: our point estimate sits at 8.3% of it and a working
posterior should move measurably.

Two caveats to carry forward. The headroom is small because ppzee's response is
mostly irreducible smearing, so a method that helps a lot on CMS J/psi could
show only a few percent here. And **`data/ppttbar*.hdf5` is not the free second
benchmark section 7 of the prompt hoped for**: artifact-measured, it is
`[N, 24]` - six four-vectors - while `cms_data._validate_cached_arrays`,
`cylindrical_physics_features`, `invariant_mass_np` and the model's six
cylindrical coordinates are all `[N, 8]` two-body. The paired loader's
`_column_slice` refuses anything but exactly 8 columns rather than silently
mis-slicing it. Turning one number into two would mean a different model
dimensionality throughout, not a config change.

#### 9. Artifacts

New, all this session: `scripts_joint/paired_data.py`,
`scripts_joint/score_paired_closure.py`, `scripts_joint/paired_leakage_audit.py`,
`tests/test_paired_data.py` (25 tests), `configs_joint/cms_Joint_ppzee.yaml`,
`outputs/ppzee_pairing.json`, `outputs/ppzee_layout.json`,
`outputs/ppzee_oracle_bounds.json`,
`outputs/ppzee_oracle_bounds_nonparametric.json`, and the run directory
`outputs/cms_Joint/ppzee/` with `paired_closure.json`,
`paired_closure_decomposition.json`, `paired_leakage_audit.json` and the two
per-stage closure files. `outputs/cms_Joint/ppzee_smoke*/` are throwaway smoke
directories and can be deleted.

**One deviation from the prompt, stated.** It asked for `pair_index` "per split
in the split manifest". That is what was done, literally - the full integer
lists are in `joint_split_manifest.json` - but it makes that file **13 MB** for
this run. If a future paired run is larger, move the arrays to a sidecar npz
and keep the per-split count, seed and sha256 in the manifest; the digest check
in `paired_data.materialize_pairs` already exists to make that safe.

### Session 31 — unifiedP1 new Upsilon prior transfer (2026-09-07)

User requested repository status and testing the new Y/Upsilon prior with
joint unifiedP1. artifact-measured: the repository started clean at 64c80cf,
one commit ahead of the locally recorded origin/main; that remote-tracking
reference became aligned during the session. No commit or push was performed
by this evaluation. Concurrent changes to existing J/psi/Z plots were observed
and left alone.

artifact-measured: decoded the full 1,000,000-event legacy prior and the new
17,802-event evaluation prior using identical frozen best_model.pt (epoch 432,
stage 4, noise 1/1). Both CUDA transfer pipelines completed, including peak
plots, state response, distribution metrics and C2ST. Source priors and model
checkpoints were read only. Outputs are under unifiedP1/upsilon_transfer_legacy,
upsilon_transfer_unified, and upsilon_transfer_comparison.

artifact-measured: the new input funnel was 3,295,815 generated -> 2,016,181
selected -> 261,026 uncapped accept-reject -> 17,802 after matching the old
prior's total signal fraction 0.1905. ESS/N=0.73354; retained state counts
1S/2S/3S/continuum=677/1053/1661/14411. The 50,000-event default minimum was
explicitly lowered to 10,000; weights were not clipped and no rows duplicated.
Inter-state ratios remain those of the new generator, not the legacy mixture.

artifact-measured: shared 8,300-event inclusive comparisons give new mass W1
identity/model/floor=0.18097/0.28268/0.01295 GeV, gauge 1.605; old
0.23668/0.35483/0.01295, gauge 1.528. New mass KS
identity/model/floor=0.12012/0.15976/0.01354, gauge 1.372. New pair-pT KS
identity/model/floor=0.32554/0.33807/0.01423, gauge 1.040. The new prior is
closer before and after decoding, but decoding degrades both priors.
The new prior's mass W1 gauge by own-distribution pair pT is 1.543 (0–5),
0.775 (5–10), 0.707 (10–20), 0.881 (20–40). Above 40 GeV only 31 matched
events are available, so no metric is scored. These are conditional marginal
comparisons, not paired response closure or statistically established gains.

artifact-measured: the saved checkpoint has global_gate_fallback=true,
hard_gates=true, gate_fail_penalty=1e6 and score 1000002.8452, inconsistent
with the current nonblocking 35/65 configuration. source-verified: current
region_data_config does not forward prior_components and load_theory_prior_z
reads zData only; no certification that the requested training-time generator
weights/mixtures were implemented is possible from this evaluation. The
frozen run was evaluated as it exists, without retraining or checkpoint tuning.

source-verified changes: materialize_eval_prior preserves top-level component
mapping, serializes nullable metadata, keeps retained weights aligned after
mixture selection and marks weighted outputs not decode-ready;
upsilon_transfer_test accepts --output-dir; compare_transfer_priors reports
old/new identity and repeated disjoint-CMS floors in fixed pair-pT bins.
artifact-measured validation: 16 materializer tests and 3 comparison tests
passed, both full transfer pipelines succeeded, and the comparison plot was
visually checked. See the retained REPORT.md and transfer_comparison.json for
methods, hashes, limitations and reproduction commands.

### Session 32 — joint Run E with legacy Upsilon prior (2026-09-07)

User requested a test of joint Run E with the legacy Y prior. artifact-measured:
a fresh CUDA decode of all 1,000,000 legacy events with best_model.pt (selected
epoch 65, deterministic stage 1, noise 0/0) and full transfer evaluation
completed successfully. No training, tuning, source changes or checkpoint
modifications. Fresh artifacts are in
outputs/cms_Joint/Run_E/upsilon_transfer_legacy_20260907/.

artifact-measured: matched 8,300-event comparisons give mass W1
identity/model/floor=0.23668/0.27764/0.01295 GeV, gauge 1.183; mass KS
0.16530/0.18386/0.01354, gauge 1.122. Pair-pT KS
0.33217/0.22000/0.01423, gauge 0.647; pair-pT W1
1.36373/1.24146/0.11627 GeV, gauge 0.902. Run E improves pair momentum but
worsens mass. It scores better than unifiedP1 on this legacy prior (mass W1
0.35483, gauge 1.528 in Session 31), but the selected deterministic epoch-65
checkpoint and stochastic fallback epoch 432 are not a controlled comparison.

artifact-measured: prior/seed/checkpoint hashes match the historic retained
Run E transfer; inputs are bit-identical, outputs differ by at most 3.052e-5
GeV per four-vector element. Standard 16,600-event evaluation mass KS
0.1916265 is identical, and W1 0.281751353 differs by approximately 1.1e-10
GeV. Historical findings are reproduced to numerical precision. The 16,600
sample must not be paired with the 8,300-event reference table's floor.
MLP C2ST AUC is 0.81455. Mean decoded-minus-input state mass shifts are
-52.76/-55.58/-57.70 MeV for 1S/2S/3S; these are not CMS-fit peak biases.

artifact-measured: differential mass W1 gauges are 1.034/6.826/1.121/1.102
in own-distribution pair-pT bins 0–5/5–10/10–20/20–40 GeV. The 5–10 GeV
reference has little headroom (identity W1 0.05976, floor 0.05297); no
significance is claimed. The 40+ bin has only 31 matched events and is not
scored. Identity references, fixed-bin results and provenance checks are in
identity_references.json. REPORT.md records full methods and limitations;
the peak plot was visually inspected. This is post-unblinding held-out transfer.

### Session 33 — joint Run E last_model.pt on legacy Upsilon (2026-09-07)

User requested the same legacy Y test with last_model.pt. artifact-measured:
decoded all 1,000,000 events on CUDA with epoch 432, stage 4, noise 1/1;
full evaluation succeeded. No source or checkpoint changes, retraining,
Upsilon tuning or checkpoint selection. Fresh artifacts are in
outputs/cms_Joint/Run_E/upsilon_transfer_legacy_last_model_20260907/.

artifact-measured: matched 8,300-event mass W1 identity/model/floor is
0.23668/0.09698/0.01295 GeV, gauge 0.376; mass KS
0.16530/0.06614/0.01354, gauge 0.347. Best_model from Session 32 gave
mass W1 0.27764, gauge 1.183. However, pair-pT W1 for last_model is
identity/model/floor 1.36373/1.71378/0.11627 GeV, gauge 1.281, and pair-pT
KS 0.33217/0.30120/0.01423, gauge 0.903. Both momentum metrics are worse
than best_model, whose gauges were 0.902 and 0.647 respectively.

artifact-measured: mass-window retention falls from best_model's 98.3644%
to 87.8701%. The inclusive metrics condition on survival and do not penalize
that loss. Mean labelled 1S/2S/3S decoded-minus-input mass shifts are
+184.9/+199.6/+205.8 MeV; final state distribution standard deviations are
480/512/537 MeV. The overlay visibly washes out the peak structure. These
are full labelled-state moments, not CMS-fitted Gaussian peak parameters.
The mass-score improvement is not a successful detector-response claim.

artifact-measured: mass W1 gauges by own-distribution pair-pT bins
0–5/5–10/10–20/20–40 GeV are 0.257/13.177/4.660/2.157. Low-pT events
carry the inclusive improvement; higher-pT bins degrade. The 5–10 identity
is close to its floor, so that gauge is unstable. Above 40 GeV only 31
matched events exist and no score is reported. No significance is claimed.

artifact-measured: standard 16,600-event evaluation gives mass W1 0.100575,
KS 0.070241 and MLP C2ST AUC 0.78698. These do not use the 8,300-event
reference table's sample. The fresh decoded arrays reproduce the existing
upsilon_transfer_last_model arrays bit-for-bit with matching checkpoint,
prior and seed. All output events are finite. REPORT.md,
identity_references.json and best_vs_last_mass.png retain the methods,
references, hashes and comparison. The plot was visually inspected.
