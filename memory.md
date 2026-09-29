# Persistent Memory — OTUS on CMS Open Data

*Condensed rewrite 2026-09-08, post-Run E. The full pre-Run E record lives in git
history (commit bf9a59f and earlier) and in `memory.md.bak.20260904`. This file now
starts from joint Run E and keeps only what is load-bearing for the next experiments.*

## 0. How to use this file

- `CLAUDE.md` sections 2-6 are still authoritative for labels, guardrails, naming, commands, and environments. Do not repeat them here.
- Label every number in this file: `source-verified`, `artifact-measured`, `reported-not-reproduced`, `hypothesis`, `proposal`.
- Update this file last. Add a short entry to section 6 when a session produces a durable finding.

## 1. Status at a glance (2026-09-08)

**Best accepted result: joint Run E** (`outputs/cms_Joint/Run_E/`), one shared mass-blind response model closing J/psi and Z simultaneously.

artifact-measured, selected checkpoint = stage-1 deterministic (`global_epoch` 65, noise 0/0):

| region | direction | mass W1 [GeV] | mass KS | pair-pT KS | C2ST (MLP) |
|---|---|---|---|---|---|
| J/psi | z->x simulate | 0.00088 | 0.017 | 0.052 | 0.62 |
| J/psi | x->z unfold | 0.00177 | 0.037 | 0.022 | 0.65 |
| Z | z->x simulate | 0.086 | 0.013 | 0.006 | 0.51 |
| Z | x->z unfold | 0.053 | 0.007 | 0.005 | 0.51 |

Two caveats:
1. The selected checkpoint is deterministic; stochastic stages have never won selection.
2. The J/psi prior was hand-smeared to ~26 MeV against ~28 MeV data, so the encoder only had to remove a few % of the mass width. Honest-prior unfolding is not demonstrated.

**Nothing is training.** The working tree contains an in-flight selection/trainer refactor (section 1.1). Tests: `artifact-measured` 233 OK, 2 skipped on 2026-09-08.

**Upsilon transfer is the honest weak point.** artifact-measured, frozen Run E, legacy Upsilon prior:

| checkpoint | mass W1 identity/model/floor [GeV] | mass KS identity/model/floor | pair-pT KS | verdict |
|---|---|---|---|---|
| best_model (ep 65) | 0.23668 / 0.27764 / 0.01295 | 0.16530 / 0.18386 / 0.01354 | 0.33217 / 0.22000 / 0.01423 | mass worsens, momentum improves |
| last_model (ep 432) | 0.23668 / 0.09698 / 0.01295 | 0.16530 / 0.06614 / 0.01354 | 0.33217 / 0.30120 / 0.01423 | mass improves, peaks washed out |

best_model per-state mean mass shifts: -52.76 / -55.58 / -57.70 MeV for 1S/2S/3S. last_model per-state mean shifts: +184.9 / +199.6 / +205.8 MeV with decoded state stds 480/512/537 MeV and mass-window retention 87.87% (best_model 98.36%).

**ppzee paired closure (Session 30) is the clearest per-event result.** artifact-measured on 160,000 held-out pairs:

| | residual_rms_vs_identity |
|---|---|
| oracle floor (best correction fitted on the pairing) | 0.8099 |
| our encoder, stage-2 checkpoint | 0.9843 |
| our encoder, stage-1 checkpoint | 0.9808 |
| identity (hand back detector event) | 1.0000 |
| upstream OTUS encoder, same events | 3.3304 |

Read: our encoder beats upstream by 3.4x but still does not invert the response. It removes 85% of the global mass bias and makes per-event scatter 2% worse (mean residual -0.104 GeV vs identity -0.699 GeV; std 2.608 vs 2.558 GeV). The decoder direction is the strongest result: pair-pT gauge 0.000, mass gauge 0.058, C2ST 0.510. ppzee is the recommended development bench for M2/M5.

### 1.1 In-flight working-tree changes (commit before any new run)

- `joint_trainer.py` / `joint_train_utils.py`: gates and 1e6 penalty removed; `eval_loss` logged next to `train_loss`; first validation checkpoint always saved (smoke no longer dies on +inf gauge).
- `joint_metrics.py`: selection score = normalized target score, no pass/fail gates.
- `dashboard.py`: plots eval loss; gate panel removed.
- `configs_joint/*`: hard gates/penalties removed from Run A chain, abNarrowSplit, unifiedP1, ppzee.
- `_retired_launchers/` deleted.
- `scripts/prior_build/materialize_eval_prior.py`: full component-mixture materialization added.
- `scaffold/`: untracked notebooks (loss comparison, mass ratio best vs last, CMS peaks).

### 1.2 Update 2026-09-09/10 — noise-control runs finished, meeting decisions

- **artifact-measured:** `Run_E_noiseLowCore` and `Run_E_noiseLowBoth` completed
  the 360-epoch stage-2→4 reruns (warm-started from Run E stage 1). Stage 4 is
  now the best checkpoint in both: selection score 0.588 (LowCore) / 0.575
  (LowBoth), against 0.663 at stage 2 and ~0.78 at stage 3. `eval_loss` falls
  0.943 → 0.420 (stage 3) and 0.394 → 0.360 (stage 4) while the train loss
  stays flat. `Run_E_noiseLowTail` has a config but no `history.json`.
- **proposal / decision, 2026-09-09:** keep the stage-1 deterministic warmup,
  keep the encoder and decoder on one shared schedule, and test one constant
  schedule in stages 2-4. New config `configs_joint/cms_Joint_runE_constNoise.yaml`
  (dry-run only; not trained). The retired `Run_E_encoderDet*` split-schedule
  arm and its two tests are stale as of this decision.
- This subsection did not previously exist; `memory.md` had stopped before the
  noise-control runs.

### 1.3 Update 2026-09-11 - joint Run H trained; the stage-3 tail breaks the Upsilon peaks

- **artifact-measured:** `outputs/cms_Joint/Run_H/` finished the 20/80/80
  schedule (global_epoch 180; `last_model.pt` written 2026-09-11 04:53). The
  stage schedule is verified from all 180 `history.json` rows: stage 1 `0/0`,
  stage 2 `1.0/0.25`, stage 3 `1.0/0.5`, encoder and decoder shared every epoch.
- **artifact-measured:** the in-domain selection score degrades through the
  stochastic stages: stage-1 best 0.687 (ep 20) < stage-2 best 0.839 (ep 90) <
  stage-3 best 2.017 (ep 115); `last_model` ep 180 scores 2.664. `best_model.pt`
  is the stage-1 deterministic warmup.
- **artifact-measured (post-unblinding held-out transfer, fixed 400k Upsilon
  subset, `outputs/cms_Joint/Run_H/tail_audit/`):** decoded Upsilon medians
  (median - CMS fit, MeV) are -14/-25/-19 for `best_RunH_stage2` (ep 90),
  -18/-29/-24 for `best_RunH_stage3` (ep 115), and **-125/-136/-136 for
  `last_model` (ep 180)**. The break happens between ep 115 and ep 180.
- **artifact-measured:** the shift is in the deterministic map, not the sampled
  noise. Decoding `last_model` at inference multipliers (core, tail) =
  (0,0)/(1,0)/(1,0.25)/(1,0.5)/(1,1) leaves the 1S median at -125 MeV; forcing
  tail 0.5 onto the healthy ep-90/100/115 checkpoints leaves their peaks
  unchanged. It is not noise overflow.
- **artifact-measured:** the model switched the noise off. Learned `tail_sigma`
  fell 8.1e-4 (init) -> ~1.8e-4 and keeps drifting down; `core_sigma` sits on
  its 1e-3 floor. The native decoder injects ~10 MeV (core) + ~0.8 MeV (tail) of
  Upsilon mass noise against the 84 MeV CMS resolution. The decoded 1S std is
  0.2823 GeV at noise 0 and 0.2827 GeV at native noise, so the width is carried
  by the mean map, not the noise (mechanism hypothesis in the diagnosis doc).
- **proposal / decision:** use `best_RunH_stage2_stochastic_core.pt` (ep 90)
  for any Upsilon transfer until the tail stage is fixed; do not use
  `last_model.pt`. Full audit and fix options:
  `outputs/cms_Joint/Run_H/tail_audit/DIAGNOSIS_AND_FIX.md`.

### 1.4 Update 2026-09-11 - drift control implemented (Run H anchor / fix arms)

- **source-verified:** the 2026-09-11 audit fix is implemented. New
  `scripts_joint/joint_anchor.py` (`MeanMapAnchor`) freezes the deterministic
  encoder/decoder map captured at the end of the declared noiseless reference
  stage, on a fixed evenly spaced reference batch, and penalises a
  per-coordinate variance-normalised MSE in later stages.
  `run_joint.assert_mean_map_anchor_contract` is the guard, called in `main()`
  beside the invariant-mass guards. It requires the top-level declaration, a
  valid enabled `reference_stage`, at least two reference events, and **rejects
  any anchored stage that switches the encoder or decoder noise off** (the
  model must stay stochastic in both directions). Tests:
  `tests/test_joint_anchor.py` (5); full suite 255 tests, OK (4 skipped).
- **source-verified:** `joint_trainer.py` now carries the AdamW moments across
  stages (`carry_optimizer_state: true`; only the moments are loaded and the
  fresh stage hyperparameters are re-applied, so the LR/weight-decay schedule is
  not overwritten) and supports a per-step weight EMA
  (`loaders.ema_decay`, default 0 = off). When EMA is on, each validation scores
  the live weights and the EMA and checkpoints the better one; the live weights
  are always restored so training continues from the un-averaged iterate.
- **artifact-measured:** dry-run and smoke passed for
  `configs_joint/cms_Joint_runH_anchor.yaml` (anchor only, the single-variable
  attribution arm) and `configs_joint/cms_Joint_runH_fix.yaml` (anchor +
  cosine LR decay to 5% + `weight_decay` 1e-4 + EMA 0.999 + stage-3 tail ramp
  0.25 -> 0.5 + optimizer carry). 830,536 parameters, 180 epochs; the smoke log
  shows the reference captured, the anchor term in the training loss, the
  optimizer state carried and the EMA armed. **No full training started.**
  Encoder and decoder noise are kept in both arms, as instructed.
- **proposal:** the two knobs to sweep are the anchor weight (0.5 in both arms)
  and the `reference_stage`. Anchoring to stage 1 (as specified) also resists
  the legitimate ~+30 MeV stage-2 calibration; anchoring stage 3 to the stage-2
  best map would preserve it. Both are config-only.

### 1.5 Update 2026-09-11 - Run H fix trained: drift slowed, asymmetry NOT solved

- **artifact-measured:** `Run_H_fix` completed (180 ep). The in-domain stage-3
  collapse is mostly tamed: selection score mean 1.318 (best 0.803 at ep101,
  last 1.904) versus Run H's 2.585 (best 2.017, last 2.664); eval_loss mean
  0.600 versus 0.726. Stage 3 still degrades monotonically, just ~3x slower
  (0.803 -> 1.904 across its 17 draws).
- **artifact-measured (Upsilon transfer, `upsilon_transfer_last_model`):** the
  *drift* is fixed. `last_model` medians are +11/-3/+2 MeV versus the CMS fit
  (Run H: -125/-136/-136); inclusive mass KS 0.0376 (Run H: 0.0724).
- **artifact-measured:** the *peak shape* is not fixed, and the anchor moved the
  problem rather than removing it. `last_model` 1S std 0.330 GeV with skew
  -1.54 (q16-median -507 MeV, q025-median -989 MeV), i.e. a heavy low-mass
  shoulder. The stable checkpoints are the opposite: `best_RunHfix_stage2` /
  `best_RunHfix_stage3` are narrow (1S std 0.125, close to the ~0.138 physical
  expectation) but sit -66/-81/-80 MeV low. The anchor pinned the biased
  stage-1 map (-50/-64/-62 MeV), so stage 2 could no longer perform the
  calibration Run H's *unanchored* stage-2 did (-14/-26/-19 MeV). The anchor
  traded "centred but over-spread" for "narrow but biased", and the late
  over-spread relapse still appears at ep180.
- **artifact-measured (slice diagnostic, noise 0):** the conditional asymmetry
  is not solved. Z is fixed (per-slice decode W1/floor 1.1-2.7 versus Run F
  4.2-10.3). J/psi degrades through the stochastic stages:
  stage 1 11.0/1.25/3.60/3.39 -> stage-2 best 14.9/1.99/8.29/2.51 ->
  stage-3 best 19.9/6.20/13.6/8.33 -> last 33.8/12.7/34.3/17.3
  (Run F best_model 17.0/5.64/14.6/3.51).
- **hypothesis:** the decoder still does not carry the resolution in the noise
  (core sigma ~0.0012 -> ~11 MeV against the 84 MeV CMS resolution), so the
  marginal loss keeps the centre/width split unidentifiable: the anchor pins the
  mean map, the model pays the penalty late, and the width is either missing
  (best checkpoints) or carried by the mean map as a heavy low-mass shoulder
  (last checkpoint). The J/psi conditional degradation is the M1
  stochastic-encoder failure, which the "keep the encoder noise" decision keeps
  in play. Both need a noise-identifiability fix, not a stronger anchor.

## 2. Established findings that still matter

1. **The J/psi success depended on the pre-smeared prior.** artifact-measured A/B: smeared arm 34/35 gate passes, narrow arm 0/35; only the prior file differed. The identity map passes every strict J/psi latent target on the smeared prior.
2. **M1 is closed.** artifact-measured `AB_narrow_split`: splitting noise schedules did not fix the encoder. Per-validation J/psi `latent_mass_ks_vs_identity` minimum 0.972, never below the 0.898 benchmark. The decoder stayed healthy (direct gauge 0.098).
3. **A stochastic encoder cannot hit a delta-like prior.** source-verified: encoder noise floor is 0.001 in ln pT -> ~2.19 MeV on J/psi mass; the honest J/psi core is ~0.16 MeV. No amount of training closes a factor of 14. Proposal: encoder must stay deterministic for delta-like regions.
4. **The encoder failure is structural, not a schedule issue.** A point-estimate encoder emits one smooth hump; the honest prior is a narrow core + continuum pedestal. It matches width but not shape. This is why the next real branch is M2 (amortized posterior) or M5 (decoder as forward simulator).
5. **The decoder is the programme's strongest asset.** Narrow-prior direct gauge 0.098; ppzee decoder manufactures the entire recoil spectrum from zero-pair-pT truth (pair-pT gauge 0.000, C2ST 0.510).
6. **An OT objective on marginals buys a global scale correction, not a per-event inverse.** Measured on ppzee with paired truth; all four leakage checks pass.
7. **The Upsilon bias is a constant fractional mass scale, not additive resolution.** artifact-measured: -52.76/-55.58/-57.70 MeV on prior means 9.43169/9.98109/10.30649 GeV = -0.5594%/-0.5568%/-0.5599%. Trained regions: J/psi direct mass shift -0.011%, Z -0.023%.
8. **The objective has no absolute mass-scale anchor.** source-verified: joint config has `resonance_mass_w1 = 0` (physical GeV, windowed). `pair_mass_w1 = 1.0` acts on standardized mass, so it fixes shape, not absolute scale.
9. **Stage rewinds + fresh optimizer + noise jump explain the loss spike / plateau.** artifact-measured: stage-2 best is its first epoch (ep 73, score 0.666); stage 3 starts from ep-73 weights with core noise 0.10 -> 1.00, train loss 0.7447 -> 2.1789. Stage bests: ep65 0.612, ep73 0.666, ep332 1.618, ep421 1.194. No stage ever beats stage 1.
10. **Standardized mass terms are very asymmetric across regions.** source-verified: with J/psi x-space std 28.1 MeV and Z x-space std 5.449 GeV, `pair_mass_w1 = 1.0` is worth 35.6 loss units per GeV of J/psi mass error and 0.18 per GeV of Z mass error.
11. **Full-pass batch cardinality is very unequal.** artifact-measured: per update, J/psi x 16,858 / J/psi z 417 / Z x 24,463 / Z z 4,651. Implemented correctly but a variance concern for the J/psi latent gradient.
12. **Unified priors exist but are not trainable yet.** `cms_data.load_theory_prior_z` reads only `zData`, ignores `FDL/component_id` and `FDL/weight`; a mixture-at-load path must be built before `data/legacy/priors/*.hdf5` can replace the legacy priors.
13. **The flat train loss is the distributional floor, not a dead objective.** artifact-measured: Run E train loss is 0.795 → 0.392 then flat (stage 1); the J/ψ region is 75% latent `L_z` then 41-54% `L_z` + `L_x`, and the Z region sits at 0.118-0.124 from epoch ~60 and contributes almost no gradient. In the refactored noise-control runs `eval_loss` still falls 0.943 → 0.420 (stage 3) and 0.394 → 0.360 (stage 4) while the train loss is flat.
14. **There are no variable loss coefficients to change.** source-verified: `beta`, `lamb`, `tau`, `nu_e`, `nu_d`, `rho` are scalars in every joint config. `_resolve_scheduled_value` supports `{start, end, schedule}`, but no joint config uses it for a loss term. The only per-epoch coefficients are the noise multipliers and the LR; loss weights change only at stage boundaries.
15. **Inference-time noise does not move the Upsilon mass.** artifact-measured (`tail_noise_sweep_stage4.json`, LowBoth/LowCore stage 4 on the labelled Υ states): setting core 0→0.5 and tail 0→1 changes the Υ(1S) decoded median bias by < 0.5 MeV and the decoded std by < 0.1 MeV. The OOD peak shift/width is baked into the learned mean map, not produced by sampling at decode time.
16. **The ppzee encoder error is 95.5% detector-inherited and common-mode.** artifact-measured (`outputs/cms_Joint/ppzee/error_decomposition/`): corr(fractional model residual, fractional identity residual) = 0.953, identity covariance share 0.955, encoder-added share 0.045. Detector per-muon log-pT errors are 13.6% and anti-correlated (ρ = −0.69), i.e. mostly differential and mass-inert; encoder residuals are 5.0% and correlated (ρ = +0.97), i.e. almost purely common-mode and mass-moving.
17. **In-distribution peak asymmetry is target-inherited; model-created asymmetry is OOD.** artifact-measured, frozen Run E epoch-65 decode: Z CMS `x_test` skew −0.504 vs decoded `D(z)` −0.604; J/ψ CMS +0.076 vs decoded −0.002 (a small shape mismatch); Υ prior subset −5.27 vs decoded best −4.84; the OOD high-pT fake J/ψ prior flips from −5.82 to +53.7 (std 0.020 → 0.456 GeV).
18. **TF32 is a ~1e-4-level perturbation here, and not the throughput lever.** artifact-measured (`scripts_joint/tf32_impact_probe.py`, Run G 4x128, 8,192-event batch, identical seeds): pure matmul max relative error 2.8e-4; model output max/rms relative error 4.9e-7 / 1.8e-8; loss relative differences 1e-5 to 3.7e-4; gradient-norm relative differences 2e-5 to 2.6e-4; matmul-only speedup 1.30x at 16384x256x256. source-verified: the pipeline bottleneck is the sorts in the sliced-Wasserstein/W1 terms, which TF32 does not touch, so the end-to-end gain is smaller. `performance.tf32` is opt-in and recorded in `provenance.json`; it is not bit-comparable with the strict-fp32 record.
19. **The invariant-mass mask and the no-mass-anchor rule are now enforced, not incidental.** source-verified: `run_joint.assert_mass_not_conditioned` rejects feature index 8 (`log_pair_mass`) in either map, and `run_joint.assert_no_mass_anchor` rejects a nonzero `resonance_mass_w1`/`mass_w1` in any region's merged loss unless `allow_mass_anchor: true` is declared. All joint configs A-G and the Run E/F checkpoints already satisfy both: 13-D condition mask without index 8, anchors 0. Tests: `test_joint_contract_masks_invariant_mass_and_anchors`, `test_mass_anchor_and_condition_guards_reject_violations`. The distributed mass terms (`pair_mass_w1`, `mass_kin_swd`, `physics_swd`, the per-event cycle) remain on -- those are targets, not anchors.
20. **The asymmetric decoded peaks are a conditional (per-pair-pT) response error, not a prior or selection effect.** artifact-measured (`scripts_joint/slice_resolved_diagnostic.py`, Run F `best_model`, noise 0, equal-count truth pair-pT quartiles): the standardized mass W1 inside a slice is 3.5-17x its finite-sample floor (J/psi decode 0.066-0.087 vs floor 0.005-0.019, encode 0.039-0.106 vs 0.010-0.021; Z decode 0.035-0.137 vs 0.008-0.014, encode 0.036-0.140 vs 0.007-0.014), while the global marginals are matched. Pooled over slices the response looks fine; inside a slice it does not. Supporting: reweighting the decoded 1S to the CMS pair-pT made the asymmetry worse (best -0.27 -> -0.42, last -0.42 -> -0.79), and removing decoder events pushed out of `pT>3, |eta|<2.4` changed it by <0.005. The fix is a conditional/sliced loss or a pT-resolved calibration, not prior reweighting.
21. **Per-slice finite-sample floor budget.** artifact-measured: at the real per-update cardinality (`min_events_per_update` 4096 / Z 4651) the per-slice floor rises ~1/sqrt(n_slice) -- J/psi 0.020 (1 slice) / 0.038 (2) / 0.031 (3) / 0.048 (4) / 0.071 (5); Z 0.030 / 0.036 / 0.061 / 0.052 / 0.059. Acceptance used for Run H: 3-4 slices, signal/floor >= 3, and added floor <= ~0.03 in loss units (<=20% of the ~0.15 learnable excess).

22. **Run H's stage-3 tail stage breaks the OOD Upsilon peak scale, and it is not
    the noise.** artifact-measured (`outputs/cms_Joint/Run_H/tail_audit/`): the
    ep-180 checkpoint puts the three Upsilon medians at -125/-136/-136 MeV
    versus the CMS fit (~-1.4% coherent scale), while ep-90/100/115 sit at
    -14/-17/-18 MeV (1S). Decoding ep 180 with tail 0/0.25/0.5/1.0 changes the
    1S median by 0 MeV, and forcing tail 0.5 onto the healthy checkpoints
    changes nothing, so the shift lives in the deterministic mean map, not the
    sampled tail.
23. **The Run H decoder collapses its stochastic channel.** artifact-measured
    (forward hooks, same 400k events): learned `tail_sigma` 8.1e-4 (init) ->
    ~1.8e-4, `core_sigma` pinned at the 1e-3 floor; effective Upsilon mass noise
    ~10 MeV (core) + ~0.8 MeV (tail at 0.5) versus the 84 MeV CMS resolution.
    Turning the noise off changes the decoded 1S std by ~15 MeV out of 0.282
    GeV, so the resolution lives in the deterministic mean map. Hypothesis:
    under a marginal-only objective the (mean spread, noise) split is
    unidentifiable, so the optimiser takes the cheap solution; that mean-map
    channel is what drifts in the untrained Upsilon region. The Run H instance
    of items 4/6.
24. **Stage 3 degrades the trained regions, as a boundary shift plus a mild
    train/validation gap.** artifact-measured (Run H `history.json`, 17
    validation draws per stochastic stage): the J/psi normalised score is
    mean 1.44 / median 1.42 / min 0.84 (std 0.42) in stage 2 and mean 2.59 /
    median 2.62 / min 2.02 (std 0.33) in stage 3; Z drifts 0.55 -> 0.62. The
    jump is at the stage boundary (tail 0.25 -> 0.5 plus a fresh optimizer at
    constant LR), not monotone. Within stage 3 the train loss still falls
    (first-10 mean 0.786 -> last-10 0.711, `jpsi_z_loss` 0.644 -> 0.559) while
    the validation loss is flat (mean 0.726) and the score does not improve.
    source-verified: validation is deterministic (all-zero noise multipliers,
    8,192 events), so that spread is real epoch-to-epoch weight oscillation
    under the stochastic loss, not sampling noise; EMA and LR decay are what
    must smooth it. hypothesis for the gap: the J/psi latent stream has only
    417 unique events, padded to 4096 per update (item 11), so it can be
    memorised.

25. **Run H fix: the mean-map anchor slows the drift but does not fix the peak
    shape.** artifact-measured (sections 1.5): the anchor + LR decay + EMA +
    weight decay cut the stage-3 in-domain score from Run H's 2.59 mean to 1.32
    and fixed the Upsilon centre (+11/-3/+2 MeV at ep180), but the last
    checkpoint now over-spreads (1S std 0.330, skew -1.54) while the stable
    checkpoints are narrow and -66 to -85 MeV low. Anchoring to the stage-1 map
    also freezes that map's -50/-64/-62 MeV bias. The conditional asymmetry
    persists for J/psi (per-slice W1/floor 12.7-34.3 at ep180) though Z is
    fixed (1.1-2.7). The next lever is noise identifiability (a calibrated
    core/tail sigma floor), not a stronger anchor.

## 3. Key measured numbers (quick reference)

### 3.1 Run E history (artifact-measured)

- Stage bests: ep65 score 0.612 (noise 0/0), ep73 0.666, ep332 1.618, ep421 1.194.
- Stage 2 never improved on its first epoch; stage 3+4 never beat stage 1.
- 75 of 92 validations passed all hard gates; all 17 failures occur after noise is on.

### 3.2 Upsilon transfer, frozen Run E (artifact-measured)

best_model (ep65, deterministic):
- Matched 8,300-event mass W1 identity/model/floor = 0.23668/0.27764/0.01295 GeV, gauge 1.183.
- Per-state mean shifts -52.76/-55.58/-57.70 MeV; decoded robust widths 21.7/23.9/25.0 MeV vs prior 0.317/0.353/0.423 MeV.

last_model (ep432, stochastic):
- Matched 8,300-event mass W1 identity/model/floor = 0.23668/0.09698/0.01295 GeV, gauge 0.376.
- Per-state mean shifts +184.9/+199.6/+205.8 MeV; decoded state stds 480/512/537 MeV.
- Mass-window retention 87.87% vs best_model 98.36%.

### 3.3 ppzee (artifact-measured)

- Detector resolution rms(mass(x)-mass(z)) = 2.6515 GeV; upstream encoder rms 8.8305 GeV.
- Oracle floor for mass-only correction = 0.8099; oracle best constant offset = 0.9647.
- Our encoder `residual_rms_vs_identity` = 0.9843 (stage-2 selected) / 0.9808 (stage-1).

### 3.4 Unified priors (Session 29, artifact-measured)

| region | file | events | components | TMS |
|---|---|---|---|---|
| Z | `data/legacy/priors/z_unified_bare_tms22p8.hdf5` | 697,180 | continuum | 22.8 |
| J/psi | `data/legacy/priors/jpsi_unified_bare_tms10.hdf5` | 1,736,351 | jpsi 737,262 + continuum 999,089 | 10.0 |
| Upsilon | `data/legacy/priors/upsilon_unified_bare_tms10.hdf5` | 3,295,815 | 1S/2S/3S/continuum | 10.0 |

All unmixed, unweighted at load, labelled. `data/legacy/priors/_compat_probe.hdf5` still needs a read test from the Windows `cms` env.

### 3.5 ppzee error decomposition (2026-09-10, artifact-measured)

New tool `scripts_joint/ppzee_error_decomposition.py`, output
`outputs/cms_Joint/ppzee/error_decomposition/{error_decomposition.json,REPORT.md}`.
It reproduces the committed aggregate numbers and adds the component split.

- Aggregate (GeV): identity mean/std/rms = -0.6988 / 2.5578 / 2.6515; model
  -0.1036 / 2.6078 / 2.6099; `residual_rms_vs_identity` 0.9843; bias removed
  0.852; scatter ratio 1.020. Constant-offset oracle 0.9646; affine oracle on
  identity 0.9338; affine oracle on the model 0.9829 (in-sample, optimistic).
  The committed `oracle_floor_ratio` 0.8099 is reported, not recomputed.
- Inherited vs added: corr 0.9530, R² 0.9058, identity covariance share 0.955,
  encoder-added share 0.045 (added rms 0.01109).
- Components (detector / encoder): per-muon log-pT std 0.136 / 0.050;
  corr(mu-,mu+) −0.686 / +0.967; common-mode std 0.0538 / 0.0499;
  differential std 0.1246 / 0.0065; R²(common + angular) 0.759 / 0.932;
  remainder rms 0.0142 / 0.0077.
- Mass shape skew: truth z +10.89, detector x +9.82, encoder +8.87 (ppzee sample
  is right-tailed at ~91 GeV).

## 4. Next plans (agreed 2026-09-08)

### Plan 1 — J/psi prior shift test (diagnostic, no training)

**Goal:** determine whether the Upsilon failure is mass-scale-driven or kinematics/shape-driven.

**Method:**
1. Take the unified bare J/psi signal component (`component_id == jpsi`, robust width 0.175 MeV) and scale all four-vectors by `9.4603 / 3.0969` so the mass peak lands at the Upsilon.
2. Decode with frozen Run E `best_model.pt` (and optionally `last_model.pt`).
3. Measure the self-shift: `mass(decoded) - mass(fake prior)`, robust-width ratio, window retention, and median pair pT.
4. Compare against the real Upsilon transfer biases in section 3.2.

**Preliminary result (artifact-measured, 2026-09-08, 20k events, CPU):**

| | fake J/psi signal @ 9.46 GeV | real Upsilon 1S |
|---|---|---|
| median mass shift | -46.6 MeV (-0.49%) | -56.8 MeV (-0.60%) |
| mean mass shift | -21.1 MeV | -52.8 MeV |
| prior robust width | 0.58 MeV | 0.32 MeV |
| decoded robust width | 37.7 MeV | 21.7 MeV |

The fake reproduces much of the failure: narrow core broadens and shifts down. It does **not** fully match the real shift, and its pair pT (median 44.7 GeV) is far above real Upsilon (2.98 GeV).

**Caveat:** one scaling test moves mass and kinematics together, so it cannot cleanly separate mass vs kinematics sensitivity alone. It is a first discriminator.

**Formalized version:** write `scripts_joint/mass_scale_probe.py`, write outputs under `outputs/cms_Joint/Run_E/mass_scale_probe/`, and run a small matrix:
- unified bare J/psi signal scaled to 9.46 GeV (narrow, high pT),
- unified Z scaled down to 9.46 GeV (broad, low pT),
- real Upsilon prior (narrow, true low pT),
- real Upsilon prior scaled by +/- 3-5% (same kinematics, slightly different mass).

Interpretation:
- all three 9.46 GeV probes fail similarly -> mass-sensitive;
- high-pT fake fails but low-pT probes do not -> kinematics-sensitive;
- narrow fakes fail but broad Z fake does not -> width/shape-sensitive.

### Plan 1 results — scaling probes (2026-09-08, artifact-measured)

New tools under `experiments/prior_JpsiScalingTest/`.

1. **Fake J/psi CKKW-L prior scaled to Upsilon(1S) mass.** 20k CPU events, scaled all four-vectors by `9.4603/3.0969`, decoded with frozen Run E.

   | quantity | fake J/psi scaled | true Upsilon(1S) |
   |---|---|---|
   | best_model median decoded shift | -7.9 MeV (-0.084%) | -52.5 MeV (-0.554%) |
   | best_model decoded robust width | 29.6 MeV | 21.8 MeV |
   | last_model median decoded shift | -1028 MeV (-10.9%) | +370 MeV (mean +186 MeV) |
   | last_model decoded robust width | 1165 MeV | 363 MeV |
   | last_model window retention | 38.2% | 94.9% |
   | input pair-pT median | ~80 GeV | ~3 GeV |

   Best deterministic model does not reproduce the true Upsilon shift; stochastic last model is badly broken on the high-pT fake.

2. **Mass-only probe.** 20k CPU events; true Upsilon(1S) energies rescaled to move E-based mass by +/-3%/+/-5% while px,py,pz unchanged. Decoded stable mass response <0.02 MeV for both checkpoints. Conclusion: without an explicit mass condition feature, a pure mass change with fixed momenta is almost invisible to the decoder.

3. **Kinematics-only probe.** 5k CPU events; true Upsilon(1S) Lorentz-boosted to pair-pT targets 0/5/10/20/40/80 GeV while preserving E-based mass and rapidity. Strong decoded-mass response; especially last_model degrades sharply with higher pair pT (retention falls, widths grow).

4. **Interpretation.** The model is kinematics-driven; noise level in later stochastic stages is a prime suspect for the Upsilon/last-model degeneration. Three noise-control runs are prepared (below).

### Plan 2b update — noise-control comparison runs (prepared, not trained)

Comparison arms keep original 432 epochs; only noise ceilings change.

- `Run_E_noiseLowCore`: stage2 core ramp `0.1 -> 0.5`, stages3-4 core `0.5`, tail unchanged.
- `Run_E_noiseLowTail`: stage3 tail end `0.125`, stage4 tail `0.125 -> 0.5`, core unchanged.
- `Run_E_noiseLowBoth`: both of the above.
- `Run_E_short`: future 360-epoch schedule `60+100+120+80`, same noise as Run E.

Readout: compare stage-best and last-model checkpoints on J/psi, Z, Upsilon held-out, fake prior, mass-only, and kinematics-only probes. If low-noise arms improve later stochastic checkpoints without hurting J/psi/Z, noise is the cause.

### Plan 2a — Mass-shift fix (controlled training run)

**Hypothesis:** the objective lacks an absolute mass-scale anchor, so the decoder is free to drift at masses outside the trained regions.

**Run:** new config `configs_joint/cms_Joint_runE_massAnchor.yaml` extending Run E, one controlled change:

```yaml
extends: cms_Joint_runE.yaml

run_label: Run E + physical resonance-mass anchor
run_name: Run_E_massAnchor

loss:
  resonance_mass_w1: 1.0

regions:
  jpsi:
    loss:
      resonance_mass_center: 3.0969
      resonance_mass_half_width: 0.06
  z:
    loss:
      resonance_mass_center: 91.1876
      resonance_mass_half_width: 10.0
```

**Readout:** Upsilon transfer state mass shifts should shrink from ~-0.56% if the hypothesis is right; J/psi and Z direct mass shifts should stay small.

### Plan 2b — Noise-training degeneration fixes (ordered)

1. **Commit the in-flight refactor** after `--dry-run` and `--smoke --num-samples 20000` preflight. This makes `eval_loss` visible and selection score meaningful.
2. **Stage-restart policy:** add `checkpoint_selection.stage_restart_policy: last|best` (default `best` for reproducibility) and test `last` on ppzee (~16 min). `last` should avoid the stage-boundary rewind + fresh-optimizer + noise-jump spike.
3. **Encoder-noise hygiene:** for delta-like regions, keep `encoder_core_noise_multiplier: 0` and `encoder_tail_noise_multiplier: 0` in all stages. This prevents later stages from actively worsening the encoder. It is necessary, not sufficient (M1 is closed).
4. **Batch-cardinality cap:** if curves remain noisy, add a per-update z-batch floor or x:z cap in full-pass epochs and test on a fast bench.
5. **If the encoder still does not unfold:** the next branch is **M2** — amortized posterior q(z|x) with a proper scoring rule / entropy term, developed on ppzee and scored by `residual_rms_vs_identity`, mass gauge, pull, and coverage. Never by latent pair-pT KS or x->z C2ST on ppzee.

### Plan 3 — constant shared-noise arm (2026-09-09 meeting; prepared, dry-run only)

**Decision:** keep the stage-1 deterministic warmup; keep the encoder and
decoder on one schedule; remove every per-epoch noise ramp and run stages 2-4
at one constant level, encoder = decoder.

**Config:** `configs_joint/cms_Joint_runE_constNoise.yaml` (extends Run E; only
the noise schedule changes). Stage 1 stays 0/0; stages 2-4 are core 1.0,
tail 0.25, shared. Loss coefficients, LR schedule, model, data and stage
lengths are inherited unchanged, because the loss coefficients were already
scalar per stage (section 2 item 14).

**Preflight:** `python scripts_joint/run_joint.py --run E_constNoise --device cuda --dry-run`
— artifact-measured 2026-09-10, passed; 5,343,304 parameters, 432 epochs,
Upsilon not opened. **Not trained.**

**Readout if trained:** compare against Run E and the noise-control arms on
(a) `eval_loss`, (b) the J/ψ and Z direct/cycle mass gauges, (c) the Υ
fractional mass scale, and (d) the ppzee common-mode log-pT residual. The
question is whether the ramps were doing work or only producing the
stage-boundary spike.

**Retired:** `cms_Joint_runE_encoderDet*` (encoder deterministic, decoder
stochastic) and its two tests, per the same decision. The code path
`set_component_noise_multipliers` remains, but no active config uses it.

### Run F — flat-loss mechanism validation (2026-09-10, prepared, not trained)

Reuses the Run F name. The original Run F prior contract is preserved verbatim
as `configs_joint/cms_Joint_runF_priorCKKWL.yaml` (`run_name:
Run_F_priorCKKWL`), so the two cannot collide.

**Diagnosis being tested (section 2 items 13-14, measured 2026-09-10):** at Run
E's real per-update cardinalities the distributional losses are dominated by an
irreducible finite-sample floor, inflated by the full-pass batch imbalance.
`scripts_joint/loss_floor_probe.py` quantifies it (artifact-measured, 8 draws):

| config | J/ψ L_x floor | J/ψ L_z floor | Z floors | weighted distributional floor |
|---|---|---|---|---|
| Run E (no floor) | 0.347 | 0.329 | 0.113 / 0.116 | 0.452 |
| Run F (4096 floor) | 0.126 | 0.123 | 0.113 / 0.116 | 0.239 |

**Fixes in Run F** (multi-variable by design; it is a validation run, not a
single-variable attribution):
1. `loaders.min_events_per_update: 4096` — pads the 417-event J/ψ prior stream
   with replacement, lowering its floor ~2.8× and the gradient variance. Unique
   rows visited per epoch are unchanged. source-verified: new
   `_expand_indices` in `joint_train_utils.py`, wired into `train_joint_epoch`;
   the effective per-update cardinalities are logged as `*_batch_per_update`.
2. One constant shared encoder=decoder noise level (core 1.0 / tail 0.25) in
   stage 2; no ramps, so the reconstruction floor stops rising.
3. Two stages with identical loss coefficients (`beta/lamb/tau = 1`, anchors
   off, `num_slices = 256`), removing the stage rewind and the objective jump.
4. `model.hidden_dims = [256, 256, 256, 256]` (4×256) instead of `[512]×6` —
   830,536 parameters versus 5,343,304. Switch to 4×512 by editing one line.

**Preflight:** `--dry-run` passed (100 epochs, 830,536 parameters, Upsilon not
opened); `--smoke --skip-evaluation` passed (2 stages, 14 updates, stage-best
and global checkpoints written). **Not trained.**

**Train:**
```
python scripts_joint/run_joint.py --run F --device cuda
```

**Not included, deliberately:** the physical-GeV mass anchor (Plan 2a) and the
M2 per-event inverse objective. Those are separate hypotheses.

**Code changes:** `scripts/cms_data.py` gained the opt-in `replace_keys`
directive (replace, not merge, inherited named lists); `scripts_joint/
joint_train_utils.py` gained `_expand_indices`; `scripts_joint/joint_trainer.py`
reads `min_events_per_update` and applies it to both sampling modes. Tests:
`tests/test_joint_runa.py` now has five tests for these paths.

### Run F Upsilon continuum diagnostic (2026-09-10, artifact-measured)

Tool: `scripts_joint/upsilon_continuum_reweight.py`; output
`outputs/cms_Joint/Run_F/upsilon_continuum_reweight/`. The CMS background is
data minus the fitted 1S/2S/3S Gaussians; the continuum shape ratio is fitted
on the sidebands 8.5-9.1 and 10.7-11.2 GeV only, then applied to the continuum
component.

Normalized density in 8.5-9.25 GeV, ratio to CMS:

| sample | unweighted | continuum reweighted |
|---|---|---|
| Run F best_model (stage-1 ep20, deterministic) | 1.756 | **1.033** |
| Run F last_model (stage-2 ep100, stochastic) | 1.976 | **1.265** |

- The 8.5-9.25 excess is **prior-continuum shape**: a sideband-derived
  reweight closes the deterministic model's excess from 1.76 to 1.03. The
  interpolated 9.1-9.25 sub-bin closes with the fitted 8.5-9.1 sub-bin, so it
  is one smooth continuum-shape error, not a local artifact.
- The stochastic `last_model` keeps a **+27% residual** after the same
  reweight. That part is model-created (the 1S tail spilling down: signal
  fraction in 8.5-9.25 rises from 2.3% prior to 12.8% last). So the last model
  is *not* better on this diagnostic.
- A fresh decode of the resampled prior with `last_model.pt` reproduces the
  post-hoc reweight to 0.4% (1.269 vs 1.265), confirming reweight-then-decode
  ≡ reweight-after-decode for a per-event map.

### Run F Upsilon checkpoint comparison (2026-09-10, artifact-measured)

All Run F checkpoints decoded on the same labelled 0j1j prior (last two rows on
the continuum-reweighted prior); metrics from `z_to_x_metrics.json` and the
8.5-9.25 ratio from the decoded spectra. Summary:
`outputs/cms_Joint/Run_F/upsilon_transfer_comparison/REPORT.md` + figure.

| sample | mass W1 [GeV] | mass KS | pair-pT KS | C2ST MLP | 1S mean [GeV] | 1S std [MeV] | 8.5-9.25/CMS |
|---|---|---|---|---|---|---|---|
| best_model (stage-1 ep20, det.) | 0.3069 | 0.2063 | 0.3305 | 0.808 | 9.3546 | 116.5 | 1.756 |
| stage-2 best (ep45) | 0.3158 | 0.1930 | 0.3405 | 0.817 | 9.3206 | 193.1 | 1.869 |
| last_model (stage-2 ep100) | 0.3316 | 0.1686 | 0.3183 | 0.818 | 9.2793 | 284.9 | 1.976 |
| last_model + continuum reweight | 0.0801 | 0.0548 | 0.3137 | 0.795 | 9.2793 | 284.8 | 1.269 |
| **best_model + continuum reweight** | **0.0376** | 0.0833 | 0.3246 | 0.793 | 9.3545 | 116.4 | **1.047** |

Read: the continuum correction is the dominant fix (mass W1 0.31 → 0.038 for the
deterministic model); the deterministic model keeps the 1S at 116 MeV while the
stochastic checkpoints broaden it to 193/285 MeV and worsen the low-mass tail.
The residual response error is common to all checkpoints: the 1S mean sits at
~9.35 GeV, ~90 MeV below the CMS fit 9.4451. KS and W1 disagree for the sharp
peak (sharp+shifted gives a lower W1 but a higher KS than broad).

### Run G — three-stage Run F variant (2026-09-10, prepared)

The stage list and the network width change; everything else is inherited
(`min_events_per_update` 4096, identical loss coefficients, constant shared
encoder=decoder noise, constant LR in the stochastic stages).
Model width: `hidden_dims = [128, 128, 128, 128]` (4x128; Run F was 4x256).

| stage | epochs | core | tail |
|---|---|---|---|
| runG_stage1_deterministic_warmup | 20 | 0 | 0 |
| runG_stage2_stochastic_core | 60 | 1.0 | 0.25 |
| runG_stage3_stochastic_tail | 60 | 1.0 | 0.5 |

Stage 3 differs from stage 2 only in name, epochs and tail multiplier. Total
140 epochs. Config: `configs_joint/cms_Joint_runG.yaml`.

### Run H — pT-sliced conditional mass (2026-09-10, prepared)

Fix for item 20. Run G's three-stage mechanism contract, but:

| | Run G | Run H |
|---|---|---|
| `hidden_dims` | [128]*4 | **[256]*4** |
| epochs | 20 / 60 / 60 | **20 / 80 / 80** |
| `num_slices` per stage | 128/192/256 | **256/256/256** |
| `mass_kin_swd` | 0.5 | **0.0** (moved, not stacked) |
| `sliced_mass_w1` | 0.0 | **0.5** (4 equal-count pair-pT slices of the truth, min 512 events/slice) |

The sliced term is a new first-class loss component in `scripts/loss.py`
(`SpaceFeatureOTLoss.sliced_mass_w1`, weight default 0 so all earlier configs
are unchanged); it uses the truth sample's pair-pT quartiles and the existing
standardized mass W1 inside each slice. Config: `configs_joint/cms_Joint_runH.yaml`.
Preflight dirs `outputs/cms_Joint/Run_H_dryrun/`, `outputs/cms_Joint/Run_H_smoke/`
(both passed; 180 epochs, 830,536 parameters; the sliced term ran in the smoke).
Tests: `test_run_h_sliced_loss_contract`, `test_sliced_mass_w1_component_and_validation`.
Not trained.

## 5. Where things live (current)

- Training entry: `python scripts_joint/run_joint.py --run <ID> --device cuda` (`<ID>` resolves `configs_joint/cms_Joint_<ID>.yaml`).
- Identity/floor references: `scripts_joint/identity_reference.py`, `identity_baseline.py`.
- Paired closure: `scripts_joint/paired_closure.py`, `score_paired_closure.py`, `paired_leakage_audit.py`.
- Upsilon transfer: `scripts_joint/upsilon_transfer_test.py`, `scripts_joint/upsilon/*`.
- Data layout after 2026-09-08 cleanup: current CKKW-L priors + CMS ROOT files stay in `data/`; all moved legacy files live under `data/legacy/` (including the old `data/priors/` tree).
- Prior-scaling probes and tools: `experiments/prior_JpsiScalingTest/`.
- Noise-control retrain configs: `configs_joint/cms_Joint_runE_noiseLowCore.yaml`, `cms_Joint_runE_noiseLowTail.yaml`, `cms_Joint_runE_noiseLowBoth.yaml`; future short schedule: `cms_Joint_runE_short.yaml`.
- Constant shared-noise arm: `configs_joint/cms_Joint_runE_constNoise.yaml`; dry-run output `outputs/cms_Joint/Run_E_constNoise_dryrun/` (not trained).
- Run F mechanism validation: `configs_joint/cms_Joint_runF.yaml`; preserved prior contract `configs_joint/cms_Joint_runF_priorCKKWL.yaml`; loss-floor probe `scripts_joint/loss_floor_probe.py`, outputs `outputs/cms_Joint/loss_floor_probe/{E,F}.json`; preflight dirs `outputs/cms_Joint/Run_F_dryrun/`, `outputs/cms_Joint/Run_F_smoke/`.
- Run G three-stage 4x128 variant: `configs_joint/cms_Joint_runG.yaml` (20 + 60 + 60; stage 3 core 1.0 / tail 0.5; hidden_dims [128]*4); preflight dirs `outputs/cms_Joint/Run_G_dryrun_4x128/`, `outputs/cms_Joint/Run_G_smoke_4x128/` (dry-run + smoke passed, not trained). TF32 arm: `configs_joint/cms_Joint_runG_tf32.yaml` (same config, `performance.tf32: true`); impact probe `scripts_joint/tf32_impact_probe.py` → `outputs/cms_Joint/tf32_probe/tf32_impact.json`.
- Run H pT-sliced conditional mass: `configs_joint/cms_Joint_runH.yaml` (4x256, 20/80/80, num_slices 256, `mass_kin_swd` 0 -> `sliced_mass_w1` 0.5, 4 slices / min 512); sliced component `scripts/loss.py::SpaceFeatureOTLoss.sliced_mass_w1`; diagnostic `scripts_joint/slice_resolved_diagnostic.py` → `outputs/cms_Joint/slice_diagnostic/{REPORT.md,slice_diagnostic.json}`; pT-response test `scripts_joint/upsilon_pt_response_test.py` → `outputs/cms_Joint/Run_F/upsilon_pt_response/`; preflight dirs `outputs/cms_Joint/Run_H_dryrun/`, `Run_H_smoke/`.
- Invariant-mass contract guards: `scripts_joint/run_joint.py::assert_mass_not_conditioned` (index 8 never a condition input, both maps) and `assert_no_mass_anchor` (no `resonance_mass_w1`/`mass_w1` anchor unless `allow_mass_anchor: true`); enforced in `main()` before training. Tests: `test_joint_contract_masks_invariant_mass_and_anchors`, `test_mass_anchor_and_condition_guards_reject_violations`.
- Run F Upsilon continuum diagnostic: `scripts_joint/upsilon_continuum_reweight.py` → `outputs/cms_Joint/Run_F/upsilon_continuum_reweight/` (`REPORT.md`, `upsilon_prior_continuumReweighted.hdf5`, `decoded_reweighted_last_model.hdf5`). Prior-vs-reweighted figure + table: `scripts_joint/upsilon_prior_reweight_plot.py` → `prior_reweight_comparison.{png,pdf,md,json}` in the same directory.
- Run F Upsilon-transfer evaluations under `outputs/cms_Joint/Run_F/`: `upsilon_transfer_{best_model,stage2best,last_model,continuumReweighted_best,continuumReweighted_last}/`, each with `decoded/`, `prior_vs_cms/` and/or `decoded_vs_cms/`, `quantitative_z_to_x/`. Combined table + figure in `upsilon_transfer_comparison/` via `scripts_joint/upsilon_transfer_comparison.py`.
- Checkpoint policy (2026-09-10): global `best_model.pt`, per-stage `best_<Run>_<stage>.pt`, global `last_model.pt`, and **new** per-stage `last_<Run>_<stage>.pt` (latest evaluated epoch of each stage, overwritten at every validation).
- ppzee error decomposition: `scripts_joint/ppzee_error_decomposition.py` → `outputs/cms_Joint/ppzee/error_decomposition/{error_decomposition.json,REPORT.md}`.
- Noise-control run outputs live under `outputs/cms_Joint/Run_E/Run_E_noiseLow{Core,Both}/` with their own `history.json` and `joint_evaluation.json`.
- Key outputs: `outputs/cms_Joint/Run_E/`, `outputs/cms_Joint/AB_narrow_split/`, `outputs/cms_Joint/ppzee/`, `outputs/cms_Joint/unifiedP1/`.
- Run H tail audit: `scripts_joint/runH_tail_audit.py` -> `outputs/cms_Joint/Run_H/tail_audit/{tail_audit.json,REPORT.md,peak_shift_vs_tail.png}`; diagnosis + fix proposal in `DIAGNOSIS_AND_FIX.md` there.
- Drift control: `scripts_joint/joint_anchor.py`; guard `run_joint.assert_mean_map_anchor_contract`; arms `configs_joint/cms_Joint_runH_anchor.yaml`, `configs_joint/cms_Joint_runH_fix.yaml`; tests `tests/test_joint_anchor.py`.
- Tests: `python -m unittest discover -s tests` (use unittest, not pytest).

## 6. Session log (condensed, post-Run E)

### 2026-09-06 — Session 29, unified priors built
Delivered the three bare, labelled, gate-passing priors in `data/priors/`. Established: the old 1M reweighted J/psi prior had ESS 63,037/1,000,000; the Upsilon prior was never smeared (robust widths 0.317/0.353/0.423 MeV); truth variant is `bare`; J/psi TMS scan fails catastrophically (TMS 5/10/20 changes pair-pT median by ~93%); priors are not trainable until component mixture loading exists.

### 2026-09-07 — Session 30, ppzee paired closure
Ran our Run E architecture unpaired on ppzee. Result: encoder `residual_rms_vs_identity` 0.9843 vs identity 1.0, oracle 0.8099, upstream 3.3304; all leakage checks pass. Decoder manufactures the entire recoil spectrum. ppzee becomes the development bench for M2/M5, with the restriction that M2 is scored by per-event residual/pull/coverage, never latent pair-pT KS or x->z C2ST.

### 2026-09-07 — Sessions 31-33, Upsilon transfer evaluations
unifiedP1 fallback checkpoint and Run E best/last checkpoints evaluated on legacy and unified Upsilon priors. Run E best_model mass W1 gauge 1.183; last_model mass gauge 0.376 but pair-pT worsens and peaks wash out (section 3.2).

### 2026-09-08 — Session 34, read-only audit
Findings in section 2 items 7-11: Upsilon bias is a constant fractional mass scale; stage rewind explains the loss spike; standardized mass terms are region-asymmetric; full-pass batch cardinality is very unequal; anchor terms are numerically inert. No training or artifacts modified.

### 2026-09-08 — current session, condensed rewrite + next plans
Rewrote `memory.md` to start from Run E. Agreed next plans: Plan 1 (J/psi prior shift test), Plan 2a (Run_E_massAnchor), Plan 2b (noise degeneration fixes, then M2 if needed). Preliminary Plan 1 probe measured (section 4, Plan 1).

### 2026-09-08 — Session 35, scaling probes + noise-control preparation (user session)

Work done in this session:

1. **Scaffold/notebooks created** under `scaffold/`: CMS peaks vertical 3x1, Run A-E loss comparison, Run E best/last mass-density ratio notebook, plus prior-generation description.
2. **Prior J/psi-scaling test added** under `experiments/prior_JpsiScalingTest/`.
   - Created fake Upsilon(1S) prior by scaling the J/psi CKKW-L signal component by `9.4603/3.0969`.
   - Decoded with frozen Run E best/last checkpoints and compared against true Upsilon(1S).
   - Results are in `MASS_ONLY_RESULTS.md`, `RESULTS.md`, `outputs/*.json`, `plots/`.
   - Paper-style plots now compare fake decoded vs true decoded vs CMS.
3. **Mass-only probe**: rescale true Upsilon(1S) stored energies by +/-3%/+/-5% while keeping momenta fixed. Decoded stable-mass response was <0.02 MeV for both checkpoints. Conclusion: pure mass change with fixed kinematics is nearly invisible because Run E excludes the explicit pair-mass condition feature.
4. **Kinematics-only probe**: Lorentz-boost true Upsilon(1S) events to pair-pT 0/5/10/20/40/80 GeV while preserving E-based mass and rapidity. Decoded mass changes strongly, and last_model degrades sharply at high pair pT. Conclusion: the model is kinematics-driven; high stochastic noise plus OOD kinematics is the leading suspect for Upsilon/last-model degeneration.
5. **Data cleanup**: `data/` now keeps only the current CKKW-L priors and CMS ROOT files. All other files were moved to `data/legacy/`, including the former `data/priors/` tree. Config paths were updated to `legacy/...` where needed.
6. **Retrain prep**:
   - `configs_joint/cms_Joint_runE_short.yaml`: 360-epoch future schedule (`60+100+120+80`).
   - Three comparison configs with original 432 epochs and lower noise ceilings:
     - `Run_E_noiseLowCore`
     - `Run_E_noiseLowTail`
     - `Run_E_noiseLowBoth`
   - Documentation in `configs_joint/noise_control_experiments.md`.
7. **Checkpoint naming bug fixed**: stage-best `.pt` files are now run-prefixed, e.g. `best_RunE_stage1_deterministic_identity.pt`, not `best_runA_stage1_...`. Existing run directories were renamed accordingly; `joint_trainer.py` has a legacy fallback for old names.
8. **Speed ideas for future runs**: use `Run_E_short`, optionally TF32, less frequent validation, lower `num_slices`, or larger batches. No training code changed for the comparison arms.

No retraining has completed yet. The noise-control comparison runs are ready to launch.

### 2026-09-09 — Session 36, noise-control runs completed (recorded 2026-09-10)

`Run_E_noiseLowCore` and `Run_E_noiseLowBoth` trained the full 360-epoch
stage-2→4 schedule, warm-started from Run E stage 1. Stage 4 is the best
checkpoint in both (0.588 / 0.575) and `eval_loss` keeps falling through stage
4 (0.394 → 0.360). `Run_E_noiseLowTail` was not run. This is the first
artifact-measured evidence that the "longer training never helps" reading was
partly the retired hard-gate selection.

### 2026-09-10 — Session 37, meeting follow-up audit

Five meeting points resolved against source and retained artifacts, no training.

1. Point 1 ("keep both stochastic, not separating E and D"): Run A–E already
   share one schedule for both maps (`set_component_noise_multipliers` falls
   back to the shared pair); the only separation lived in the retired
   `abNarrowSplit` / `encoderDet` arms. Stage 1 is deterministic. Decision:
   keep the deterministic warmup, document the shared schedule, retire the
   split arm.
2. Point 2 ("train longer; broken => peak shifts"): the peak does shift at
   later checkpoints (Υ full-prior median 9.636 → 9.952 GeV, stage 2 → stage 4),
   but inference-time noise is not the cause (tail sweep, section 2 item 15).
   Longer training helps in-domain; the OOD scale is the failure.
3. Point 3 (asymmetric peaks): in-distribution asymmetry is target-inherited
   (Z low-mass tail, Υ radiative tail); model-created asymmetry appears only
   under OOD high-pT extrapolation. See section 2 item 17.
4. Point 4 (flat loss / variable coefficients): loss coefficients are already
   constant within a stage; only the noise multipliers and LR vary per epoch.
   Flat train loss is the distributional floor; `eval_loss` is not flat.
5. Point 5 (error accumulation): the ppzee residual is 95.5% detector-inherited
   and common-mode in log-pT, not independent per-component accumulation. New
   reproducible report at `outputs/cms_Joint/ppzee/error_decomposition/`.

Deliverables this session: `configs_joint/cms_Joint_runE_constNoise.yaml`
(prepared, dry-run passed, not trained); `scripts_joint/ppzee_error_decomposition.py`
plus its report; the shared-schedule contract test
`test_run_e_constant_noise_arm_shares_one_schedule_for_both_maps`; the two
`encoderDet` tests retired with a skip reason. No checkpoint, `data/` or
existing output was modified. Separately noted: the working tree already had
886 tracked deletions under `outputs/cms_Joint/` (mostly `Run_E/plots/` and
`Run_E/upsilon_transfer/quantitative_z_to_x/`) before this session.

### 2026-09-10 — Session 38, Run F flat-loss mechanism validation prepared

Implemented the mechanism fixes from the Session 37 diagnosis and prepared
Run F (see section 4). Quantified the diagnosis with
`scripts_joint/loss_floor_probe.py`: the weighted distributional floor is 0.452
for Run E and 0.239 for Run F; the J/ψ terms fall from ~0.34 to ~0.12. Ran
`--dry-run` and `--smoke` for Run F (both passed) but did **not** start
training. Preserved the original Run F prior contract as
`cms_Joint_runF_priorCKKWL.yaml`; added the `replace_keys` config directive so
Run F could replace, not merge, Run E's four stages. Full suite: 244 tests, OK
(4 skipped). No checkpoint or `data/` artifact modified.

### 2026-09-10 — Session 39, Run F reviewed, continuum diagnostic, checkpoint policy

Run F completed (100 epochs, 4x256). Selected model is stage-1 epoch 20
(deterministic, score 0.628); stage-2 best is epoch 45 (0.836); global last is
epoch 100 (2.013). Per-stage last checkpoints were added
(`_stage_last_path` + save in `run_joint_training`) because `best_model.pt` is
the global best and is allowed to be a pre-noise checkpoint, and because
`last_model.pt` is overwritten by the last stage.

Upsilon continuum diagnostic (section 4, Run F block): the 8.5-9.25 GeV excess
is prior-continuum shape for the deterministic model (1.76 -> 1.03 after a
sideband-derived reweight); the stochastic last model keeps a +27% model-created
residual from the 1S tail. Full suite: 245 tests, OK (4 skipped). No checkpoint
or `data/` artifact modified; new analysis files only.

Decoded and scored every Run F checkpoint on the Upsilon transfer, including
the previously missing stage-2 best (ep45) and the continuum-reweighted
best/last combinations. Combined table and overlay figure:
`outputs/cms_Joint/Run_F/upsilon_transfer_comparison/`. The continuum
correction dominates the inclusive mass improvement (best_model mass W1
0.307 -> 0.038 GeV; 8.5-9.25/CMS 1.756 -> 1.047); the deterministic checkpoint
keeps the 1S at 116 MeV while the stochastic ones broaden it to 193/285 MeV.
The common residual is a ~90 MeV downward 1S shift.

### 2026-09-10 — Session 40, Run G prepared

Run G prepared (section 4): stage list replaced with 20 + 60 + 60 and a new
stage 3 at core 1.0 / tail 0.5, network width halved to 4x128, everything else
inherited from Run F. Contract test `test_run_g_three_stage_contract`; full
suite 246 tests, OK (4 skipped); dry-run and smoke passed (per-stage best and
last checkpoints written for all three stages); not trained.

### 2026-09-10 — Session 41, Run G 4x128 confirmed, TF32 probed

Run G is 4x128: `hidden_dims [128]*4`, 218,696 parameters (dry-run), 140 epochs;
`Run_G_dryrun_4x128` and `Run_G_smoke_4x128` passed with per-stage best and last
checkpoints. Added `configs_joint/cms_Joint_runG_tf32.yaml` (Run G plus
`performance.tf32: true`). Measured the TF32 impact with
`scripts_joint/tf32_impact_probe.py` (section 2 item 18): ~1e-4-level relative
differences in losses/gradients, 1.30x on the matmul alone, but the pipeline
bottleneck is the SWD sorts, so the end-to-end gain is smaller. Run G stays
strict fp32 so the stage/width comparison with Run F is unconfounded. Full
suite: 246 tests, OK (4 skipped). No checkpoint or `data/` artifact modified.

### 2026-09-10 — Session 42, invariant-mass mask and anchor enforced

Verified that every joint config (A-G) masks `log_pair_mass` (feature index 8)
out of the 14-D condition vector (13-D mask) and that all mass anchors are zero;
Run E/F checkpoints carry the same 13-D mask. Turned this from an incidental
property into an enforced contract: `run_joint.assert_mass_not_conditioned`
(both maps) and `run_joint.assert_no_mass_anchor` (regions' merged loss, with a
declared `allow_mass_anchor: true` override) run in `main()` before training.
Added the two regression tests; full suite 248 tests, OK (4 skipped). Run G
dry-run passed through both guards. Note: the distributed mass terms
(`pair_mass_w1`, `mass_kin_swd`, `physics_swd`, the cycle) stay on; they are
targets, not anchors. Also noted: the current `cms_Joint_runG.yaml` on disk uses
`num_slices` 128 / 192 / 256 by stage (not 256 everywhere as first written).

### 2026-09-10 — Session 43, sliced conditional loss implemented (Run H)

Implemented the pT-sliced conditional mass W1 as a first-class loss component
(`SpaceFeatureOTLoss.sliced_mass_w1`, weight default 0 so every earlier config
is byte-for-byte unchanged). Ran the slice-resolved diagnostic on Run F
(section 2 items 20-21): the conditional mismatch is 3.5-17x the per-slice
floor while the global marginals are matched. Resolved the two collisions:
`mass_kin_swd` is replaced (0.5 -> 0) rather than stacked, and the floor budget
is held by 4 slices with min 512 events/slice (added floor ~0.025, ~17% of the
learnable excess; 5 slices rejected). Prepared Run H (4x256, 20/80/80,
num_slices 256/256/256). Full suite: 250 tests OK (4 skipped); dry-run and smoke
passed (sliced term exercised in training); not trained.

### 2026-09-11 - Session 44, Run H trained; core/tail audit and tail fix proposal

Run H completed (20/80/80, ep 180). Read-only audit of the stage noise
multipliers, the decoded Upsilon peaks and the learned noise scales. Verified
the schedule from `history.json`; measured the stage-3-only peak break and
showed it is in the deterministic mean map, not the sampled tail. Found the
learned tail sigma collapsed (8.1e-4 -> 1.8e-4) and the core sigma pinned to its
1e-3 floor, so the decoder injects ~10 MeV of Upsilon mass noise against an
84 MeV resolution and manufactures the width in the mean map. New tool
`scripts_joint/runH_tail_audit.py`; deliverables under
`outputs/cms_Joint/Run_H/tail_audit/` (REPORT.md, tail_audit.json, plot,
DIAGNOSIS_AND_FIX.md). Recommendation: use `best_RunH_stage2` (ep 90) for
Upsilon; fix the tail with LR decay plus a noise floor / mean-map anchor. No
checkpoint, data or existing output modified.

### 2026-09-11 - Session 45, drift control implemented (mean-map anchor contract)

Implemented the four drift-control levers as an opt-in, default-off change set.
Mean-map anchor (`scripts_joint/joint_anchor.py`) is enforced as a guarded
contract in `run_joint.py`, mirroring the invariant-mass guards, including the
rule that an anchored stage keeps both directions' noise on. Optimizer-state
carry and per-step weight EMA added to `joint_trainer.py`. Two arms prepared and
preflighted: `Run_H_anchor` (anchor only) and `Run_H_fix` (all levers). 255
tests OK (4 skipped); dry-run and smoke passed. No checkpoint, data or existing
output modified; no full run started. Open tuning: anchor weight and reference
stage (section 1.4).

### 2026-09-11 - Session 46, Run H fix audited (drift slowed, asymmetry persists)

Analyzed the completed `Run_H_fix` run and its Upsilon transfer and
slice-resolved diagnostics. The drift control worked in the sense that the
stage-3 in-domain collapse is ~3x slower and the ep-180 Upsilon medians now sit
on the CMS fit, but the peak shape is not fixed: the centred checkpoint
over-spreads with a heavy low-mass shoulder, the narrow checkpoints are 70-85
MeV low (the anchor preserved the stage-1 bias), and the J/psi conditional
asymmetry still degrades through the stochastic stages while Z is fixed.
Findings in sections 1.5 and 2 item 25. New diagnostics:
`outputs/cms_Joint/Run_H_fix/slice_diagnostic_{best,stage2,stage3,last}/`.
No checkpoint, data or existing output modified.

### 2026-09-23 - Session 47, meeting references resolved and mapped to the plan

**source-verified:** the four cited works are now all resolved. 2304.13586 =
"Energy-Based Sliced Wasserstein Distance" (Nguyen et al.). 2303.08797 =
"Stochastic Interpolants: A Unifying Framework for Flows and Diffusions"
(Albergo, Boffi, Vanden-Eijnden). The two 2026 links resolve from their titles
to unfolding papers, not generative-model papers:

- 2602.24282 = "Unfolding without Iterations, Adversaries, or Surrogates"
  (Ore & Plehn; v2 2026-08-22). AUSSIE: discriminative reweighting of a
  simulated reference. Step 1 trains a reco-level density-ratio classifier
  `R_theta(x) ~ p_data(x)/p_sim(x)`. Step 2 trains a latent network
  `Rbar_phi(z)` by minimising the RKHS norm of the functional gradient of an
  MLC loss; the stationary condition is the unfolding integral equation. It
  **uses paired (x, z) simulation events** (Eq. 12 of the paper). Two
  implementations: analytic Gaussian kernel (low-dim) and NTK via autograd
  (high-dim). MIT code: `github.com/heidelberg-hepml/aussie`.
- 2603.20903 = "Unfolding with a Wasserstein Loss" (Craig, Faktor, Nachman;
  v2 2026-04-22). Replaces the KL Richardson-Lucy loss with a p-Wasserstein
  loss for unfolding with a Markov-kernel noise model. Sharp existence
  (Thm 1) and uniqueness (Thm 2: uniqueness iff the injectivity condition
  (INJ) plus feasibility) results, and a provably convergent generalized
  Sinkhorn algorithm that needs only **empirical samples of the kernel** and
  scales with data size, not ambient dimension. Numerics are 1D/2D
  jet-mass-inspired; no code link found in the paper.

**hypothesis (mapping):** both papers address the encoder / M2 half (invert a
*given* forward or noise model K); neither addresses the decoder
mean-vs-noise identification failure. Craig et al.'s (INJ) condition is the
formal version of our identifiability requirement -- a collapsed decoder
(`rho_x ~ delta_{D(x)}`, resolution faked by the mean map) makes the inverse
ill-posed. So sections 1.5 / 2 items 23-25 remain the prerequisite. The
useful structural point for M2: our OT terms are *training losses for an
amortized map* and identify marginals only (item 6); AUSSIE's stationary
condition and Craig et al.'s OT plan solve the unfolding equation/coupling
itself, so the per-event object comes out of the solution, not the loss.

**proposal:** (a) run AUSSIE on ppzee (the only bench with paired simulation)
as the M2 per-event reference against our encoder (`residual_rms_vs_identity`
0.984 vs oracle 0.810); it has an information advantage there, so treat it as
a ceiling, not a like-for-like replacement. (b) Pilot the Craig generalized
Sinkhorn on ppzee with the known kernel, then on the joint model with the
frozen learned decoder as the kernel, for the J/psi mass coordinate first.
(c) Keep P1 -- cycle decoupled from the noise channel plus a fixed-z noise
budget -- ahead of both, because both invert K and our K is currently
degenerate.

**reported-not-reproduced (retained pre-Run-E record `memory.md.bak.20260904`,
git `bf9a59f`; not re-measured this session):** the modern generative branch
was already prototyped in the SOTA line. SOTA Run F (bidirectional minibatch-OT
flow matching / stochastic interpolant, 300 epochs) did not beat SOTA Run E on
generator mass closure (D(z) mass W1 0.00537 vs 0.00105 GeV) and had lower
fixed-z stochasticity (median 3.0 vs 9.2 MeV). SOTA Run G0 (reciprocal
stochastic bridge with conditional score head, best epoch 270) reports x->z
mass W1 0.00137 GeV and z->x 0.00188 GeV on its own locked evaluation. This is
relevant to the "substitute a more modern generative method" proposal: it was
tried and did not win on the marginal metrics.

No checkpoint, `data/` or existing output modified; no training started.

**proposal (executed this session):** `docs/project_tree.md` was rewritten in
place, superseding the 2026-09-07 tree (git `bf9a59f`). It now reviews the four
failure modes (F1 dead noise channel / mean-map resolution, F2 non-identified
mean/noise split, F3 marginal-only inverse, F4 honest-prior and OOD transfer)
plus hygiene, and organises the work into branches A (response identification;
A1 decouple-cycle is the decisive single-variable run), B (inverse: ppzee bench,
AUSSIE, OT unfolding), C (estimator/data hygiene), D (acceptance gates) and E
(retired/parked). The old M1/M2 step labels are retired in that file: M1 stays
falsified, M2 is replaced by the AUSSIE/OT-unfolding leaves.

### 2026-09-23 - Session 48, A0 fixed-z noise budget delivered

**source-verified:** new read-only diagnostic `scripts_joint/fixed_z_noise_budget.py`
and `tests/test_fixed_z_noise_budget.py` (8 tests). For one common fixed z sample
per region/state it decodes 32 times at the checkpoint's recorded (native)
multipliers, at a probe (1.0, 0.25), and at zero noise; it reports the per-event
within-z mass width, `sigma_noise_only = sqrt(E_z[Var(mass|z)])`, the exact
variance split `Var = E[within] + Var[mean map]`, and the learned sigma from
forward hooks. Checkpoints are opened read-only.

**artifact-measured:** artifacts in `outputs/cms_Joint/noise_budget/`
(`noise_budget.json`, `noise_budget_table.csv`, `REPORT.md`, `noise_budget.png`,
`noise_budget.pdf`, `run.log`). Matrix: 14 checkpoints x {jpsi, Z, Upsilon
1S/2S/3S}, 256 fixed z events per region/state, 32 draws, CUDA, git bf9a59f
(dirty).

- **Upsilon (the transfer target):** the 11 core-on stochastic checkpoints carry
  a within-z robust mass width of only **10.4-14.4 MeV (1S)** / 11.1-15.8 (2S) /
  11.2-15.9 (3S) against the 84 MeV CMS resolution, i.e. 12-19% of it. The
  noise-carried variance fraction is 0.0011-0.0144 (1S). The largest
  `sigma_noise_only` on 1S is 16.3 MeV, failing the 50 MeV G-A criterion by 3.1x.
- **Z:** within-z robust width 95-221 MeV against the ~2.5-2.9 GeV required
  (ratio 0.04-0.09); noise-carried variance fraction 0.0005-0.0026. The Z
  marginal is matched by the deterministic mean map, not the noise.
- **J/psi:** within-z robust width 20.5-25.2 MeV, noise-carried variance
  fraction 0.41-0.53. That is ~2x the 12.2 MeV additional smearing the
  pre-smeared prior needs and 0.73-0.90 of the physical 28.1 MeV resolution.
  The channel is live for J/psi and dead for Z and Upsilon.
- **Collapse signature:** the deterministic stage-1 checkpoints (sigma never
  trained) probe at 25.3-26.0 MeV (Upsilon 1S) and 245-257 MeV (Z) with
  multipliers 1.0/0.25; the Run H/F/G stochastic checkpoints probe at only
  10.6-13.6 MeV and 98.6-173.8 MeV. The stochastic stages shrank the learned
  sigma toward its floor rather than growing it. Run H last is tail-inert:
  native (1.0, 0.5) 10.71 MeV vs probe (1.0, 0.25) 10.73 MeV.
- **Zero-multiplier control:** every region records
  `zero_noise_draws_identical: true`; the ensemble-difference estimator
  `sqrt(sigma_native^2 - sigma_zero^2)` is noisy at 256 events, so the report
  quotes the exact within-z estimator and stores both.

**hypothesis:** this is F1/F2 measured per checkpoint. The marginal metrics
cannot see the missing resolution because the mean map absorbs it, and the
deficit is worst where the required absolute resolution is largest (Z, Upsilon).
A1 (cycle decoupled from the noise channel) plus A2 (calibrated/floored sigma)
remain the next experiments; A0 now supplies the gate metric.

No checkpoint, `data/` or existing output modified; no training started.

### 2026-09-23 - Session 49, non-training backlog delivered (no full run)

**source-verified:** everything below is code/config/tests/preflight. No full
training run was started; no checkpoint, `data/` file or existing output was
modified. Every item goes `--dry-run` and, where the training path changed,
`--smoke` on CPU.

**A0.3 - noise-aware validation metric.** `joint_metrics.noise_budget_metrics`
plus the opt-in `loaders.validation_noise_budget` block (draws/events caps and
explicit multipliers). It adds `noise_budget_*` keys to each region's
`region_validation` and a `validation_noise_budget` note to `joint_selection`,
and never enters the configured selection targets. Verified in the A1/A2 smokes:
stage 1 (0/0) gives a zero width; stage 2 (1.0/0.25) and stage 3 (1.0/0.5) give
the expected nonzero widths. Config-only native-noise scoring was already
available through `loaders.validation_noise_multipliers`.

**A1 - cycle decoupled from the noise.** `joint_trainer._decode_cycle` plus the
top-level `cycle_decoder_noise: zero` key zeroes only the decoder multipliers
around the `x_reco` forward and restores them; encoder noise and the
direct/latent terms are untouched. Config `cms_Joint_runH_cycleNoNoise.yaml`
(one controlled change versus Run H; the budget block is passive). Tests
`tests/test_a1_a2_a0_noise.py` (18). Dry-run and smoke passed on CPU; the smoke
resolved config carries `cycle_decoder_noise: zero`. The full 180-epoch run is
NOT started.

**A2 - per-map and tail sigma floors.** Optional `tail_sigma_floors` buffer
(absent by default, so every earlier checkpoint's state dict is unchanged) and
`encoder_noise_overrides` / `decoder_noise_overrides` in
`JointDimuonAutoencoder` (only the four noise keys are overridable).
`restore_joint_checkpoint` now tolerates a checkpoint missing only the floor
buffer. Config `cms_Joint_runH_noiseFloor.yaml`; dry-run and smoke passed. A
fresh A2 model applies encoder core floor 1e-4 and decoder tail floor
1.5e-3 / 1e-3.
- **artifact-measured caveat:** the required per-coordinate log-pT sigma spans
  ~0.001 (J/psi) to ~0.008 (Upsilon, 84 MeV) to ~0.03 (Z, 2.85 GeV). A single
  global floor cannot span that range; the shipped config is a mechanism demo
  and the real A2.2 choice is a condition-dependent floor or a frozen measured
  amplitude, which needs sign-off.
- Warm-start caveat: floor buffers travel with the checkpoint, so loading a
  pre-change checkpoint restores its floors over the new config.

**C4 - unified-prior component loading.** `scripts/cms_data.py` gained
`read_prior_component_arrays` (zData + component_id + weight),
`legacy_jpsi_effective_fraction`, `mix_prior_components` (component selection,
target fraction, `resample_if_ess_ge_0p5` with seed and stop threshold),
`load_theory_prior_z_components` and `resolve_prior_component_spec`;
`joint_data.region_data_config` projects the enabled per-region spec. New config
`cms_Joint_unifiedP1_components.yaml`; tests `tests/test_prior_components.py`
(20). Dry-run with `--num-samples 20000` passed (5,343,304 params, 432 epochs).
- **artifact-measured:** J/psi raw {jpsi 737,262 / continuum 999,089} ->
  post-selection {493,394 / 133,253} -> target fraction 0.850000 measured from
  the legacy file's FDL attributes (n_signal 80,648 / n_continuum 14,232; the
  legacy file has NO per-row component_id) -> output {532,650 / 93,997},
  realised 0.850000; ESS/N 0.9564 / 0.9218. Z: one component 697,180 ->
  502,148; ESS/N 0.9863.
- **Operational notes:** the base `cms_Joint_unifiedP1.yaml` already sets
  `prior_components.enabled: true`, so `--run unifiedP1` now loads
  component-aware; the existing `outputs/cms_Joint/unifiedP1` artifacts predate
  the loader and came from the inert union-of-components path. The cms_data
  pipeline fingerprint changed, so existing region caches are invalidated by
  design and the next dry-run/training re-reads ROOT/HDF5. The inherited
  top-level `selection:` (35/65 vs_identity score) and `metrics:`
  (`latent_mass_core_fraction`) blocks remain unimplemented and silently
  ignored - C4 covers `prior_components` only; that is a sign-off item.

**D4 - ppzee posterior calibration baseline.** New read-only
`scripts_joint/ppzee_posterior_predictions.py` plus
`tests/test_ppzee_posterior_predictions.py` (15); artifacts
`outputs/cms_Joint/ppzee/pull_coverage/{pull_coverage.json,REPORT.md}` (the
~1 GB npz intermediates were pruned after the metrics and are regenerable with
`--draws 32`).
- **artifact-measured (160k pairs, 32 draws):** `residual_rms_vs_identity`
  0.9808-0.9830 against the 0.8099 oracle; pull std 12.6 (untrained sigma) to
  20.7 (trained) against nominal 1; 1-sigma coverage 0.058-0.103 against
  nominal 0.6827; pull mean ~ -1. The deterministic stage-1 encoder is a
  zero-variance degenerate branch.
- **hypothesis/conclusion:** the encoder is not a calibrated posterior, and the
  joint-trained sigma collapse makes it worse. Any M2 method must report pull
  and coverage, not only the residual. Caveat: ppzee truth has zero pair pT
  (LO 2->1), so it is a valid per-event bench but not a realistic response
  geometry.

**C1 - sliced-Wasserstein estimator probe (non-training).** New
`scripts_joint/swd_variant_probe.py` plus `tests/test_swd_variant_probe.py`
(16); artifacts `outputs/cms_Joint/swd_variant_probe/{swd_variant_probe.json,
swd_variant_probe.csv,REPORT.md,*.png,*.pdf}`. Bounded synthetic probe
(8 replicates, N=1024, 8-D four-vector space, p=1, 100 adversarial steps;
direction ascent only inside the probe; independent truth/detector batches
matching the unpaired loss).
- **artifact-measured, resonance scenario** (3.0969 GeV, 5 MeV truth width,
  1% log-pT smearing, 23.5 MeV detector mass width): every estimator's
  signal/floor ratio is 0.80-0.90 - in this synthetic setup the resonance
  signal sits *below* its own finite-sample floor, the same floor-dominance
  measured on the real joint loss (`loss_floor_probe`, 0.452 -> 0.239). A
  learned direction set gives no gain here (max-SWD 0.796 in-sample / 0.889
  held-out vs random 0.882 at 256 slices). A +50 MeV shift has power 0.25-0.50
  at 8 replicates (binomial uncertainty ~0.17).
- **artifact-measured, hard-direction scenario**: random SWD signal/floor
  2.57-2.77; learned max-SWD 5.47-8.39 (3x, held-out directions), with mean
  |cos| alignment 0.96 for 8 directions and 0.75 for 32. Learned directions
  trade a higher floor for a larger signal; the useful comparison is the ratio
  and paired detection power, not the raw distance.
- **hypothesis:** max-SWD is a targeted win for directionally concentrated
  mismatches, not a blanket replacement of random-slice SWD for diffuse
  resolution-like smearing. EBSW and an explicit control-variate estimator
  were NOT implemented (proposal, not a claim). No training A/B was run.

**Still open, and NOT quick non-training jobs:** the full A1/A2 runs; AUSSIE
(B1) port plus classifier/unfolder training; the B2 generalized-Sinkhorn
implementation; A5 flexible residual density; the unifiedP1 `selection` /
`metrics` blocks.

### 2026-09-23 - Session 50, new rule: preflight directories are transient

**proposal, adopted by the user this session:** after a `--dry-run` or
`--smoke` passes - with the focused or full test suite green - delete its output
directory (`*_dryrun*` / `*_smoke*`) before starting the next piece of work.
Smoke checkpoints are not evidence; the session log is.

**source-verified:** the rule is in `CLAUDE.md` section 3; the helper is
`scripts_joint/clean_preflight.py` (lists by default, `--apply` deletes; only
directories whose name carries a dryrun/smoke token and strictly inside the
root; symlinks skipped) with `tests/test_clean_preflight.py` (9 tests).
Command: `python scripts_joint/clean_preflight.py --root outputs --apply`.

**artifact-measured:** applied this session; removed 13 directories, 8426.2 KiB
total.
- `cms_Joint`: `AB_narrow_split_dryrun`, `Run_G_guardcheck_dryrun`,
  `Run_H_anchor_smoke`, `Run_H_cycleNoNoise_{dryrun,smoke}`,
  `Run_H_noiseFloor_{dryrun,smoke}`, `unifiedP1_components_dryrun`.
- `cms_Jpsi_sota`: `runG0_stable_dryrun`, `runG0_stable_smoke`,
  `runG0_stable_smoke2`, `runG0_stable_smoke3`.
- `cms_JpsiDoubleMuons/archive`: `ab_plots_smoke`.

**note:** earlier sections reference preflight dirs (`Run_F_dryrun`,
`Run_H_dryrun`, `Run_E_constNoise_dryrun`, ...); under this rule they are
transient and are no longer on disk. Regenerate them with the documented
dry-run/smoke command if a specific check is ever needed. The durable artifacts
of Sessions 47-49 (`noise_budget`, `swd_variant_probe`, `pull_coverage`, the
trained run directories) are not preflight dirs and were kept.

### 2026-09-23 - Session 51, A1 full run complete: kill condition triggered

**source-verified:** `outputs/cms_Joint/Run_H_cycleNoNoise/` finished the
180-epoch 20/80/80 schedule (all 180 history rows: stage 1 0/0, stage 2
1.0/0.25, stage 3 1.0/0.5). The only change versus Run H is
`cycle_decoder_noise: zero`.

**artifact-measured, in-domain:** stage bests ep15 0.696, ep35 0.789, ep105
0.937 against Run H's 0.687 / 0.839 / 2.017. `best_model.pt` is still stage 1
(ep15). The stage-2 best 0.789 is at or below the 0.839 acceptance value and the
stage-3 collapse is much milder than Run H's.

**artifact-measured, noise budget** (`Run_H_cycleNoNoise/noise_budget_audit/`,
256 fixed z, 32 draws; A0 baseline in parentheses):
- Upsilon 1S within-z robust 10.9 / 10.7 / 10.1 MeV for stage2best / stage3best
  / last versus 84 MeV required (ratio 0.12-0.13); `sigma_within` 12.2 / 12.1 /
  55.7 MeV. **The kill rule (`sigma_noise_only < 20 MeV`) is triggered** for the
  peak channel. The last checkpoint's mean is tail-dominated: robust 10.1 MeV
  but mean 55.7 MeV.
- Z within-z robust 157 / 132 / 97 MeV (Run H 98 / 95 / 96): 1.4-1.6x larger at
  the stage bests, still only 4-6% of the ~2.5 GeV required.
- J/psi within-z robust 22.5 / 23.5 / 21.6 MeV (Run H 21.4 / 22.1 / 21.1);
  noise fraction 0.42 / 0.40 / 0.41 - unchanged.

**artifact-measured, Upsilon transfer** (`upsilon_peak_medians/`, 400k fixed
subset, median minus CMS fit, 1S/2S/3S in MeV): A1 stage1 -35/-48/-44,
stage2best -36/-50/-46, stage3best -34/-48/-45, last -29/-45/-41. The tool
reproduces Run H stage2best at -14/-26/-20 (audit: -14/-25/-19). Best-vs-last
drift is 4-7 MeV (<40 MeV pass), but the absolute medians are outside +/-30 MeV
and are inherited from the stage-1 map.

**artifact-measured, slice diagnostic** (`slice_diagnostic_stage{2,3}_best/`,
noise 0): J/psi per-slice W1/floor 19.0/2.9/8.2/5.7 (stage2best) and
19.7/5.5/13.5/7.1 (stage3best) - still 3-20x the floor, no improvement over Run
H; Z sits at 1.7-4.6x and near the floor.

**hypothesis:** the cycle penalty was not the binding constraint. Removing it
improves in-domain stability and prevents the ep-180 OOD blow-up (Run H last
-125/-136/-136 -> A1 last -29/-45/-41) and lets the Z channel grow modestly, but
the marginal objective is still indifferent to the mean/noise split, so the
optimizer keeps the resolution in the mean map. A2.2/A2.3 (a condition-dependent
floor or a frozen measured amplitude) or an architectural change is required for
identification; the cycle decoupling should be kept as a stability measure.

**new tool:** `scripts_joint/upsilon_peak_medians.py` (read-only, arbitrary
checkpoints, per-state medians vs the CMS fit, with pairwise drift).

**artifact-measured, Upsilon decode + combined overlay (Session 51b):** full
1M-event raw-prior decodes with plots under
`Run_H_cycleNoNoise/upsilon_transfer_best_RunHcycleNoNoise_stage2_stochastic_core/`
and `.../upsilon_transfer_last_model/` (19 plots each), and a combined overlay in
`.../upsilon_transfer_comparison/` (`scripts_joint/upsilon_transfer_comparison.py`
gained `--run-dir` / `--sample LABEL=TRANSFER_DIR`). Raw prior
8.5-9.25/CMS = 1.724.

| sample | mass W1 [GeV] | mass KS | pair-pT KS | C2ST | decoded mean [GeV] | 1S mean [GeV] | 1S std [MeV] | 8.5-9.25/CMS |
|---|---|---|---|---|---|---|---|---|
| Run H stage2-last (ep100) | 0.0252 | 0.0234 | 0.321 | 0.788 | 10.042 | 9.348 | 283 | 1.175 |
| Run H last (ep180) | 0.0776 | 0.0724 | 0.296 | 0.796 | 9.989 | 9.291 | 302 | 1.318 |
| A1 stage2-best (ep35) | 0.3019 | 0.1707 | 0.334 | 0.808 | 9.765 | 9.344 | 170 | 1.864 |
| A1 last (ep180) | 0.2926 | 0.1509 | 0.318 | 0.809 | 9.774 | 9.329 | 294 | 1.874 |

**hypothesis:** A1 traded the peak-scale blow-up for a worse low-mass response.
Its 1S peak position is comparable to Run H's stage 2 (9.344/9.329 vs 9.348) and
far from Run H last's 9.291, but the inclusive marginal is much worse
(W1 0.30 vs 0.025-0.078) because it leaves or amplifies the prior's low-mass
continuum excess (1.86-1.87 versus the prior's 1.72 and Run H's 1.18-1.32). The
next branch must therefore address the continuum/conditional response, not only
the peak scale or the noise amplitude.

**artifact-measured, paper-style in-domain plots:** 22 plots per checkpoint for
`best_model` (ep15), `stage2_best` (ep35), `stage3_best` (ep105) and
`last_model` (ep180) under `Run_H_cycleNoNoise/paperstyle_*/`.

**artifact-measured, reweighted-prior transfer (Session 51c) - qualifies 51b.**
The A1 and Run F decodes were repeated on the sideband-derived evaluation prior
`data/upsilon_prior_continuumReweighted.hdf5` (`upsilon_transfer_test.py
--prior ...`), with a combined overlay whose reference curve is that same prior
(`upsilon_transfer_comparison.py` gained `--prior` and
`--sample LABEL=TRANSFER_DIR[::DECODED_PATH]`):

| sample | mass W1 [GeV] | mass KS | pair-pT KS | C2ST | decoded mean [GeV] | 1S mean [GeV] | 1S std [MeV] | 8.5-9.25/CMS |
|---|---|---|---|---|---|---|---|---|
| Run F best + reweight | 0.0376 | 0.0833 | 0.3246 | 0.793 | 10.0315 | 9.3545 | 116.4 | 1.047 |
| Run F last + reweight | 0.0801 | 0.0548 | 0.3137 | 0.795 | 9.9869 | 9.2793 | 284.8 | 1.269 |
| A1 stage2-best + reweight | 0.0417 | 0.0499 | 0.3304 | 0.789 | 10.0252 | 9.3443 | 169.5 | 1.154 |
| A1 stage3-best + reweight | 0.0398 | 0.0383 | 0.3252 | 0.789 | 10.0271 | 9.3327 | 194.7 | 1.185 |
| A1 last + reweight | 0.0445 | 0.0341 | 0.3162 | 0.792 | 10.0225 | 9.3290 | 292.4 | 1.196 |

- **Correction to 51b:** the alarming raw-prior W1 0.30 was dominated by the
  raw prior's own continuum shape (raw prior 8.5-9.25/CMS = 1.724; A1 raw
  1.86). On the reweighted evaluation prior A1's mass W1 is 0.040-0.045 and KS
  0.034-0.050 - competitive with Run F best (0.038/0.083) and better on KS.
- The model-created low-mass residual is +15-20% (1.15-1.20) versus Run F best
  1.05, and the remaining decisive deficit is the **1S width**: A1 170/195/292
  MeV versus Run F best 116 MeV and the 84 MeV CMS resolution. The mean map
  still over-spreads the peak because the noise channel is too weak - F1 again,
  now visible in the transfer width.
- **proposal:** standardize the transfer evaluation prior across arms and state
  it explicitly; use the reweighted prior as the primary readout with the raw
  prior as a secondary. Run H's own reweighted decode is not computed yet.

### 2026-09-24 - Session 52, A2 preflight complete (no full run)

**source-verified:** the A2 chain is trainable; nothing beyond the standard
smoke was trained, and no checkpoint/data/existing output was modified.

**A2.0 - resolution calibration (read-only).** New
`scripts_joint/resolution_kernel_calibration.py` +
`tests/test_resolution_kernel_calibration.py` (22); artifacts
`outputs/cms_Joint/resolution_kernel/{kernel_spec.json,
kernel_spec_recommended.json,REPORT.md,resolution_vs_pt.png/pdf}`.
- **artifact-measured:** the requested `sqrt(a^2+(b/pT)^2)` form cannot
  represent the data - the per-muon log-pT amplitude *increases* with pT
  (J/psi cells 0.0106-0.0175 -> 23-38 MeV; Z cells 0.027-0.055 -> 1.3-3.6 GeV),
  so all six eta-bin fits collapse to `b~0` (`adequate: false`). Recommended
  power law `c(eta)*(pT/10)^alpha` (c 0.0107-0.0166, alpha 0.30-0.63); the
  Upsilon(1S) closure prediction is 65.9 MeV versus the 84 MeV reference
  (ratio 0.78), with the A0-empirical cross-check at 0.008 sigma_logpT.

**A2.3 mechanism.** `CylindricalFlowStep` now accepts a physics kernel spec
(`core_sigma_floor_spec`) in three forms (a/b, power law, linear), a
`freeze_noise_amplitude` mode (sigma = kernel) or a floor mode
(sigma = max(learned, kernel)), and optional `sigma_cap` / `pt_max_gev` clamps.
All are plain attributes, so old checkpoints keep an unchanged state dict.
`tests/test_a1_a2_a0_noise.py` grew to 30 tests.

**Model-effective kernel fit.** New `scripts_joint/kernel_powerlaw_fit.py`
(read-only decode probe). The **linear** basis `offset(eta)+slope(eta)*pT`
fits best: offset 0.00595-0.00917, slope 3.9e-4-6.0e-4 per GeV; validation at
scale 1: J/psi 1.21 (physical 28.1 MeV), Upsilon(1S) 0.86 (84 MeV), Upsilon
2S/3S 0.93/0.99, Z 0.83 (2.50 GeV). The power-law basis left J/psi 39% high;
the linear form is the shipped kernel.

**Bug found and fixed (source-verified).** `physics.invariant_mass_torch` used
unclamped `sqrt` on the muon pT, on the stable `mass2`, and the kernel's zeroed
phi/eta channels; a decoded near-collinear or zero-pT dimuon produced `inf`
from `SqrtBackward0` and NaN grads in the encoder. All sites now clamp with a
positive floor; regression test
`test_torch_degenerate_pairs_have_finite_gradients`. This is why the first A2
smokes failed and the A1/Run-H smokes had not hit it.

**Configs and preflight.** `cms_Joint_runH_A2frozen.yaml` and
`cms_Joint_runH_A2floor.yaml`: A1 cycle fix + A0.3 budget logging + honest
component-aware priors (C4) + the fitted kernel. Dry-runs passed on the C4
cache; CPU smokes (3 stages, tiny model, 20k events) passed. Smoke budget:
stage 1 (0/0) gives within-z 0 and noise fraction 0; stage 2/3 (1.0/0.25,
1.0/0.5) give frozen J/psi ~24 MeV and Z ~1.64 GeV, floor J/psi ~30 MeV and
Z ~1.64 GeV, with noise fractions 0.69-0.79 (J/psi) and 0.12-0.13 (Z) versus
~0.001 for Z in A1. The resolution is finally in the noise channel; these are
the tiny smoke numbers, not the physics result.

**Plan-tree consequence:** the honest J/psi prior is now a *prerequisite* for a
physical kernel - on the legacy hand-smeared prior (26 MeV) the kernel would
over-smear J/psi to ~43 MeV. The A2 arms therefore train on the unified
component-aware priors (C4); the change set versus Run H is documented as
(A1 cycle fix) + (honest priors) + (kernel).

**Next (training decisions, NOT started):** run `H_A2frozen` (primary) and
`H_A2floor` (floor + learned correction), then score G-A, the A0.3 noise budget,
the reweighted-prior transfer (1S width, low-mass ratio) and in-domain closure.
Full suite after the preflight: 383 tests, OK (4 skipped); 388 after the
close-out guards below.

**Close-out (same session).** The plan tree was updated: A2's leaf now carries
the 2026-09-24 preflight status (A2.0 kernel result, model-effective linear fit,
"honest prior is a prerequisite", both configs, the `invariant_mass_torch`
fix), the ordering row 5 records "preflight done 2026-09-24; full run open"
with A2frozen as primary, and section 8 maps the A2 scripts/artifacts. The four
green preflight directories (`Run_H_A2frozen_{dryrun,smoke}`,
`Run_H_A2floor_{dryrun,smoke}`, 3.31 MiB) were removed under the CLAUDE.md
section 3 cleanup rule, leaving no `*dryrun*`/`*smoke*` output behind. The
retained A2 preflight record is `outputs/cms_Joint/resolution_kernel/` plus
`outputs/cms_Joint/Run_H_cycleNoNoise/kernel_*`. `H_A2frozen`/`H_A2floor` were
deliberately **not** launched: the next step is a serious multi-hour training
slot.

Five more tests were added so the launch state is locked, not just verified
once: `tests/test_a1_a2_a0_noise.py::ShippedA2ConfigTests` resolves both A2
configs through `load_config`/`resolve_joint_config`, pins the fitted
coefficients, tail ratio, cap and `pt_max_gev`, checks that the floor arm
differs from the frozen arm **only** in the freeze flag, verifies the kernel
values at four (eta, pT) points including the 200 GeV clamp, and shows
behaviourally that frozen mode ignores the learned sigma (ratio 1.0) while the
floor arm lets it grow. Full suite after the close-out: **388 tests, OK (4
skipped)** (was 383 + 4 before the guard). Because the smoke directories are
gone, the launch commands in `docs/project_tree.md` section 8 start from a
clean `outputs/cms_Joint/`.

### 2026-09-25 - Session 53, H_A2frozen stage-2 crash diagnosed, fixed, run resumed

**artifact-measured (crash).** The first `H_A2frozen` full run (started
2026-09-25 11:49 on the RTX 4080 Laptop) died in stage 2 at local epoch 34
(global 54) with `RuntimeError: The total norm of order 2.0 for gradients from
'parameters' is non-finite`, raised by `torch.nn.utils.clip_grad_norm_`
(`error_if_nonfinite=True`) in `joint_trainer.train_joint_epoch`. The last
persisted state is `last_model.pt` = stage 2, global epoch 50 (stage-best score
3.81827, set at global 35); `history.json` holds 53 rows. Per-epoch `grad_norm`
over the whole run is 200-550 in both stages, so this is a single-update
outlier, not a divergence.

**Root cause (source-verified + reproduced).**
`scripts_sota/ot.py::cylindrical_physics_features(mass_from_energy=True)` - the
mass column of the DECODER's condition
(`joint_model.JointResidualFlowMap.condition`) - computed
`mass = torch.sqrt(torch.clamp(mass2, min=0.0))`. The naive float32
`pair_E**2 - |pair_p|**2` difference cancels for a boosted, nearly collinear
pair, so `mass2` can land on exactly 0; `SqrtBackward0` is then infinite and
NaN-poisons the whole backward pass. This is the same trap session 52 fixed in
`physics.invariant_mass_torch`; that fix missed this inline branch, and the
decoder's condition reaches it from both `decode(z)` and the cycle
`decode(encode(x))`. Debug probe `debug_nan/stress_probe.py` (real
`train_joint_epoch` updates from the epoch-50 checkpoint, amplified noise,
`torch.autograd.set_detect_anomaly`) reproduces it with the exact forward
traceback: `train_joint_epoch` -> `_decode_cycle`/`model.decode(z)` ->
`condition(current)` -> ot.py `SqrtBackward0`. At 1x noise the trigger is rare -
re-running the same deterministic epoch-54 batches from `last_model.pt` did not
hit it in 6 epochs - which is why the run survived 53 epochs.

**Fix (source-verified).** Positive floor
`mass = torch.sqrt(torch.clamp(mass2, min=0.25 * eps**2))`.
`sqrt(0.25*eps**2) == eps/2` sits below the `log(torch.clamp(mass, min=eps))`
floor that immediately follows, so every feature VALUE is unchanged while the
sqrt derivative becomes finite and the downstream clamp then zeroes it. Two
regression tests: `test_cylindrical_features_degenerate_mass_has_finite_gradients`
and `test_cylindrical_mass_feature_value_unchanged_by_the_floor`
(`tests/test_phase0_correctness.py`).

**Guard (source-verified).** `train_joint_epoch` no longer ends the run on a
non-finite gradient: it drops that update (`optimizer.zero_grad`, no
`optimizer.step()`) and counts it. Both shapes are covered - a non-finite entry,
and every entry finite but so large that the float32 total norm overflows
(`clip_grad_norm_` reports that as non-finite too, and
`error_if_nonfinite=False` would write NaN into the weights). Per epoch the
history row now carries `nonfinite_gradient_skips`, `optimizer_updates`
(applied) and `optimizer_steps_attempted`, so a systematic recurrence is
visible instead of silent. Test:
`tests/test_joint_runa.py::test_non_finite_gradient_skips_update_without_crashing`.

**Resumed (artifact-measured).** `python scripts_joint/run_joint.py --run
H_A2frozen --device cuda --resume`, relaunched 2026-09-25 13:36 from
`last_model.pt` (stage 2 local 30 -> resumes at local 31 = global 51; the
stage-best score 3.81827 is restored for selection). Console log:
`logs/Run_H_A2frozen_resume.log`. Remaining schedule: stage 2 (50 epochs) +
stage 3 (80 epochs) = 130 epochs at ~86 s (~3.1 h). Epoch 31 logged
`train_loss=1.56495` with zero skips.

**Stress evidence (artifact-measured, diagnostic only - not the physics run).**
With the fix, 3x noise completes with skips=0; 10x/30x drop 1-2 of 5 updates
through the guard; 100x-3000x (production-like, no anomaly detection) complete
with skips and no crash. This bounds the guard, not the physics: at 1x the
observed grad norms are 200-550 and the expected skip count is zero.

### 2026-09-27 - Session 54, H_A2frozen scored; honest-prior latent gauge is the blocker; A2.4 launched

**artifact-measured (training).** `outputs/cms_Joint/Run_H_A2frozen/` finished
the 20/80/80 schedule: 180 `history.json` rows, stage bests 4.214 (ep 20) /
3.818 (ep 35) / 3.798 (ep 110) at zero-noise selection; `best_model.pt` =
global 110. `nonfinite_gradient_skips` = 0 over the whole run (the Session-53
guard never fired).

**artifact-measured (A0 noise budget, new).** New directory
`outputs/cms_Joint/Run_H_A2frozen/noise_budget/` (4 A2frozen checkpoints + A1
and Run H stage-2 best as secondary rows; 256 fixed z per region, 32 draws;
reference checkpoint = `Run_H_A2frozen/best_model.pt`, so the J/psi and Z
samples use the honest component priors). Native within-z robust widths:
- Upsilon 1S: 86.5-91.6 MeV (stage 2/3/last) vs 84 MeV required, ratio
  1.03-1.09; 2S 93.9-96.7, 3S 95.3-98.4; `sigma_noise_only_from_within`
  91.3-100.9 MeV. **G-A width and `sigma_noise_only >= 50 MeV` criteria pass.**
- Z: 2.12-2.18 GeV vs 2.54 required (ratio 0.77-0.86); J/psi 31.1-34.4 MeV
  (1.23-1.34x the 28.1 physical, 1.03-1.14x the 30.0 robust requirement).
- Same reference sample, A1 / Run H stage-2 best: Upsilon 1S 11.0 / 10.5 MeV.
- **Report-verdict bug fixed** in `scripts_joint/fixed_z_noise_budget.py`: the
  G-A line hardcoded "below the 50 MeV criterion ... Every checkpoint fails"
  and the per-region lines hardcoded "the resolution sits in the deterministic
  mean map". Verdicts are now data-driven; the regenerated report states the
  pass.

**artifact-measured (in-domain at native noise, new tools).**
`run_joint.py` gained `--evaluate-only` (labeled re-score of a checkpoint at an
explicit noise policy; never overwrites `joint_evaluation.json`), and
`scripts_joint/rescore_validation.py` re-runs `joint_trainer.validate_joint`
with an explicit `validation_noise_multipliers` policy. Selection score
(worst region, configured targets):
- A2frozen stage2 best: zero 3.818, native 6.099; stage3 best: zero 3.798,
  native 5.374; last: zero 7.575, native 5.934. Worst metric is always
  `latent_mass_ks` (J/psi encoder).
- Same tool on the legacy-prior runs: Run H stage2 best zero 0.839 / native
  2.782; A1 stage2 best zero 0.790 / native 3.594. So A2frozen is ~4.5x the
  legacy in-domain score at zero noise, and the gap is not a scoring artifact.
- On the honest prior the J/psi encoder is nearly a no-op on mass: latent
  `mass_ks` 0.372 (stage3, zero noise) vs identity 0.385, floor 0.019, target
  0.04. That is the D3 honest-prior failure measured inside the A2 arm.
- Native-noise evaluation (`joint_evaluation_native_*.json`): J/psi z->x std
  40.8-46.7 MeV vs 28.1 data (over-spread), Z z->x std 5.34-5.40 vs 5.42 (right
  width) but shape W1 0.39-0.45 vs A1 0.135.

**artifact-measured (Upsilon transfer, reweighted evaluation prior, 1M events).**
Directories `upsilon_transfer_continuumReweighted_{stage2_best,stage3_best,last}`
plus `upsilon_peak_medians/` (400k subset). Per-state medians minus CMS fit:
- stage1 +8.3 / +1.3 / +3.7 MeV; stage2 -147.7 / -169.1 / -172.4;
  stage3 +21.6 / +2.9 / +6.3; last +18.9 / +6.9 / +13.3.
- mass W1 0.0727 / 0.0407 / 0.0260; mass KS 0.0760 / 0.0295 / 0.0267;
  pair-pT KS 0.325-0.329; C2ST MLP 0.787-0.807.
- **D2 criterion 1 passes for stage1/stage3/last** (all three medians within
  +/-30 MeV, best-vs-last drift ~1-3 MeV). stage2-best is an outlier on 2S/3S
  (-169/-172 MeV) while its 1S passes. stage3/last mass W1 beats A1
  stage2-best (0.0417) and Run F best (0.0376).
- Decoded 1S std is 290-363 MeV (A1 169-195, Run F best 116): peak positions
  are right, the peaks are over-spread.

**artifact-measured (slice-resolved, native 1.0/0.5).**
`slice_diagnostic_native_stage3_best/` (the tool gained `--noise-core/--noise-tail`):
J/psi per-slice decode mass W1 / floor = 36-87x (A1 zero-noise 3-19x), Z =
6.4-8.8x. The conditional response regressed with the frozen noise.

**hypothesis (the blocker).** A2frozen inherits `cycle_decoder_noise: zero`
(A1) while the decoder amplitude is frozen. The noise-free cycle can be
satisfied by a deterministic autoencoder, so the encoder keeps the detector
resolution in z (latent gauge ~ identity) and the mean map spreads to
reconstruct x; the marginal then adds the frozen kernel on top, over-spreading
the direct response (J/psi native std 46.7 vs 28.1 data) and leaving the Z
shape wrong. The A1 zero-noise cycle was a noise-collapse fix for a *learned*
sigma; with a frozen sigma it instead blocks identification.

**proposal + action, 2026-09-27.** New one-key arm
`configs_joint/cms_Joint_runH_A2cycleNoise.yaml` (A2.4):
`cycle_decoder_noise: native`, everything else identical to A2frozen (contract
test `test_cycle_noise_config_is_a_one_key_change_from_the_frozen_arm` pins
it). The cycle then sees the same frozen kernel the decoder uses - a denoising
cycle whose optimum is the unfolding encoder. Dry-run and smoke green
(09:46); full run launched, log `logs/Run_H_A2cycleNoise.log`.

**artifact-measured (A2.4 refuted, same day).** The cycle term is visibly
larger (`jpsi_x_loss` 0.234-0.274 vs 0.146 in A2frozen at the same epochs, so
the change did reach training), but the encoder did not unfold: stage-2
`latent_mass_ks` 0.483 (g25) / 0.410 (g30) / 0.464 (g35) / 0.736 (g40) versus
A2frozen 0.462 / 0.445 / 0.401 / 0.360. Cycle noise is not the binding
constraint on the honest-prior latent gauge. The run was stopped at global
epoch 40; `outputs/cms_Joint/Run_H_A2cycleNoise/` keeps the partial history
and checkpoints as the record.

**artifact-measured (A2floor, 180 epochs; stage bests 3.935 / 3.810 / 4.167 at
g20/40/140).** The declared A2.2 control (kernel floor + learned correction).
- Noise budget (`Run_H_A2floor/noise_budget/`): the mechanism holds - native
  within-z robust Upsilon 1S 85.3-90.7 MeV vs 84 (ratio 1.02-1.08), 2S
  90.3-97.1, 3S 93.9-99.1; `sigma_noise_only_from_within` 90.8-94.2 MeV;
  J/psi 31.6-38.6 MeV; Z 2.03-2.22 GeV. G-A's two mechanism criteria pass.
- The learned correction above the floor is **rejected**: in-domain score
  stage3-best 4.167 zero / 6.464 native (worse than A2frozen's 3.798/5.374);
  Upsilon medians +55.6/+39.5/+44.2 (stage3-best) and +84.8/+69.5/+75.6
  (last; best-vs-last 1S drift +97 MeV) versus A2frozen's +21.6/+2.9/+6.3 and
  +18.9/+6.9/+13.3; decoded 1S std 277-486 MeV; J/psi slice ratios 56-145x
  floor. More learned noise without closure: the frozen arm stays the A2
  deliverable.
- `best_model.pt` for A2floor is the stage-2 best (g40, score 3.810), whose
  Upsilon medians are -12.3/-40.2/-37.9 MeV - another instance of the
  zero-noise selection picking a poor transfer checkpoint (A0.3).

**A2 verdict (2026-09-27).** A2.1/A2.3 delivered (floors + frozen amplitude =
resolution in the noise channel, Upsilon transfer medians passing for the
frozen arm's stage3-best/last, mass W1 0.026-0.041). A2.2 rejected, A2.4
refuted. The A2 leaf's remaining gate item is the honest-prior latent gauge
(D3): the encoder is a no-op on J/psi mass (latent KS 0.372 vs identity 0.385,
target 0.04). **artifact-measured bound:** on the locked honest J/psi split, the
best *mass-only contraction* of x reaches latent KS 0.279 (identity 0.369),
while the target 0.04 needs a true unfolding - so G-A's "in-domain score <=
0.839" cannot pass until D3 does. J/psi direct over-spread (native std 40.8-46.7
vs 28.1 MeV) and the Z shape (native W1 0.39-0.45 vs A1 0.135) remain open
after D3. Full tables: `outputs/cms_Joint/Run_H_A2frozen/A2_VERDICT.md`;
plan-tree A2 leaf and G-A updated.

**Tools added this session (source-verified).** `run_joint.py --evaluate-only`
(labeled checkpoint re-score at an explicit noise policy; evaluate-only never
rewrites provenance/manifest/config), `scripts_joint/rescore_validation.py`
(re-runs `validate_joint` at native/zero noise and reproduces the training
selection score exactly), `fixed_z_noise_budget.py` data-driven G-A verdicts
(the old report hardcoded a failure), `slice_resolved_diagnostic.py
--noise-core/--noise-tail`, and `run_joint.py --cycle-weight-scale` (per-stage
cycle-weight multiplier for D3 probes). Tests: `test_rescore_validation.py`
(4), `test_cycle_noise_config_is_a_one_key_change_from_the_frozen_arm`,
`test_cycle_weight_scale_multiplies_every_enabled_stage`. Full suite 397, OK
(4 skipped).

**proposal (next run).** D3-focused, not another A2 variant: a z-space
consistency cycle `encode(decode(z)) ~ z` (the natural inverse condition, not
currently in the loss), or a heavier per-event weight with cycle noise
consistent with the frozen kernel (probe with `--cycle-weight-scale 5
--epochs 20` and compare the first 40 globals against A2frozen).

### 2026-09-29 - Session 55, A0.3 selection policy refuted (8/8 arms); Run_H_anchor attribution closed

**artifact-measured (A0.3 selection study, new tool).** New read-only driver
`scripts_joint/a03_selection_study.py` re-runs the trainer's own validation path
(`rescore_validation.py --noise native --compare-zero`, CPU) over every retained
checkpoint of nine response arms; artifacts
`outputs/cms_Joint/a03_selection_study/` (`a03_selection_summary.csv`,
`REPORT.md`, 58 per-checkpoint JSONs). Nothing trained; new files only.

Headline: **native-noise scoring does not let stochastic checkpoints win - it
selects the deterministic stage-1 warmup in 8 of 8 scorable arms.** A
deterministic checkpoint's recorded multipliers are 0/0, so its native score
equals its zero score; a stochastic checkpoint scores *worse* at its own
operating point, because injecting its noise degrades the decoded-x marginal
that the selection score measures. Zero -> native examples: A2frozen stage2 best
3.818 -> 6.047, stage3 best 3.797 -> 5.464; A2floor stage2 best 3.809 -> 6.988;
Run_F stage2 best 0.836 -> 3.345; Run_H stage2 best 0.838 -> 2.921; A1 stage2
best 0.789 -> 3.554; H_fix stage2 best 0.967 -> 4.427; Run_G_tf32 stage2 best
0.667 -> 3.513.

| arm | zero-noise pick (score) | native-noise pick (score) | same? |
|---|---|---|---|
| Run_F | g20 det (0.629) | g20 det (0.629) | same |
| Run_G_tf32 | g50 stoch (0.667) | **g20 det (0.758)** | DIFFERENT |
| Run_H | g20 det (0.688) | g20 det (0.688) | same |
| Run_H_anchor | g20 det (0.643) | g20 det (0.643) | same |
| Run_H_fix | g20 det (0.688) | g20 det (0.688) | same |
| Run_H_cycleNoNoise (A1) | g15 det (0.696) | g15 det (0.696) | same |
| Run_H_A2frozen | g110 stoch (3.797) | **g20 det (4.215)** | DIFFERENT |
| Run_H_A2floor | g40 stoch (3.809) | **g20 det (3.934)** | DIFFERENT |
| Run_E | not re-scorable - see below | | |

**conclusion (A0.3 refuted as specified).** The recorded premise - "the
zero-noise default is why stochastic checkpoints never win" - is wrong. The
zero-noise pick was *already* deterministic in Run F/H/H_fix/H_anchor/A1; in the
three arms where the zero-noise pick was stochastic (Run_G_tf32 and both A2
arms) native scoring takes it away and selects the deterministic warmup. The
real reason stochastic stages never won is that at their operating point they
reproduce the data marginal worse - the direct response over-spreads (J/psi
native std 40.8-46.7 MeV vs 28.1 data). **A selection policy is therefore not
the fix; select instead on the A0.1/A0.2 noise-budget columns (within-z width vs
resolution, `sigma_noise_only`), which measure the noise channel directly rather
than through the marginal.** Limitation: the pool is *retained* checkpoints
only, so "same" rows are weaker than they look; the DIFFERENT rows are robust
because the deterministic stage-1 best is the minimum over that whole pool.

**artifact-measured (Run_E not re-scorable - tooling gap, no data lost).** Run
E's `config.resolved.json` points at
`data/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5`, which was moved to `data/legacy/`
(documented in `data/legacy/README.md`); the file exists at the new path, and
`rescore_validation.py` exposes no prior/config override, so the legacy-prior
arms cannot be re-scored without a tool change.

**artifact-measured (Run_H_anchor closed out).** The anchor-only arm (the
single-variable control for the five-change `Run_H_fix`) stopped at global
92/180 with no final evaluation. Read-only audit written to
`outputs/cms_Joint/Run_H_anchor/final_audit/` (Upsilon peak medians on the
reweighted prior, 400k, default seed; noise budget; native slice diagnostic);
driver `logs/anchor_final_audit.ps1`.

In-domain (zero-noise selection score, from `history.json`): stage-1 best 0.642
(g20); stage-2 best 0.945 (g25), last 2.112 (g90) - against Run H 0.839 (g90) /
1.881 (g100) and H_fix 0.696 (g21) / 1.889 (g100). The anchor arm never reaches
Run H's stage-2 best and degrades from g25.

Upsilon transfer at matched epochs (median - CMS fit, MeV, 1S/2S/3S; same 400k
reweighted-prior decode):

| checkpoint | 1S/2S/3S medians [MeV] | 1S std [MeV] |
|---|---|---|
| Run H unanchored g90 (stage-2 best) | -14.3 / -26.0 / -19.0 | 274 |
| Run H unanchored g100 (stage-2 last) | -17.6 / -28.4 / -21.8 | 282 |
| anchor-only g25 (stage-2 best) | -73.8 / -89.9 / -88.8 | 127 |
| anchor-only g92 (last) | -43.1 / -60.4 / -60.8 | 295 |
| H_fix (anchor+extras) g21 (stage-2 best) | -65.6 / -81.5 / -79.8 | 125 |
| H_fix (anchor+extras) g100 (stage-2 last) | +13.2 / -0.0 / +2.9 | 327 |
| H_fix g180 (last) | +11.3 / -2.7 / +1.7 | 330 |

Pipeline validation: the two H_fix rows reproduce the recorded
`best_RunHfix_stage2/stage3` (-66/-81/-80) and `last_model` (+11/-3/+2) to
<=1 MeV.

**conclusion (attribution corrected).** The H_fix centring is not the anchor's
doing. (a) The anchor alone at g92 is -43.1/-60.4/-60.8, *worse* than unanchored
Run H at g90-100 (-14.3...-17.6) and still over-spread (std 295 vs 274/282).
(b) The anchor freezes the map: anchor-only stage-2 best (-73.8/-89.9/-88.8) is
no better than its own stage-1 best (-74.6/-90.1/-90.1), whereas unanchored Run
H calibrates from -50.0/-64.2/-61.6 (g20) to -14.3/-26.0/-19.0 (g90). (c) The
narrow-but-biased signature (std 125-127) appears in *both* anchored arms at
their stage-2 best, so it is the anchor's signature. (d) Centring
(+13.2/-0.0/+2.9 at g100) requires the other four changes in
`cms_Joint_runH_fix.yaml` (cosine LR decay to 5%, `weight_decay` 1e-4, EMA
0.999, stage-3 tail ramp, optimizer carry) and/or the longer schedule.
**proposal:** correct the A2/A4 leaf text that credits the anchor with "the
drift is fixed" to "the anchor freezes the map; it does not centre the peak."
Caveats: the anchor-only arm ends at g92, so it cannot test whether the anchor
alone prevents the ep-180 OOD blow-up; and the extras are a five-change bundle,
so which one centres is not identified (one-controlled-variable rule).

**artifact-measured (anchor noise budget).** The anchor does not restore the
noise channel: anchor stage-2 best/last Upsilon 1S within-z 11.7 / 10.5 MeV
against 84 required, `sigma_noise_only` 13.0 MeV - the same dead channel as Run
H stage-2 best (10.9 / 15.5) and H_fix last (12.0 / 13.8). The anchor contract's
"stay stochastic" guard is satisfied at the *multiplier* level (1.0/0.25) while
the learned sigma stays on its floor: F1/F2 again.

**artifact-measured (anchor slice diagnostic, native 1.0/0.25, g90).** J/psi
per-slice decode mass W1 / floor = 36.7 / 15.9 / 27.7 / 19.5; Z = 2.5 / 3.8 /
3.5 / 1.8 - the same broken conditional regime as H_fix last
(33.8 / 12.7 / 34.3 / 17.3).

**proposal + action (D3 probe).** Launched the plan's D3 probe as a single
controlled change - `--cycle-weight-scale 5` (beta x5 in every stage) on top of
the refuted A2.4 arm (`cycle_decoder_noise: native`, frozen kernel), 20
epochs/stage = 60 globals, `--skip-evaluation`, run name
`Run_H_A2cycW5native`. Rationale: the plan names the knob on the frozen arm but
describes that arm as "a heavier per-event weight **with cycle noise consistent
with the frozen kernel**", and A2.4 already supplies the beta=1 baseline for
exactly that cell (45 globals). Dry-run and CUDA smoke green; both knobs
verified in `config.resolved.json` before launch.

**artifact-measured (D3 probe refuted, same day).** `Run_H_A2cycW5native`
finished the full 60-global schedule (20/20/20 at beta x5) with
`nonfinite_gradient_skips` = 0 throughout; the comparison window is the first
40 globals. The weight did reach training - the resolved config carries
`beta=5.0` in every
stage, and the raw `jpsi_x_loss` over the window is 0.446/0.242/0.227/0.227/0.207
at g21/25/30/35/40, slightly below A2.4's 0.473/0.274/0.242/0.234/0.225 at
beta=1 and well above A2frozen's 0.349/0.146/0.120/0.112/0.099 (the cycle is
denoising, not noise-free). **But the D3 gauge does not move:**

| global | probe b=5 native | A2frozen b=1 zero | A2.4 b=1 native |
|---|---|---|---|
| 21 | 0.509 | 0.531 | 0.365 |
| 25 | 0.410 | 0.462 | 0.483 |
| 30 | 0.562 | 0.445 | 0.410 |
| 35 | 0.369 | 0.401 | 0.464 |
| 40 | 0.601 | 0.360 | 0.736 |

(J/psi encoder latent mass KS; identity baseline 0.384, D3 target 0.04, best
mass-only contraction 0.279.) All three arms stay in 0.36-0.74 - at or above
identity, an order of magnitude above the target - and the probe shows no
downward trend and no advantage over either beta=1 reference. The in-domain
selection score agrees: probe 6.507/4.128/6.162/5.206/8.770 at g21/25/30/35/40
vs A2frozen 6.444/5.292/5.928/3.818/4.490, i.e. neutral to worse.

**conclusion (both cheap D3 options are now exhausted).** Neither the
cycle-noise regime (A2.4 at beta=1) nor a 5x heavier per-event cycle weight
moves the honest-prior latent gauge, so **D3 is not a loss-weighting problem**.
The remaining plan option is the z-space consistency cycle `encode(decode(z)) ~
z`, which is a new loss *term*, not a knob - or the architectural branch. Tool:
`scripts_joint/d3_probe_compare.py`; artifacts
`outputs/cms_Joint/Run_H_A2cycW5native/d3_compare/` (`REPORT.md`,
`d3_compare.json`).

**Tools added this session (source-verified).** `scripts_joint/a03_selection_study.py`
(read-only, resumable A0.3 study driver),
`scripts_joint/d3_probe_compare.py` (first-40-globals D3 readout vs both
beta=1 references), `logs/anchor_final_audit.ps1` (anchor close-out readouts).
Preflight dirs removed per the CLAUDE.md section 3 rule
(`clean_preflight.py --root outputs --apply`: 2 dirs, 1708.0 KiB). Full suite
397 tests, OK (4 skipped), `logs/tests_session55.log`.

## 7. Procedures and tests (2026-09-09/11 session)

Operational companion to sections 1-6: what to run, what it writes, and what
number decides the outcome. Labels follow `CLAUDE.md` section 2.

### 7.1 Test suite

```
python -m unittest discover -s tests        # unittest, not pytest
```

artifact-measured 2026-09-10: **250 tests, OK (4 skipped)**. The four skips are
the two retired `Run_E_encoderDet*` tests (meeting decision) plus the two
environment skips (MPS unavailable; one prior file absent).

Tests added or changed this session:

| test | locks |
|---|---|
| `test_run_e_constant_noise_arm_shares_one_schedule_for_both_maps` | constant shared E=D noise schedule |
| `test_load_config_replace_keys_replaces_named_stage_list` | `replace_keys` replaces, not appends |
| `test_run_f_flat_loss_mechanism_contract` | Run F 4x256, 2 stages, constant shared noise |
| `test_run_f_prior_contract_is_preserved_and_disjoint` | old Run F kept as `Run_F_priorCKKWL` |
| `test_run_g_three_stage_contract` | Run G 20/60/60, stage 3 core 1.0 / tail 0.5 |
| `test_run_h_sliced_loss_contract` | Run H 4x256, 20/80/80, `num_slices` 256, `mass_kin_swd` 0 -> `sliced_mass_w1` 0.5 |
| `test_sliced_mass_w1_component_and_validation` | sliced component is 0 for identical, >0 for a pT-local distortion, 0 when weight 0, rejects bad settings |
| `test_expand_indices_pads_to_floor_and_is_reproducible`, `test_batch_floor_pads_small_streams_and_is_reported` | `min_events_per_update` padding and logging |
| `test_stage_last_checkpoint_path_is_run_prefixed_per_stage` | per-stage last checkpoint naming |
| `test_joint_contract_masks_invariant_mass_and_anchors`, `test_mass_anchor_and_condition_guards_reject_violations` | invariant-mass mask and no-anchor guards |
| updated: `test_v38_vanilla.test_only_reconstruction_and_latent_weights_nonzero`, `test_cms_loss.test_distribution_components_match_pre_refactor_reference` | new component key is backward compatible |

### 7.2 Run H training procedure (trained 2026-09-11; audit in 7.8)

Run H finished the 20/80/80 schedule (ep 180). This section keeps the
procedure; the post-training audit and the recommended transfer checkpoint
are in section 7.8. Do not use `last_model.pt` for Upsilon.

Checkpoints: `best_RunH_stage1_deterministic_warmup.pt`,
`best_RunH_stage2_stochastic_core.pt`, `best_RunH_stage3_stochastic_tail.pt`,
`best_model.pt`, `last_model.pt`, `last_RunH_stage*.pt`.

```
# 1. preflight (passed 2026-09-10; repeat after any config edit)
python scripts_joint/run_joint.py --run H --run-name Run_H_dryrun --device cuda --dry-run
python scripts_joint/run_joint.py --run H --run-name Run_H_smoke --device cuda --smoke --skip-evaluation

# 2. train
python scripts_joint/run_joint.py --run H --device cuda
```

artifact-measured preflight: dry-run 180 epochs, 830,536 parameters; smoke 3
stages / 21 updates with the sliced term exercised; Upsilon not opened.

After training, in this order:

1. **In-domain regression gate.** `outputs/cms_Joint/Run_H/joint_evaluation.json`
   (automatic) and `history.json`. Compare J/psi and Z mass W1/KS against Run F.
   The sliced term must not buy Upsilon structure by hurting the trained regions.
2. **Conditional mismatch (the point of Run H).**
   ```
   python scripts_joint/slice_resolved_diagnostic.py \
     --checkpoint outputs/cms_Joint/Run_H/best_model.pt \
     --output-dir outputs/cms_Joint/Run_H/slice_diagnostic --device cuda
   ```
   Success = the per-slice `mass W1 / floor` ratios fall from the Run F
   3.5-17x toward the floor. No movement means the sliced term is not biting.
3. **pT-resolved response.**
   ```
   python scripts_joint/upsilon_pt_response_test.py --run-dir outputs/cms_Joint/Run_H --device cuda
   ```
   Read `outputs/cms_Joint/Run_H/upsilon_pt_response/REPORT.md`. Success =
   position slope toward the data's -0.4 MeV/GeV (pT<=20: -1.2) instead of the
   Run F stochastic -6 to -8, and sigma slope toward ~0 instead of growing with
   pT (Run F +1.1 to +1.4; data -1.7), with the median position still flat at
   low pair-pT.
4. **Upsilon transfer** (frozen, post-unblinding):
   ```
   python scripts_joint/upsilon/decode_prior.py \
     --prior data/cms_upsilon_mumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_8p5_11p5_1M.hdf5 \
     --checkpoint outputs/cms_Joint/Run_H/best_model.pt \
     --output outputs/cms_Joint/Run_H/upsilon_transfer_best_model/decoded/upsilon_0j1j_prior_decoded_xspace.hdf5 \
     --device cuda --overwrite
   python scripts_joint/upsilon/compare_prior_cms.py \
     --prior outputs/cms_Joint/Run_H/upsilon_transfer_best_model/decoded/upsilon_0j1j_prior_decoded_xspace.hdf5 \
     --prior-dataset FDL/xData --sample-label "Run H" \
     --output-dir outputs/cms_Joint/Run_H/upsilon_transfer_best_model/decoded_vs_cms
   python scripts_joint/upsilon/evaluate_z_to_x.py \
     --decoded outputs/cms_Joint/Run_H/upsilon_transfer_best_model/decoded/upsilon_0j1j_prior_decoded_xspace.hdf5 \
     --output-dir outputs/cms_Joint/Run_H/upsilon_transfer_best_model/quantitative_z_to_x --device cuda
   ```
   Also decode the stage-1 and stage-2 best/last checkpoints. Read the 1S core
   asymmetry (Run F: -0.27 deterministic, -0.42 stochastic) and the 8.5-9.25 GeV
   excess (1.76/1.98; 1.03/1.27 after the continuum reweight).
   `scripts_joint/upsilon_transfer_comparison.py` is still Run-F-pointed; copy
   its `SAMPLES` table for Run H until it gains a `--run-dir` option.
5. **Decide.** If in-domain holds, the slice ratios fall to the floor, but the
   Upsilon per-event asymmetry persists, the residual is the M2 identifiability
   gap (marginals vs per-event inverse), not the sliced term.

### 7.3 Diagnostic tools

| tool | command | output | answers |
|---|---|---|---|
| loss floor | `python scripts_joint/loss_floor_probe.py --run F --output outputs/cms_Joint/loss_floor_probe/F.json` | weighted distributional floor | how much of the loss can move |
| ppzee decomposition | `python scripts_joint/ppzee_error_decomposition.py --overwrite` | `outputs/cms_Joint/ppzee/error_decomposition/` | inherited vs encoder-added error; common vs differential log-pT |
| slice diagnostic | `python scripts_joint/slice_resolved_diagnostic.py --checkpoint <ckpt> --output-dir <dir>` | `slice_diagnostic.json`, `REPORT.md` | conditional mass mismatch vs per-slice floor |
| pT response | `python scripts_joint/upsilon_pt_response_test.py --run-dir <run-dir>` | `upsilon_pt_response/{REPORT.md,json,png,pdf}` | model vs data peak position/width vs pair-pT |
| continuum reweight | `python scripts_joint/upsilon_continuum_reweight.py --run-dir <run-dir> --write-reweighted-prior <path>` | `upsilon_continuum_reweight/` | sideband-reweighted continuum and its decode |
| prior reweight plot | `python scripts_joint/upsilon_prior_reweight_plot.py` | `prior_reweight_comparison.{png,pdf,md,json}` | original vs reweighted prior |
| transfer comparison | `python scripts_joint/upsilon_transfer_comparison.py` (Run-F-pointed) | `upsilon_transfer_comparison/` | checkpoint table + overlay |
| TF32 impact | `python scripts_joint/tf32_impact_probe.py` | `outputs/cms_Joint/tf32_probe/tf32_impact.json` | numerical impact and matmul speedup |
| identity/floor reference | `python scripts_joint/identity_baseline.py --run-dir <run-dir> --out <file.json>` | identity-map and finite-sample-floor numbers (nothing without `--out`) | what the strict latent gate reads with no model at all |

Upsilon pipeline pieces: `scripts_joint/upsilon/decode_prior.py` (z -> x),
`compare_prior_cms.py` (peak test), `evaluate_z_to_x.py` (paperstyle density
ratios + `z_to_x_metrics.json`).

### 7.4 Contract guards

source-verified, `scripts_joint/run_joint.py` (called in `main()` before training):

- `assert_mass_not_conditioned(model, run_label)` - feature index 8
  (`log_pair_mass`) must be absent from both encoder and decoder masks.
- `assert_no_mass_anchor(config, run_label)` - `resonance_mass_w1` and `mass_w1`
  must be 0 in every region's merged loss. A deliberate mass-anchor experiment
  must set top-level `allow_mass_anchor: true`.

The distributed mass terms (`pair_mass_w1`, `mass_kin_swd`, `physics_swd`, the
per-event cycle, `sliced_mass_w1`) are targets, not anchors, and stay on.

### 7.5 Checkpoint policy

- `best_model.pt` - global best by in-domain selection score (it is a pre-noise
  stage-1 checkpoint for Run E and Run F; Run G/H will write it if trained).
- `best_<Run>_<stage>.pt` - per-stage best.
- `last_model.pt` - global last evaluated epoch (last stage's last epoch).
- `last_<Run>_<stage>.pt` - per-stage last evaluated epoch (new 2026-09-10).
- Resume uses `last_model.pt` + `history.json`; the per-stage files are for
  post-hoc model selection, not resume.

### 7.6 TF32

`performance: {tf32: true}` (per config, CUDA only; `Run_G_tf32` is the ready
variant). Recorded in `provenance.json` as `matmul_precision`. artifact-measured:
pure matmul max rel error 2.8e-4; model outputs 4.9e-7 max / 1.8e-8 rms; losses
1e-5..3.7e-4; gradients 2e-5..2.6e-4; matmul-only speedup 1.30x. It is not
bit-comparable with the fp32 record, and the pipeline bottleneck is the SWD
sorts, so the end-to-end gain is smaller.

### 7.7 Run keys

`--run` resolves `configs_joint/cms_Joint_<key>.yaml`

| key | file | note |
|---|---|---|
| `E` | `cms_Joint_runE.yaml` | accepted deterministic J/psi+Z model |
| `E_constNoise` | `cms_Joint_runE_constNoise.yaml` | constant shared noise arm (dry-run only) |
| `F` | `cms_Joint_runF.yaml` | flat-loss mechanism validation (trained) |
| `F_priorCKKWL` | `cms_Joint_runF_priorCKKWL.yaml` | preserved old Run F prior contract |
| `G` | `cms_Joint_runG.yaml` | three-stage 4x128 (prepared, not trained) |
| `G_tf32` | `cms_Joint_runG_tf32.yaml` | Run G + TF32 |
| `H` | `cms_Joint_runH.yaml` | pT-sliced conditional mass (trained 2026-09-11; audit in 7.8) |

### 7.8 Run H post-training audit (2026-09-11)

artifact-measured, `scripts_joint/runH_tail_audit.py --device cuda --max-events 400000`.
Full write-up: `outputs/cms_Joint/Run_H/tail_audit/DIAGNOSIS_AND_FIX.md`.

- Stage schedule: 20 ep `0/0`, 80 ep `1.0/0.25`, 80 ep `1.0/0.5`, shared E=D.
- Upsilon median minus CMS fit [MeV]: `best_RunH_stage2` (ep 90) -14/-25/-19;
  `best_RunH_stage3` (ep 115) -18/-29/-24; `last_model` (ep 180) -125/-136/-136.
- The ep-180 shift is unchanged with tail 0/0.25/0.5/1.0 and with all noise off;
  it lives in the deterministic mean map. The tail multiplier is inert because
  the learned tail sigma collapsed to ~1.8e-4 (init 8.1e-4) and the core sigma
  sits on the 1e-3 floor.
- Use `best_RunH_stage2_stochastic_core.pt` (ep 90) for Upsilon transfer. Any
  tail fix must satisfy the four acceptance criteria in the diagnosis doc,
  especially criterion 2 (turning the noise off must shrink the decoded width).

### 7.9 Drift-control arms (2026-09-11)

```
python scripts_joint/run_joint.py --run H_anchor --run-name Run_H_anchor_dryrun --device cuda --dry-run
python scripts_joint/run_joint.py --run H_anchor --run-name Run_H_anchor_smoke  --device cuda --smoke
python scripts_joint/run_joint.py --run H_fix    --run-name Run_H_fix_dryrun    --device cuda --dry-run
python scripts_joint/run_joint.py --run H_fix    --run-name Run_H_fix_smoke     --device cuda --smoke
```

- `H_anchor`: single controlled change versus Run H -- the frozen mean-map
  anchor (weight 0.5) on stages 2-3, reference = the stage-1 warmup.
  Encoder/decoder noise untouched.
- `H_fix`: all four levers -- anchor + cosine LR decay to 5% + `weight_decay`
  1e-4 + per-step EMA 0.999 + stage-3 tail ramp `0.25 -> 0.5` + optimizer
  carry.
- Contract keys: `mean_map_anchor: {enabled, reference_stage,
  events_per_region}`; stage keys `mean_map_anchor_weight`,
  `carry_optimizer_state`; loader key `ema_decay`. The guard rejects an anchored
  stage that zeroes either direction's noise.
- Acceptance: the four criteria in
  `outputs/cms_Joint/Run_H/tail_audit/DIAGNOSIS_AND_FIX.md` section 5; the
  decisive one is that turning the noise off must shrink the decoded width
  (`sqrt(sigma_native^2 - sigma_noise0^2) >= 50 MeV`, today ~15 MeV).
