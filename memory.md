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

### 1.6 Update 2026-10-04 — where the project actually stands

**The live programme is branch P of `docs/project_tree.md`** ("conditional spread /
scoring-rule programme", phases P0–P4, approved 2026-10-04). Read that section
before proposing any new experiment.

- **artifact-measured.** The strictly proper scoring rule on the conditional
  (`Run_H_SR`) is **refuted by its own pre-declared criteria**: 4 of 5 fail, and
  the OOD peak shift triples against its baseline (`Run_H_D3b`). See
  `outputs/cms_Joint/Run_H_SR/VERDICT.md` and Session 72.
- **artifact-measured (new, Session 73).** The missing control cell was measured:
  `Run_H_A2floor` is a kernel floor with a learned amplitude and **no** score
  term, and its zero-noise mean-map spread runs 29.1 -> 32.8 -> 33.0 -> 15.1 MeV
  (g20 -> g180). Contraction of the mean map happens **with or without** the
  score, so Session 72's "the score actively prefers a narrower map" is
  downgraded to a hypothesis. What survives unchanged: the score carries no
  upward gradient (pinned at 91.5 from epoch 1) and every live arm is
  **kernel-limited** (kernel-only 34–45 MeV against 28.0 MeV of data).
- **Five mechanisms are refuted** for the same failure (learned amplitude, cycle
  noise, beta x 5, z-cycle, proper scoring rule). The open object is not the
  loss: it is the **mean map** and the **kernel's calibration target**.
- **Nothing is training.** GPU free, no background jobs. Tests green at 545 OK
  (4 skipped) before Session 74's additions.
- **P2 and P4 are DONE (Session 74).** The criterion exists
  (`scripts_joint/conditional_spread.py` + the ppzee paired bench) and the
  calibration target is pinned (`docs/calibration_target_2026-10-04.md`). Two
  corrections came out of it and supersede the Session 71-73 wording:
  **84 MeV is the Upsilon(1S) reference, not a J/psi number** (our own fit gives
  84.42 +/- 3.12 MeV), and the Phase 1 score term is now known to have been
  **clamped in all 8 coordinates** (claimed/reference 0.014-0.031 against a 0.1
  floor), so it never had an amplitude gradient — Session 72's "the score drove
  the amplitude down" is withdrawn.
- **The live blocker is the kernel amplitude.** Adopted target J/psi
  23.81 MeV (std quadrature); the kernel as shipped implies 33.97 MeV = **1.43x**.
  The mean-map variance budget is negative under every reading, so P3 (mean-map
  architecture) cannot work until the kernel comes down. **Next: a kernel rescale
  is the first authorised-looking run — but it needs explicit approval.**

### 1.7 Update 2026-10-04 (Session 75) — decision documents rewritten to the latest state

- **source-verified (this session):** the decision-document set was consolidated
  at the user's request. `docs/project_tree.md` is now **v2**: every inline
  update/correction from Sessions 29-74 is folded into the final state, branch P
  is closed with its final verdicts, a benchmark board (section 6.0) fixes the
  reference numbers all claims are scored against, and a new branch **I**
  records six innovation candidates (I1 alternating-freeze identification, I2
  two-resonance joint kernel calibration, I3 the formal non-identifiability
  proposition, I4 paired-sim amplitude transfer, I5 component-aware latent for
  the honest J/psi, I6 Upsilon width pT/eta audit) — **all labelled proposal,
  none authorised**. `paper/FRAMING.md` is v2: final RQ1-RQ3 answers, claim
  ladder with item 5 withdrawn, updated PRD assessment, milestones M1 closed /
  M2-M4 live. `docs/calibration_target_2026-10-04.md` was re-laid out with
  **no number changed**. `docs/literature_2026-10-04.md` gained section 4.1
  recording that its recommendation 1 (a proper scoring rule) was implemented
  and refuted the same day. This file's history is untouched (append-only
  convention).
- **proposal:** the live next step is unchanged by the rewrite: **S2** (kernel
  rescale by 0.701 to the adopted P4 target, one controlled change on the D3b
  config, ~5 h GPU) **needs explicit approval**. Cheap read-only parallel items:
  S2c (Crystal Ball + exponential J/psi peak fit, decides E vs A) and I2 (joint
  two-region kernel fit predicting the Upsilon amplitude).
- No training started, no checkpoint or artifact modified. The P-branch
  working-tree changes (scripts, configs, outputs) remain uncommitted.

### 1.8 Update 2026-10-04 (Session 75-76) — decision docs consolidated, I2 refuted the global kernel rescale, the knee kernel is training

- **source-verified (Session 75).** `docs/project_tree.md` is v2 (all inline
  corrections folded in, benchmark board section 6.0, branch I candidates);
  `paper/FRAMING.md` is v2; `docs/calibration_target_2026-10-04.md` re-laid out
  with no number changed; `docs/literature_2026-10-04.md` gained section 4.1
  recording that its recommendation 1 was refuted the same day.
- **artifact-measured (I2, read-only).** `scripts_joint/kernel_joint_calibration.py`
  → `outputs/cms_Joint/kernel_joint_calibration/`. Median muon pT J/psi 13.14,
  Z 40.96, Upsilon(1S) 4.67 GeV; required per-muon log-pT amplitudes J/psi
  0.01371, Upsilon 0.01262, Z 0.03936 (flat between 4.7 and 13.1 GeV). **A single
  `offset + slope*pT` law cannot satisfy the three regions**: solved exactly on
  J/psi + Z (robust denominators 30.02 MeV / 2.538 GeV) it predicts Upsilon(1S)
  **0.367x**. **S2 as written (global factor 0.701) is refuted before any GPU
  time** — it would drive Z to 0.58x and Upsilon to 0.60x. **Disclosure:** the
  shipped kernel's Upsilon 0.86 is not a prediction — `upsilon1s: 0.084` is one
  of its three fit targets.
- **source-verified (the knee kernel).** `cylindrical_flow.py` gained an optional
  `knee_pt_gev` on the linear form (plateau = multiple-scattering floor, linear
  rise above); `tests/test_kernel_knee.py` (5, green). Fit by
  `scripts_joint/kernel_knee_fit.py` → `outputs/cms_Joint/kernel_knee_fit/`
  (+ `selected_knee20/`). At knee 20 GeV: J/psi 1.021x, Z 1.007x, and
  **Upsilon 1S/2S/3S predicted at 0.959/1.021/1.078x**. Adopted knee 20 GeV (the
  plateau middle, NOT the grid argmax 24; 16/24 are the systematic band).
- **artifact-measured (I6).** `scripts_joint/upsilon_map_spread_audit.py` →
  `outputs/cms_Joint/upsilon_map_spread/`. The D3b Upsilon zero-noise "214 MeV"
  is a **pair-pT extrapolation artefact**: residual std 27.5 MeV at pair-pT
  0.03-1.34 GeV rising to **372.7 MeV** in the top sextile (7.4-160 GeV), with the
  bias flipping from +88.7 to -163.9 MeV. 2S/3S identical. The Upsilon width fix
  is therefore a **pT-resolved** mean-map constraint.
- **artifact-measured (S2c, inconclusive).** `outputs/cms_Joint/s2c_peak_fit/`:
  the Crystal Ball + exponential fit on the locked split has chi2/dof 1233 with
  the tail parameters railed, so its widths are not usable. **The calibration
  target remains a decision (E adopted), not an empirical settlement.**
- **source-verified (I3).** `docs/non_identifiability_2026-10-04.md`: re-mixing
  invariance proves every marginal-only objective is exactly invariant under
  `(M, p_eps) -> (M∘T, p_eps∘T)`, `T#sigma = sigma`, so the split is not
  identified without pairs, a constrained mean map, a supplied kernel, or an
  injectivity condition.
- **RUNNING.** `Run_H_kneeKernel` launched 2026-10-04 20:04 (`configs_joint/
  cms_Joint_runH_kneeKernel.yaml`, extends D3b, kernel spec the only change;
  dry-run + smoke green, preflights removed). 180 epochs, 30,960 updates, 830,536
  parameters, ~3.2 GiB. Pre-declared readout and kill condition in the config
  header. **This is the only thing running.**
- **artifact-measured + source-verified (2026-10-04, later the same day).** The
  run was stopped at **global epoch 144** on the user's instruction (the arm is
  to be redeployed on a Linux server *next* time), then **resumed in place**.
  `--resume` resolves `<run>/last_model.pt` (`run_joint.py:606`), whose recorded
  state was **global 140 / stage 3 / stage_epoch 40** — validations are every 5
  epochs and history.json logs every epoch, so 141-144 were lost, not the whole
  segment. The resume path skips completed stages, restarts the matching stage at
  `stage_epoch + 1`, and **restores the AdamW moments**
  (`joint_trainer.py:815-817`); the log confirms
  `Resume: runH_stage3_stochastic_tail stage-best score restored as 4.65235`.
  **A `--dry-run --resume` preflight is NOT possible**: dry-run rewrites the
  output directory to `*_dryrun` and rebuilds the manifest from the
  1000-sample splits, so both the checkpoint lookup and the
  `joint_contract_sha256` guard fail for reasons that do not exist in the real
  run. The guard must be validated by launching for real and reading the log.
  Resumed 2026-10-04, local epoch 41/80 (global 141), train_loss 1.30,
  log `logs/Run_H_kneeKernel_resume.log`. ~40 epochs to go. The resume
  **truncates `history.json` to the checkpoint's epoch and re-runs from there**:
  the file now holds epochs 1-143 with no duplicate `global_epoch` entries, so the
  four epochs (141-144) that ran after the last validation were re-run rather than
  kept — expected, and the reason a resumed run's history is internally
  consistent instead of carrying duplicate rows. (Note for future probes:
  `history.json` rows do **not** carry a `train_loss` key; the per-epoch loss is
  only in the log.)

### 2026-10-04 - Session 77, prior layout flattened into data/ (and the two-resolver hazard)

**source-verified (what changed).** The three unified priors were copied from
`data/legacy/priors/` to `data/`, and the four joint configs that referenced them
were repointed from `legacy/priors/<name>` to the bare `<name>`:
`runH_A2frozen`, `runH_A2floor`, `unifiedP1`, `unifiedP1_components`. Reason: the
validation host stages `data/` flat, and the server run died in
`cms_data.file_fingerprint` on the old path. **Copied, not moved** - every existing
checkpoint embeds the OLD absolute path in its own config, and
`scripts_joint/rescore_validation.py` resolves the config from the checkpoint, so
the originals must stay for old runs to remain re-scorable.

**source-verified (two resolvers that need not agree).**
`cms_data.data_cache_metadata` fingerprints the CONFIGURED path with **no
fallback**; `fixed_z_noise_budget.resolve_prior_path` tries
`data/<configured>` -> `data/legacy/<basename>` -> `data/<basename>`. The
2026-10-04 server failure came from the first. When they resolve differently the
run loads one prior while fingerprinting another, so the new checker reports a
disagreement as a **failure**, not a pass.

**artifact-measured (the contract consequence - this one bites).**
`contract_sha256` is `_json_hash(identity)` over `regions.<name>`, and only
`data_cache` is popped (`joint_data.py:299-305`); `data_contract` embeds
`file_fingerprint` (absolute path + size + mtime). **Repointing a prior path
therefore changes the contract digest**: `Run_H_kneeKernel` can no longer be
re-scored with `run_joint.py --evaluate-only` or `--resume` (the guard at
`run_joint.py:617` rejects it). It **can** still be re-scored with
`scripts_joint/rescore_validation.py`, which reads the config from the checkpoint
and still finds the legacy copy. No run in flight is affected: a running process
resolved its config at startup.

**source-verified (new tool).** `scripts_joint/check_data_layout.py` (read-only)
resolves each config's `cms_root_file` and every region prior with **both**
resolvers, checks the HDF5 datasets the loader needs (`FDL/zData`, plus
`FDL/component_id` and `FDL/weight` when `prior_components.enabled`), and exits
non-zero on a missing file, a missing dataset or a resolver disagreement.
Flags: `--run <ID>`, `--config`, `--all`, `--json`. artifact-measured:
`--all` = **34 configs, 0 problems** on this host.

**Verification.** 91 focused tests OK, 2 skipped (`test_joint_runa`,
`test_a1_a2_a0_noise`, `test_cms_loss`, `test_kernel_knee`). The region cache is
invalidated for the four changed configs (the key hashes the prior path), so the
next run on them rebuilds the cache once - expected, not a fault.

**proposal (not done).** The originals under `data/legacy/priors/` are now a
second copy (225 MB). They are load-bearing for old checkpoints, so do not delete
them without checking which runs still need re-scoring.

### 2026-10-05 - Session 78, Run_H_kneeKernel read out: the calibration landed, the width did not

**artifact-measured (execution).** `Run_H_kneeKernel` finished: 180/180 epochs, exit 0,
`joint_evaluation.json` written 00:34, `history.json` 180 rows, peak CUDA 3.20 GiB,
~81 s/epoch in stage 3. Upsilon was never opened. The resume carried stage 3 from local
epoch 41 to 80 (global 141 -> 180).

**artifact-measured (selection; lower is better).** `joint_selection.selection_score`:
stage-1 best ep20 **4.3126** (jpsi 4.313, z 0.783) < stage-2 best ep75 4.362 < stage-3
best ep165 4.464 < last ep180 4.816. The selected checkpoint is the **stage-1
deterministic warmup**, and the stochastic stages degrade monotonically - the pattern
every stochastic arm here has shown. D3b's stage scores were 4.337 / 4.551 / **3.886**,
so **D3b's best beats anything this arm produced**.

**artifact-measured (Axis R - the knee calibration works).** `response_scorecard`, 8
checkpoints: jpsi within-z robust **29.9 MeV against a required 30.0 (ratio 0.996)**,
`sigma_noise_only` 33.2 MeV; z within-z robust 2048.4 against 2537.6 (0.807). Gate R =
**PASS**, `degenerate=False`, `identified=True`. The kernel lands exactly on target.

**artifact-measured (the physics readout FAILS as pre-declared).**
`d3b_readout_probe` on four checkpoints: zero-noise mean-map spread
18.4 / 22.8 / 20.5 / 19.9 MeV and **native total 43.7 / 42.9 / 40.8 / 39.8 MeV**
(stage1best / stage2best / stage3best / last). D3b was 17.6 and 38.8. The pre-declared
band was **28-35 MeV**, so the kill condition fires **on its letter**.

**hypothesis, now measured (the band was wrong; the kernel attribution was not).** The
band was sqrt(17.6^2 + 30.0^2) = 34.8, which mixes a **robust** kernel target (30.0) with
a **std** decode. The estimator-consistent std quadrature is
sqrt(20.5^2 + 33.2^2) = **39.0**, and that is what was measured (39.8-40.8). The channel's
robust width is 29.9 but its **std is 33.2**, because the kernel shape is a Gaussian core
plus a 25% Student-t tail, while the CMS data has std 28.03 **below** its robust 29.97.
Matching the robust denominator with a tail-heavy kernel therefore **overshoots the std by
~19%**. The D3b attribution survives; my numeric prediction failed for the
estimator-mismatch reason `docs/calibration_target_2026-10-04.md` keeps warning about.

**artifact-measured (Axis C: jpsi still fails, Z is at gate).** jpsi native slice ratio
median **24.1**, max 43.9 (gates 3 / 10); zero-noise median 31.9. D3b was 18.6-23.4 /
33.6-51.4, so jpsi conditional closure is unchanged to slightly worse. **z native median
3.15, max 5.50 - essentially AT the gate**, so the Z conditional closure is solved. Gate
line: `R=PASS C=FAIL degenerate=False identified=True`; the scorecard rejects
`last_model` and ranks the stage-1 checkpoint first.

**artifact-measured (OOD, report-only; continuum-reweighted prior, native multipliers,
400k events).** Per-state medians minus the CMS fit, 1S/2S/3S: stage1best (selected)
**+71 / +57 / +58**; stage2best -148 / -187 / -195; stage3best -91 / -124 / -130; last
-60 / -90 / -92 MeV. Against D3b (+40 / +28 / +31) and A2frozen stage-3
(+20.5 / +4.1 / +6.0) the selected checkpoint is **worse**, even though its channel
amplitude is now correct - so the OOD peak shift is not driven by the channel amplitude
either.

**conclusion.** S2' did **not** produce the M2 positive result. What it established:
1. the knee law is a genuine single-law calibration - it hits its target in both training
   regions, and read-only it is what made a single law possible at all (I2);
2. once the amplitude is calibrated, the decoded J/psi width is governed by the kernel's
   **shape** (the core/tail ratio) and by the ~20 MeV the mean map still carries, **not**
   by the amplitude;
3. Z conditional closure is at gate; J/psi is not;
4. the in-domain selection score and the OOD peak shift are both worse than D3b, so the
   knee kernel is not an improvement on the shipped one at equal training budget.
**Next lever, in order:** the kernel `tail_ratio` (0.25) - the single number that
separates the robust and the std widths - then the mean map's ~20 MeV residual. Do not
spend another arm on the amplitude. **And write the next arm's pre-declared criteria in ONE
estimator**: this arm failed its own readout only because the prediction mixed robust and
std.

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

### 3.6 Prior component widths, measured (2026-10-04, artifact-measured)

*Added because a question about "are we training on a smeared prior?" produced a
wrong answer from filenames alone. Measured per `FDL/component_id` with the
region's own mass window; `robust` = (q84-q16)/2.*

| file | component | n | std [MeV] | robust [MeV] | weights |
|---|---|---|---|---|---|
| `jpsi_unified_bare_tms10.hdf5` (moved to data/ root) | 0 (jpsi) | 717,904 | 30.53 | **0.35** | ESS/N 0.982 |
| " | 3 (continuum) | 933,788 | 172.26 | 202.44 | 0.985 |
| `cms_jpsi_..._1M_reweighted.hdf5` (data/ root) | 0 | 850,000 | 6.48 | **0.20** | none |
| " | 1 | 150,000 | 34.69 | 41.02 | none |
| `legacy/cms_jpsi_ab_smeared.hdf5` | (unlabelled) | 943,231 | 26.25 | **27.73** | none |
| `z_unified_bare_tms22p8.hdf5` (moved to data/ root) | 3 | 668,278 | 4864.7 | 2792.7 | 1.029 |
| `cms_dymumu_..._70_110_1M.hdf5` (data/ root) | (unlabelled) | 1,000,000 | 4670.8 | 2654.0 | none |
| `upsilon_prior_continuumReweighted.hdf5` | 0/1/2 (1S/2S/3S) | 131,611 / 39,484 / 25,761 | 110 / 166 / 189 | 3.4 / 4.2 / 5.6 | none |

**Read.** The **signal** components of the priors the joint configs point at are
narrow (0.35 MeV robust), so the A2 -> D3b -> kneeKernel line does **not** train
on a smeared prior. The only smeared file on disk is
`cms_jpsi_ab_smeared.hdf5` (27.73 MeV), the A/B arm behind the documented caveat
that Run E's success depended on pre-smearing. **The root-level
`..._1M_reweighted.hdf5` is ALSO narrow (0.20 MeV) and must not be called
smeared**; what distinguishes it is that it was 4-D reweighted to CMS kinematics
(ref-cache lost, ESS 63k/1M) and carries no `weight`/`component_id` labels for
the loader.

**Why the config cannot use the root-level Z file.** Measured, the two Z samples
are nearly the same distribution (std 4864.7 vs 4670.8 MeV; robust 2792.7 vs
2654.0). The blocker is the data contract, not the physics:
`prior_components.enabled: true` with `resample_if_ess_ge_0p5` makes the loader
require `FDL/component_id` **and** `FDL/weight`, and the root-level Z file has
neither. Substituting it would be a contract change, not a free swap.

**TMS, and why these versions.** source-verified,
`docs/unified_prior_method.md` section 5: the merging scale is set per region by
`TMS = max(10 GeV, M/4)` -> J/psi 10.0, Upsilon 10.0, Z **22.8**. The recovered
original cards used `Merging:TMS = 5.0` everywhere; section 5 states that
Upsilon and Z show real TMS sensitivity because the bulk of their fiducial
samples sits near or below TMS. The unified priors were regenerated at the
per-region value, which is why the filenames carry `_tms10` / `_tms22p8`.

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
- P branch, conditional spread / scoring rule (added 2026-10-04; decision tree is `docs/project_tree.md` section 5 "P"): instruments `scripts_joint/scoring_rules.py` (Gaussian log score only; three rejected rules documented in the docstring), `scripts_joint/toy_identifiability.py`, `scripts_joint/toy_unpaired_score.py` (the unpaired-target extension with its `--kappa 0` control and `--mean-head unbounded`), `scripts_joint/single_observation_limits.py`, `scripts_joint/conditional_spread.py` (the unpaired criterion: split, claim vs both denominators, coverage with a finite-draw null, permutation control, identity rail), `scripts_joint/paired_quantile_calibration.py` (the paired ppzee bench: PIT, coverage, pull, each against a same-n/same-D null), `scripts_joint/marginal_vs_scoring_rule.py` (`--shuffle-target`), `scripts_joint/d3b_readout_probe.py` (`--checkpoint LABEL=RELATIVE_PATH`, zero/native decomposition, ~7 s/checkpoint on CUDA), `scripts_joint/plot_joint_paperstyle.py --noise native|zero|both`, `scripts_joint/plot_upsilon_arm_comparison.py`, `scripts_joint/upsilon_noise_channel_test.py`; wiring `scripts/loss.py::DualSpaceFeatureOTLoss.energy_score_loss`, `joint_trainer` `kappa` + `scoring_rule.{draws,score_batch}`, `run_joint.build_loss_factories` (`scoring_rule_config`); config `configs_joint/cms_Joint_runH_scoringRule.yaml` (extends `cms_Joint_runH_D3b.yaml`); artifacts `outputs/cms_Joint/{toy_identifiability,single_observation_limits,toy_unpaired_score,toy_unpaired_score_kappa0,toy_unpaired_score_unbounded,conditional_spread,Run_H_SR,Run_H_D3b,d3b_readout_probe}`, `Run_H_A2floor/final_readout/`, `ppzee/quantile_calibration/`, `conditional_spread/score_anatomy.json`; documents `docs/calibration_target_2026-10-04.md` (P4, adopted target), `docs/literature_2026-10-04.md`, `litreview/unfolding_identifiability_review_2026-10-03.md`; tests `tests/test_scoring_rules.py` (19), `tests/test_energy_score_term.py` (10), `tests/test_conditional_spread.py` (26), `tests/test_paired_quantile_calibration.py` (44).
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

### 2026-10-01 - Session 56, latest/best Run H identified; paperstyle + Upsilon transfer delivered

**artifact-measured (arm choice).** The latest Run H arm by directory mtime is
`Run_H_A2cycW5native` (2026-09-29), a refuted 60-global D3 probe (beta x5, no
latent-gauge movement) - not a candidate. The best completed arm is
`Run_H_A2frozen` (A2.3 frozen kernel, 180 epochs, 20/80/80):
`best_model.pt` is the stage-3 best at **global epoch 110**, zero-noise
selection score **3.7981**, the minimum over the completed honest-prior arms
(`Run_H_A2floor`'s best is 3.8095 at g40, and its learned above-kernel
correction was rejected). source-verified: the full state dict of
`best_model.pt` is tensor-identical to
`best_RunHA2frozen_stage3_stochastic_tail.pt` (`torch.equal` over every
key, 2026-10-01); sha256
`e8a9b25f6683a8a9b51b5be1fef7950a2d6a55fa345e369d34ca3637de2a6be4`; native
multipliers (core, tail) = (1.0, 0.5), encoder = decoder.

**artifact-measured (new paperstyle, in-domain).**
`outputs/cms_Joint/Run_H_A2frozen/paperstyle_best_model/` and
`paperstyle_last_model/` - 22 plots each (15000 events per distribution,
split test, the run's own `config.resolved.json`), epochs 110 / 180.
Shown sliced-Wasserstein^2 on the J/psi x-space mass: cycle 2.104e-05, decoded
prior 1.536e-05; Z 4.200e-01 / 3.995e-01. These are J/psi + Z only and carry the
known honest-prior shape deficit (Z ratio 0.5-1.5 across 75-105 GeV).

**artifact-measured (new Upsilon transfer, frozen checkpoints).**
`scripts_joint/upsilon_transfer_test.py` was run on the raw 1M prior for
stage2-best, stage3-best (= `best_model.pt`) and last; overlays were built
for both priors. Primary (continuum-reweighted) readout, mass W1 / KS / 1S std /
8.5-9.25-over-CMS: stage3 best **0.0407 / 0.0295 / 297.3 MeV / 0.948**, last
0.0260 / 0.0267 / 290.1 / 1.123, stage2 best 0.0727 / 0.0760 / 361.3 / 1.320.
Raw-prior rows carry the raw prior's own continuum shape (Session 51c) and are
secondary. Peak medians (existing `upsilon_peak_medians/`): stage3
+21.6/+2.9/+6.3 MeV, last +18.9/+6.9/+13.3, stage2 -147.7/-169.1/-172.4;
best-vs-last 1S drift -2.7 MeV. Consolidated write-up:
`outputs/cms_Joint/Run_H_A2frozen/UPSILON_TEST_2026-10-01.md`.

**D2 status (four criteria, `DIAGNOSIS_AND_FIX.md` section 5).** 2 of 4 green:
criterion 1 (medians for best and last) passes for g110/g180, criterion 2
(`sigma_noise_only >= 50 MeV`) passes at 91-101 MeV; criterion 3 (in-domain
score) stays blocked on D3/honest-prior latent gauge and criterion 4 (slice
ratios toward the floor) still fails at 36-87x floor. The Y(1S) width is
290-299 MeV against the 84 MeV CMS resolution - F2/D3, not tail noise.

**source-verified (tool change).** `scripts_joint/upsilon_transfer_comparison.py` gained
`--title` (default derived from the run directory), reference-prior and CMS
provenance lines in its REPORT.md, and a hard failure when no decoded transfer
directory is found. Its test file is untouched; full suite **397 tests, OK
(4 skipped)**, `logs/tests_session56.log`.

No checkpoint, `data/` file or existing output was modified; nothing was
trained.

### 2026-10-01 - Session 57, A0.4 response scorecard built, runs E-H scored, evaluation notebook

**source-verified (new mechanism).** `scripts_joint/response_scorecard.py`
(plan-tree leaf A0.4) + `tests/test_response_scorecard.py` (19 tests). It measures
three axes per checkpoint and turns them into a selection key:

- **R - response identification:** fixed-z decode budget per region (within-z
  robust width vs the required resolution, `sigma_noise_only`, and the exact split
  `Var = E_z[Var(x|z)] + Var_z[E(x|z)]`). Gate band 0.8-1.25; **degenerate when
  within/required < 0.5**.
- **C - conditional closure:** per pair-pT quartile mass W1 / finite-sample floor at
  zero and native noise; gate median <= 3, worst slice <= 10.
- **T - transfer (report-only):** Upsilon per-state medians vs the CMS fit. A unit
  test proves Axis T cannot change the selection key.
- **Key:** reject R-fail/degenerate, rank survivors by C median then in-domain
  score, else `identification_failed` with the least-bad R. This is the
  operational form of the A0.3 verdict ("select on the A0.1/A0.2 columns").

**artifact-measured (11 arms / 22 checkpoints, `outputs/cms_Joint/scorecard/`).**
256 fixed z, 32 draws, 40k slice events, GPU, ~17 s per arm. Every legacy arm
(Run E, F, G_tf32, H, H_fix, H_anchor, A1) is **degenerate at every checkpoint**
- Z within/required 0.038-0.088, deterministic checkpoints exactly 0. The four A2
arms' stochastic checkpoints are the **only identified** ones (J/psi 0.98-1.22, Z
0.81-0.85) and their keys select a stochastic checkpoint. Axis C fails everywhere
(median slice ratio 3-75x floor), including the identified A2 checkpoints - the
conditional defect is unchanged.

**artifact-measured (retrospective check, the A0.3 question).** On the complete
A2frozen pool (`scorecard/Run_H_A2frozen_all`, 8 checkpoints) the key **rejects
both stage-1 deterministic warmups** as degenerate and selects
`last_RunHA2frozen_stage2_stochastic_core` (g100, C median 20.2). Native-noise
marginal scoring selected the g20 warmup in 8 of 8 arms; this key cannot.

**Cross-validation (artifact-measured).** The scorecard reproduces the A0 numbers
through a different code path: Run H last J/psi 1.73 x 12.2 = 21.1 MeV (A0
21.1-22.1), Z 0.0377 x 2537.6 = 95.7 MeV (A0 95-98); A2frozen J/psi
1.09 x 30.0 = 32.7 MeV (A2_VERDICT 31.1-34.4), Z 0.829 x 2537.6 = 2.10 GeV
(A2_VERDICT 2.12-2.18).

**source-verified (notebook).** `scaffold/response_scorecard_runs_E_H.ipynb` -
executed with results inline (12 arms / 30 checkpoints, table + figure + automatic
verdicts). Runner: `scripts_joint/execute_notebook.py` (nbclient; nbconvert is not
installed). nbformat/nbclient/ipykernel were installed into the working Python
3.10; **pandas is not installed** and the notebook deliberately avoids it.

**Bugs found by running and fixed (source-verified).** (1) `math.log(0)` crash on
deterministic checkpoints (within-z width exactly 0) - now degenerate with an
infinite R deviation; (2) the degeneracy flag was first keyed on the noise
variance share, which is prior-dependent (A2frozen J/psi share 0.016 with a live
channel) - re-keyed to the width ratio, share kept as a diagnostic; (3) D2
criterion 2 now reports *not measured* (`None`) unless Axis R ran the Upsilon
states; (4) Axis C degrades gracefully when the region cache cannot be built,
leaving an R-only scorecard instead of failing.

**Run_E data note (source-verified, from a read-only audit).** Run_E's config
points at the moved legacy jpsi prior and `load_joint_regions` raises before it
checks the cache; the scorecard's fallback resolves `data/legacy/...` in memory
and lands on a cache hit, so Run_E scores without touching `data/`.

Full suite: **416 tests, OK (4 skipped)**, `logs/tests_session57.log`. No
checkpoint, `data/` file or existing output was modified; nothing was trained.

### 2026-10-01 - Session 58, D3 z-space consistency cycle implemented and preflighted

**source-verified (new loss term, plan-tree D3).**

- `scripts/loss.py`: `DualSpaceFeatureOTLoss.z_cycle_loss(z_true, z_reco)` = a
  paired standardized MSE in the z coordinates the encoder targets, registered
  as `z_cycle_mse_raw` / `z_cycle_mse_weighted` components. An exact inverse
  scores exactly zero.
- `scripts_joint/joint_trainer.py`: `_resolve_z_cycle_noise` (choices
  `native` | `zero`, default `native`) and `_z_cycle_roundtrip` =
  `encode(decode(z))` with the noise policy honoured and the multipliers
  restored afterwards. The epoch loop adds `zeta * z_cycle_loss` and registers
  `jpsi_z_cycle_loss` / `z_z_cycle_loss` **only when zeta > 0**, so every
  earlier config keeps a byte-identical history row.
- `scripts_joint/run_joint.py`: `--z-cycle-weight-scale` probe knob
  (getattr-guarded, so synthetic arg namespaces in existing tests keep working).
- `configs_joint/cms_Joint_runH_D3zcycle.yaml` (run key `H_D3zcycle`): extends
  `cms_Joint_runH_A2frozen.yaml` with one new term - `z_cycle_noise: native`
  and `zeta` 0.0 / 0.5 / 0.5. Stage 1 is deliberately off so the deterministic
  stage-1 reference map (the map A1/A2, the anchor and the OOD argument are
  built on) is unchanged; everything else - A1 `cycle_decoder_noise: zero`,
  honest component priors, the frozen kernel, the 20/80/80 schedule,
  beta/lamb/tau/nu_e/nu_d, the noise multipliers - is inherited untouched.
- `tests/test_d3_z_cycle.py` (12): exact-inverse zero, gradient through the
  roundtrip into the model parameters, the noise policy (default / stage
  override / reject / multipliers restored), epoch-level presence and absence of
  the component, and the shipped-config one-change contract.

**artifact-measured (preflight green; both dirs deleted per the CLAUDE.md rule).**

- dry-run `Run_H_D3zcycle_dryrun`: 830,536 freshly built shared parameters, 180
  epochs, Upsilon not opened.
- CUDA smoke `Run_H_D3zcycle_smoke`: 3 epochs, no non-finite skips; history has
  **no** z_cycle keys in stage 1 (zeta 0) and `jpsi_z_cycle_loss` 8.24e-4 ->
  7.45e-4, `z_z_cycle_loss` 2.25e-3 -> 2.29e-3 in stages 2-3.
- Full suite: **428 tests, OK (4 skipped)**.

**proposal (full run, needs sign-off).**

- train: `python scripts_joint/run_joint.py --run H_D3zcycle --device cuda`
- readout: `python scripts_joint/response_scorecard.py --run-dir
  outputs/cms_Joint/Run_H_D3zcycle --device cuda --slice-samples 40000` and
  `python scripts_joint/rescore_validation.py --checkpoint
  outputs/cms_Joint/Run_H_D3zcycle/best_model.pt --noise zero --compare-zero`
- decision: if the honest-prior J/psi latent mass KS does not fall below the
  identity 0.385 (target 0.04) and the scorecard Axis C does not improve, D3 is
  refuted like A2.4 and the next option is the architectural branch.

**update (same session) - per-epoch logging now flushes, and the D3 run is live.**

- **source-verified (logging).** The trainer's two per-epoch print sites in
  `scripts_joint/joint_trainer.py` (the eval branch and the plain branch) now pass
  `flush=True`, so a piped or redirected run streams in real time instead of
  block-buffering 4-8 KB - which is why a `Tee-Object` launch appeared to print
  nothing while the run was in fact progressing. Full suite after the change:
  **428 tests, OK (4 skipped)**.
- **artifact-measured (live run, read-only).** The D3 full run was launched
  2026-10-01 14:47:11 as `C:\Users\AhrixMarin\.conda\envs\cms\python.exe
  scripts_joint/run_joint.py --run H_D3zcycle --device cuda` (PID 3320). At
  14:48:45 `outputs/cms_Joint/Run_H_D3zcycle/` held one history row (g1, stage 1,
  eval 2.8667, score 5.5974, no z_cycle keys yet, as designed) plus the stage-1
  best/last checkpoints. There is no `logs/Run_H_D3zcycle.log` because the launch
  carried no `Tee-Object`. The running process keeps the pre-flush code in memory:
  do not restart it for logging. First `jpsi_z_cycle_loss` / `z_z_cycle_loss`
  values are expected at global epoch 21 (stage 2, zeta 0.5).
- **estimate (NOT artifact-measured - my weighting, to be revisited).** Plan-tree
  completion: **~59% goal-weighted** (tree section 1 items: simulate + Upsilon
  transfer 60%, per-event inverse 20%, stochastic decoder carrying the resolution
  65%, identity/floor-reported gauges 90%) and **~67% by leaf count** (17.95 of 27
  leaves carry a delivered artifact or a verdict, with A3 0.5, C1 0.7, C2 0.3,
  D2 0.5, D3 0.25, D4 0.7 credited). The unfinished mass is concentrated in D3
  (in flight), A5, the inverse branch B1/B2, and the gate items G-A in-domain,
  G-B and G-D criteria 3-4.

### 2026-10-01 - Session 59, paper re-framed: identification, Upsilon transfer, inversion

**proposal / decision (user instruction).** The old narrative - "a physics-cost
Neural-OT upgrade of the SWAE for detector-level J/psi simulation (Run D / Run E)" -
is dropped. The paper is re-scoped to one question with three measurable parts:

1. **RQ1 identification** - does the stochastic channel carry the detector
   resolution, or does the deterministic mean map fake it?
2. **RQ2 transfer** - does one shared mass-blind response trained on J/psi + Z
   predict the held-out Upsilon family in position *and* width?
3. **RQ3 inversion** - does the same object unfold per event with calibrated
   uncertainty?

**source-verified (artifacts).** `paper/main.tex` rewritten and compiles to 4 pages
(`pdflatex`, MiKTeX); new `paper/FRAMING.md` (tone, claim ladder, PRD assessment,
open problems, ordered plan); `paper/README.md` updated. The previous skeleton is
recoverable from git history only.

**Assessment (judgement, not artifact-measured).** Against the criteria a PRD
referee applies, the current state scores: novelty 3.5/5, completeness 2/5, external
comparison 2/5, systematics 1/5, physical impact 2/5, reproducibility 4.5/5 - i.e.
**not PRD today**. Three additions would make it a candidate: (1) one positive
physics result (Upsilon widths near the 84 MeV resolution with conditional closure,
or inversion reaching the 0.810 oracle); (2) a quantified systematics section (prior
dependence, seed/checkpoint stability, the post-unblinding meaning of the Upsilon
claim); (3) an external comparison (AUSSIE on ppzee as a ceiling; one modern
generative baseline on the locked splits). **Fallback paper**: if RQ1 succeeds and
RQ2/RQ3 do not, the paper becomes the identification-criterion + scorecard
diagnostics paper (not PRD; MLST / JINST / EPJC territory) - still a complete,
honest, reproducible result.

**Current blockers (unchanged).** P1 encoder no-op on the honest-prior J/psi mass
(latent KS 0.372 vs identity 0.385, target 0.04); P2 mean map still absorbs the
width (Y(1S) 290-299 MeV vs 84); P3 conditional closure 44-87x floor; P4 inversion
0.981-0.983 vs oracle 0.810; P5 the A0.4 key is not wired into trainer selection;
P6 paper engineering (setup section, figures/tables dirs, Results, Conclusion).

**Next.** The D3 z-space cycle run is in flight (started 14:47:11; first z_cycle
readings at global epoch 21). Its outcome chooses the paper branch at milestone M1;
milestones M2 (one positive result) and M3 (systematics + comparison) are what move
the target from a diagnostics paper to a PRD candidate.

### 2026-10-01 - Session 60, S5 systematics: what survives the obvious variations

**source-verified (new tool).** `scripts_joint/systematics_study.py` +
`tests/test_systematics_study.py` (15). Four axes per headline claim: **prior**
(two Upsilon evaluation priors, read from existing decoded files so the checkpoint
and the noise multipliers are identical), **checkpoint** (the retained checkpoints
of the arm), **seed** (re-decode at three subsample seeds), and a **bootstrap**
floor; plus the required-resolution shift between the legacy and honest J/psi
priors. Verdicts are robust/fragile against each claim's own tolerance (median
30 MeV, drift 40 MeV, width 30 MeV, R band 0.225, slice ratio 3).

**artifact-measured (`outputs/cms_Joint/systematics/`, 400k Upsilon events per seed).**

- **Peak positions are prior-robust and seed-robust.** 1S median-CMS 20.95 MeV (raw
  prior) vs 20.82 MeV (continuum-reweighted); seeds 20260822/23/24 give
  +19.4/+21.6/+20.9 MeV (half-range 1.09 MeV); bootstrap floor 0.41-0.43 MeV.
  2S/3S prior half-ranges 0.25/0.10 MeV.
- **But checkpoint-fragile**: half-range 84.8/89.0/91.5 MeV across
  stage2-best (g35) / stage3-best (g110) / last (g180). The **selection rule, not
  the evaluation noise, is the dominant systematic** for every transfer claim -
  the quantitative form of the A0.3/A0.4 argument.
- **Width**: 298 MeV with a 35.6 MeV checkpoint spread; the 84 MeV claim fails on
  every axis.
- **In-domain identification is seed-robust**: jpsi within/required 1.083 +/- 0.084,
  z 0.848 +/- 0.028 against a 0.225 tolerance; the scorecard key selects the same
  checkpoint at all three seeds.
- **Conditional closure is not a sampling artefact**: jpsi slice ratio
  38.3 +/- 7.9 against a gate of 3.
- **The prior axis moves the continuum, not the peak**: 8.5-9.25/CMS = 1.57 (raw)
  vs 0.948 (reweighted) for the same checkpoint, while the 1S peak moves 0.06 MeV -
  consistent with the Session 51c correction.
- **Required-resolution shift**: legacy jpsi 12.2 MeV vs honest 30.0 MeV robust
  (prior robust 27.4 vs 0.65 MeV), i.e. honest/legacy = **2.46x**. Any in-domain
  ratio is undefined without its prior.
- Full suite: **443 tests, OK (4 skipped)**.

**consequence for the paper.** (i) quote the prior with every in-domain ratio;
(ii) report the checkpoint spread as the dominant uncertainty on the Upsilon claim,
and tie the analysis to a pre-declared selection rule (the A0.4 key); (iii) quote
~1 MeV as the evaluation-noise floor; (iv) the conditional-closure failure is robust.
A training-seed study (new runs) remains the only untested axis.

### 2026-10-01 - Session 61, paper re-laid out: single column, Nature-style

**source-verified.** `paper/main.tex` rewritten as a single-column Nature-style
working draft: unnumbered bold flush-left headings (`secnumdepth` 0 plus custom
`\@startsection`), superscript numbered citations (`natbib` with
`super,sort&compress` and `unsrtnat`), A4/11pt/2.6 cm margins, Methods at the end,
then Data availability, Code availability, Acknowledgements, Author contributions
and Competing interests. New content relative to the previous skeleton: a
related-work paragraph folded into the Introduction (OmniFold, AUSSIE,
Wasserstein-loss unfolding, the generative-simulation line), a systematics-budget
table from S5, and the identification-scorecard table. Compiles with
`pdflatex` + `bibtex` to **7 pages**; page renders were verified with `pdftoppm`.
Two-column conversion is deliberately postponed to submission time so the draft
stays readable. `paper/README.md` and `paper/FRAMING.md` updated (section 2b).

### 2026-10-01 - Session 62, D3 early readout and the pre-declared M1 decision rule

**artifact-measured (history.json only, no new GPU work; 15:45).** Matched-epoch comparison of
the running D3 arm (`Run_H_D3zcycle`) against its control `Run_H_A2frozen`, honest-prior J/psi
latent mass KS (raw; identity 0.385, D3 target 0.04, A2frozen final 0.372):

| global epoch | D3 | A2frozen | D3 jpsi_z_cycle_loss |
|---|---|---|---|
| 21 | 0.504 | 0.531 | 3.82e-2 |
| 25 | 0.504 | 0.462 | 1.66e-2 |
| 30 | 0.481 | 0.445 | 7.86e-3 |
| 35 | 0.414 | 0.401 | 6.04e-3 |

Stage 1 (epochs 1-20) is identical by construction (zeta = 0, so the stage-1 reference map is
unchanged). From the first stochastic epoch the cycle term is optimised hard - a factor 6 in 15
epochs - while the encoder gauge tracks the control arm and sits slightly behind it at every
matched epoch. That is an early warning, not a verdict; the decisive checkpoints are the stage-2
and stage-3 best/last (g100/g110/g180).

**hypothesis (if the pattern persists):** the z-space cycle can be satisfied without unfolding -
the composition encode(decode(z)) can be made near-identity on the decoded support without the
encoder contracting the detector resolution out of x. Testable by inspecting the decoded support
and the per-coordinate z residuals at the stage-2 best.

**proposal (pre-declared M1 rule, so the call is not made after the fact).** D3 is **positive** if
the honest-prior J/psi latent mass KS at the stage-2/3 best checkpoints falls clearly below the
A2frozen band (toward the 0.279 best mass-only contraction) with a downward trend, while scorecard
Axis R does not regress and the Upsilon medians stay within +/-30 MeV; **negative** if it stays
inside the 0.36-0.74 band seen in the earlier probes. Negative -> the mean-map contraction (D3b)
becomes the primary fix and the paper takes the diagnostics branch.

### 2026-10-01 - Session 63, figures staged in PaperPlots/images and the paper filled in

**source-verified.** New figure directory `PaperPlots/images/` (8 PNGs) wired into the
paper with `\graphicspath{{../PaperPlots/images/}}`; the paper now compiles to **12
pages** with all eight figures embedded (`pdflatex`, MiKTeX; renders verified with
`pdftoppm`):

- in-domain closure: `jpsi_mass_ratio_rune.png`, `z_mass_ratio_rune.png` - generated
  for **Run E** with `plot_joint_paperstyle.py`, which gained a `data/legacy/` prior
  fallback so the legacy arms plot without touching `data/`.
- identification: `identification_scorecard.png` (12 arms / 30 checkpoints),
  `peak_shift_vs_tail.png` (the late shift is inert under tail noise),
  `noise_budget.png` (14 checkpoints).
- transfer: `upsilon_transfer.png` (continuum-reweighted prior, three checkpoints),
  `systematics_budget.png` (S5 prior and seed axes).
- inversion: `per_event_closure.png` - new read-only script
  `scripts_joint/plot_per_event_closure.py` reading the retained ppzee payloads
  (identity 1.000, upstream 3.330, ours 0.981-0.983, oracle 0.810; pull std 20.7,
  1-sigma coverage 5.8% for the trained checkpoints, degenerate for the deterministic
  one).

**content filled in the paper (all [measured]).** Real per-arm scorecard numbers in
Table 2 (legacy arms $Z$ ratio 0.038-0.088 and $J/\psi$ slice ratios 6.9-13.8; A2
arms pass the response gate at 1.09-1.22 with slice ratios 44.6/75.1); locked-split
event counts in Methods (2,899,563 / 71,659 $J/\psi$; 4,207,696 / 800,000 $Z$;
honest group 501,317 / 401,718 truth-level); model size (830,710 parameters, 415,355
per direction, 3.32 MB float32).

Full suite unchanged at 443 tests. No checkpoint, data file or existing output was
modified; the new files are the figure directory, one plotting script and the paper
edits.

### 2026-10-01 - Session 64, D3 (z-space cycle) read out: refuted; M1 falls back to D3b

**artifact-measured.** `Run_H_D3zcycle` finished the full 20/80/80 schedule (180/180 epochs)
in 4 h 42 min (14:47:11-19:28:59); `joint_evaluation.json` written 19:31:32. Selected
checkpoint = stage-3 epoch 45 (`global_epoch` 145, selection score 3.9318). `grad_norm`
456-512, `nonfinite_gradient_skips` 0 in every epoch. The z-cycle loss fell 3.82e-2 ->
5.0e-3 in the first 15 stochastic epochs and then stayed flat for the remaining 80: the
term is satisfied, the gauge does not move.

**Verdict: NEGATIVE, by the rule pre-declared in Session 62** (positive required the
honest-prior J/psi latent mass KS at the stage-2/3 best/last checkpoints to fall clearly
below the A2frozen band toward 0.279 while Axis R held; negative if it stays inside the
0.36-0.74 band). Decisive numbers, zero-noise locked validation:

| checkpoint | jpsi latent mass KS | identity | `latent_mass_ks_vs_identity` |
|---|---|---|---|
| D3 ge 145 (best) | 0.4325 | 0.3682 | **1.1766** |
| D3 ge 180 (last) | 0.4058 | 0.3682 | - |
| A2frozen ge 110 / ge 180 | 0.3716 / 0.372 | - | 1.0585 (native) |

Matched at the *same* operating point (native core 1.0 / tail 0.5, `rescore_validation.py`
on D3 ge 145 vs `Run_H_A2frozen/joint_evaluation_native_stage3.json` ge 110): jpsi latent
`vs_identity` **1.0656 vs 1.0585** - a 0.7% difference, i.e. the z-cycle term moved the
encoder not at all.

**artifact-measured, A0.4 scorecard (`outputs/cms_Joint/scorecard/Run_H_D3zcycle/scorecard.json`,
`--all-checkpoints`).**

| checkpoint | R jpsi within/required | R jpsi noise var fraction | R z within/required | C jpsi native median/max slice ratio |
|---|---|---|---|---|
| D3 ge 145 (best_model) | 1.0107 | 0.01505 | 0.8081 | 49.03 / 77.71 |
| D3 ge 180 (last_model) | 1.0063 | 0.01562 | 0.8190 | 23.06 / 44.40 |
| A2frozen ge 110 (best) | 1.0867 | 0.01577 | 0.8286 | 44.59 / 65.41 |
| A2frozen ge 180 (last) | 0.9831 | 0.01514 | 0.8132 | 43.98 / 52.60 |

Axis R: indistinguishable from the control (mean-map variance fraction 0.985 in all four;
the learned noise channel never carries more than ~1.6% of the variance). The key calls D3
`identified` and rejects both stage-1 deterministic checkpoints as degenerate - the R gate
works, but nothing D3 did changed the R numbers. Axis C: gates are median <= 3.0 and
max <= 10.0, so **every checkpoint in both arms fails by roughly an order of magnitude**;
D3's last_model value (23.06) is the lowest of the four but its sibling is the worst (49.03),
so that spread is run-to-run scatter, not a signal.

**The decoder side remains the asset.** D3 ge 145, jpsi direct mass KS `vs_identity` 0.1109
at zero noise and 0.3017 at native; z direct 0.7979 native; direct width relative error
`vs_identity` 0.2311 (jpsi) and 0.8297 (z).

**hypothesis confirmed.** The Session-62 hypothesis - the z-space cycle is satisfiable
without unfolding - is now measured, not assumed: a 7x reduction in `encode(decode(z)) - z`
left the x -> z gauge at 1.18x the identity gap (worse than no model at all) and left Axis R
byte-for-byte at the control's level.

**proposal (pre-declared fallback, needs sign-off).** D3b becomes the primary fix: contract
the mean map (penalise the variance of the deterministic `mean_delta`, or parameterise the
conditional mean and learn only its shape), one variable, ~4 h for a 20/80/80 arm. The
architectural option (a flexible conditional density `p(eps|z)`) is the untested alternative.
The paper takes the diagnostics branch: D3 is the fourth documented refutation (after A2.2,
A2.4 and beta x5), and it is the sharpest of them because it closes the 'a missing loss term
would fix it' explanation.

**Files (all new, nothing existing modified).** `outputs/cms_Joint/scorecard/Run_H_D3zcycle/`,
`logs/scorecard_Run_H_D3zcycle.log` (UTF-16, `Tee-Object` default - read it with
`Get-Content`, not the file reader), `outputs/cms_Joint/Run_H_D3zcycle/validation_native_best.json`.

**correction (same session, after reading the probe code) - the Axis-R variance split is NOT F1 evidence.**
`fixed_z_noise_budget.py` draws its fixed-z probe from the prior *file* without applying the region
mass window, and normalises the split by the variance over that probe. For jpsi the probe has
`prior_mass_std_gev` 0.2647 GeV while the run's own locked region split has 0.0148 GeV
(`reference_resolution`), an 18x difference. A noise share of ~1.5% is roughly what a *correct*
model would also give on a 265 MeV-wide probe with a 34 MeV resolution, so the 0.985 mean-map
fraction quoted above must not be read as 'the mean map absorbs the resolution'. The load-bearing
measurement is the zero-vs-native width of the decoded region sample, D3 ge 145, jpsi z->x mass
`width_rel_error`: **0.118 at zero noise** (`vs_identity` 0.2405) vs **0.753 at native**
(`vs_identity` **1.5996**). With its own response channel switched on, the decoder is 60% worse
than the identity map on the width; with it off, it removes 76% of the identity gap.

**Consequence that stands on its own.** Every headline in-domain closure number in this project
(Run E J/psi KS 0.017, the D3 `joint_evaluation.json`) is evaluated at **zero noise** - the
config's `final_evaluation.noise_multipliers` are all 0.0, and the selected checkpoints are
unanimously the zero-noise ones. They therefore describe the *deterministic mean map with the
response channel off*, and at the native operating point the same checkpoint is worse than
identity on the width. Reporting closure only at zero noise is not defensible and this must
change before any further claim.

**to verify (proposal).** Confirm which z sample the Axis-R probe loads for jpsi (prior file
without the region window vs the locked region split). If it is the unwindowed file, the R axis
split and the `min_noise_variance_fraction` gate need re-deriving on the region sample.

### 2026-10-01 - Session 65, the root cause is nailed down and D3b is built (config-only)

**source-verified (why selection could never see this).** `joint_trainer.validate_joint` scores every
checkpoint with all four noise multipliers at zero and says so: scoring at the stage's training noise
"makes stage scores incomparable and hides deterministic-map degeneration". The A0.3 fixed-z noise
budget is merged into the region metrics but is **diagnostic only - it never enters the selection
score**. So both the training curriculum and the checkpoint key were blind to the response channel by
construction, and a deterministic map can match any marginal (Monge), so the key could only ever
reward the degenerate map.

**artifact-measured (stage-wise, zero noise, J/psi z->x, `rescore_validation.py`).** This is the
cleanest statement of the mechanism the project has:

| checkpoint | width_rel_error | vs_identity | direct mass KS | latent KS vs_identity |
|---|---|---|---|---|
| stage-1 ge20 (no noise at all) | **0.0385** | **0.0735** | 0.0221 | 1.237 |
| stage-2 best ge35 | 0.2315 | 0.4729 | 0.1469 | 1.101 |
| stage-3 best ge145 (final) | 0.1181 | 0.2405 | 0.0474 | 1.177 |
| stage-3 ge145, native noise | **0.7533** | **1.5996** | 0.1227 | 1.066 |

A purely deterministic map closes the decoded J/psi width to 3.9% in 20 epochs, and 125 epochs of
stochastic training make the zero-noise marginal **worse**, not better. The deterministic map
manufactures the data width (prior robust 0.65 MeV -> decoded 31.4 MeV against 28.1 MeV of data); the
frozen kernel then adds ~34 MeV on top, giving sqrt(31^2+34^2) ~ 49 MeV, i.e. 1.75x too wide and
worse than the identity map on the width at the model's own operating point.

**The arithmetic that makes a fix feasible (artifact-measured).** With the frozen kernel the within-z
decoded width is already ~30.2 MeV (robust) against ~30.0 MeV of data width and a ~0.65 MeV prior.
The noise channel alone therefore SATURATES the data width, so the marginal can only close if the mean
map contributes almost no width. Turning the channel on where the decision is made should make the
existing marginal objective fix the attribution by itself - no new penalty term, no new hyperparameter.

**proposal -> implemented: `Run_H_D3b`** (`configs_joint/cms_Joint_runH_D3b.yaml`, key `H_D3b`,
extends `cms_Joint_runH_A2frozen.yaml`). One intervention, two config edits, documented in the header:
(1) stage-1 noise 0/0 -> 1.0/0.25, the channel stage 2 already uses; (2) validation and
final-evaluation noise -> decoder 1.0/0.25 with the encoder left at 0/0 so the latent numbers keep the
convention of every earlier arm (identity 0.385, target 0.04). Splitting the two edits is not
meaningful: changing only the schedule leaves selection rewarding the degenerate checkpoints, changing
only the scoring leaves stage 1 forcing the construction.

**Naming warning (source-verified).** Stage 1 keeps the inherited name
`runH_stage1_deterministic_warmup` because `_deep_merge` merges stage lists by name and **appends**
unknown names - a rename would silently produce a four-stage schedule. In this arm stage 1 is
stochastic; do not use its checkpoint as a deterministic reference map. `tests/test_d3b_response_channel.py`
(6 tests) pins the stage names, the one-intervention diff against A2frozen, the stage-1 channel and the
scored-map policy, and asserts the baseline still scores the deterministic map.

**Pre-declared readout (before the run).** Primary, native-noise J/psi z->x at the selected checkpoint:
`width_rel_error <= 0.15` (A2frozen final 0.753) and `direct_mass_ks_vs_identity <= 0.5` (0.302).
Secondary: A0.4 Axis C native slice median ratio toward the 3.0 gate (23-49 now). OOD: Upsilon widths
toward 84 MeV (290-299 now). **Mechanism check:** at zero noise the decoded width must now be near the
PRIOR width (~0.65 MeV), not the data's ~30 MeV - the signature that the mean map stopped carrying the
resolution; if the native numbers improve while this fails, the gain is not identification. Refuted if
the native width stays ~1.7x and Axis C does not move: then the fix has to be structural (explicit
frozen smearing channel plus smooth-correction-only mean map).

**artifact-measured (preflight, both exit 0).** dry-run `Run_H_D3b_dryrun`: 180 epochs, 830,536
parameters, Upsilon not opened. Smoke `Run_H_D3b_smoke`: 3 epochs, no non-finite skips, and every stage
logs `val_noise=enc0/0,dec1/0.25` - the new policy is live in stage 1 as well as 2-3. Resolved config
confirms stage core/tail 1.0/0.25, 1.0/0.25, 1.0/0.5 and the scored map decoder 1.0/0.25.

### 2026-10-01 - Session 66, HPC3 deployment plan and the post-run git-sync design

**proposal, not executed on HPC3.** New directory `deploy/hpc3/`: `README.md` (runbook),
`env_create.sh`, `stage_data.sh`, `train_joint.sbatch`, `sync_after_run.py`, `pull_run.ps1`.
Nothing has run on the cluster; section 9 of the runbook is the list of unknowns to close on
the first interactive session.

**artifact-measured (what the workload costs, from `outputs/cms_Joint/Run_H/history.json`).**
Run H is 180 epochs / 13,551 s = **3.76 h**, median 74.4 s/epoch (min 66.1, max 83.2), peak CUDA
**3.21 GB reserved / 2.11 GB allocated**. Source-verified: there is no `DistributedDataParallel`,
`torchrun`, `nccl`, `world_size` or `multiprocessing` anywhere under `scripts_joint/`, so the whole
request is `--gres=gpu:1`. Every HPC3 GPU (smallest 16 GB V100) is >=5x oversized. The config's
`cuda_memory_limit_gb: 11` needs no edit because `configure_cuda_memory_limit` renormalises it to a
fraction of the device total.

**source-verified (HPC3 policy that shapes the plan).** `gpu` needs a Slurm account ending in `gpu`;
"There are NO personal GPU accounts" and the PI must specifically request GPU-hours, so the GPU
allocation is the blocker, not the code (https://rcic.uci.edu/slurm/slurm.html,
https://rcic.uci.edu/about/allocations.html). `free-gpu` is preemptible and uncharged. "Do not run
Slurm jobs in your $HOME. Instead, use your DFS storage /pub/UCInetID"
(https://rcic.uci.edu/account/acceptable-use.html) -> the checkout, data, env prefix and outputs all
live under `/pub/$USER`. Login nodes forbid conda installs and multi-GB downloads, so the 2.1 GB CMS
file is fetched inside a compute job. Charging is 34 units/GPU-hour (32 GPU + 2 CPU).

**source-verified (GPU arch pin).** V100 (sm_70) works only because the environment pins
torch 2.12.0+cu126: CUDA 12.6 was the last build publishing Volta kernels and is removed from CD in
PyTorch 2.15 (https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).
Do not raise the torch pin past 2.14 while training on V100 nodes.

**source-verified (why the CMS file is not shipped).** `data/Run2012BC_DoubleMuParked_Muons.root` is
byte-identical to CERN Open Data record 12341 (CC0-1.0, DOI 10.7483/OPENDATA.CMS.LVG5.QT81): local size
2,244,449,133 B and a HEAD on
https://opendata.cern.ch/record/12341/files/Run2012BC_DoubleMuParked_Muons.root returns 200 with the
same `Content-Length`. `stage_data.sh` re-checks the size and the record's `adler32:1fa61aca`.

**source-verified (the region cache is machine-bound).** The cache key hashes a record containing the
**absolute path, `mtime_ns` and size** of the CMS ROOT file and the prior HDF5 (see any
`.region_cache/*/selected_split_*.json` sidecar), and the key also includes `num_samples`. So the 2.3 GB
cache must not be transferred (a different path can never hit) and dry-run/smoke
(`num_samples: 1000`) do not warm the cache a full run (`num_samples: null`) will use. The full run
builds its own cache inside its own time limit.

**artifact-measured (the git-sync trap this session exists to close).** The repo tracks 3,171 files,
**2,669 of them under `outputs/` and zero of them JSON**: `.gitignore` line `*.json` excludes every
metric file a run produces (`history.json`, `joint_evaluation.json`, `provenance.json`,
`joint_split_manifest.json`, `config.resolved.json`). A hand-rolled "commit the run" therefore either
records plots with no numbers behind them or drags 77 MB of checkpoints plus 104 MB of decoded HDF5
into git forever, and it fails silently. `sync_after_run.py` instead force-adds a whitelist inside one
run directory only: 99 of Run H's 109 files = **8.8 MB** (65 PNG 7.94 MB, 21 JSON 0.72 MB, 6 CSV, 4 PDF,
3 MD), skipping the 8 `.pt` and the 2 `.hdf5` by type plus anything over 8 MB. Every sync also writes
`<run>/sync_manifest.json`, the durable record of job id, node, start SHA, epoch count, wall time, peak
VRAM and per-file sha256.

**artifact-measured (sync script tested locally, throwaway bare remote seeded with this `.gitignore`).**
`--dry-run` exits 4 and writes nothing; `--no-push` commits exactly the whitelist; a push over a remote
that had advanced on an unrelated file rebased and pushed with `rev-list --merges --count == 0`; a push
over a remote that had edited the same `history.json` exited 2 with `rebase --abort` clean and the
commit preserved locally. Not yet exercised: real network, real credentials, real SLURM env.
`pull_run.ps1` was tested in dry-run and its dirty-tree guard was fixed (it tested output truthiness
instead of `$LASTEXITCODE`).

**Files (all new, nothing existing modified).** `deploy/hpc3/{README.md,env_create.sh,stage_data.sh,train_joint.sbatch,sync_after_run.py,pull_run.ps1,vscode_connect.ps1}`.
`README.md` is matched by `.gitignore` line `*.md` and needs `git add -f`; the six scripts are
tracked normally. `.gitignore` itself was deliberately left unmodified; adding `!outputs/**/*.json`
next to the existing `!outputs/**/*.md` is the proposal that would remove the trap for hand commits.

**source-verified (VS Code on HPC3 is not a plain Remote-SSH).** "We do not allow running VSCode on
login nodes ... Any VSCode server instances will be removed from login nodes without a notice", and
the supported route is "the only accepted method to run VSCode on HPC3": `sbatch
/opt/rcic/scripts/vscode-sshd.sh`, which starts a per-user sshd on a compute node, reached by
ProxyJump through `hpc3.rcic.uci.edu` (https://rcic.uci.edu/account/login.html#using-vscode). The
sshd picks a free port on whatever node the job lands on, so node name and port change every job and
the local `~/.ssh/config` must be rewritten each time. Key-based auth with a >=10 character
passphrase is mandatory (https://rcic.uci.edu/account/generate-ssh-keys.html), and an un-cancelled
job keeps charging because the `standard` default limit is 2 days. `vscode_connect.ps1` automates
submit -> wait -> parse `~/vscode-sshd-<jobid>.out` -> rewrite a delimited `Host hpc3-*` block in
`~/.ssh/config`, plus `-Stop` (scancel) and `-Status`; `-SelfTest` (exit 0, artifact-measured locally)
covers the parsing, idempotent block replacement that preserves unrelated `Host` stanzas, and a
BOM-free write (PS 5.1 `Set-Content -Encoding utf8` emits a BOM that breaks ssh's config parser).
Not yet tested end to end against the cluster.

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

### 2026-10-04 - Sessions 67-70, D3b: noise-from-the-start works; the residual is the kernel

**artifact-measured (new run).** `Run_H_D3b` (`configs_joint/cms_Joint_runH_D3b.yaml`,
`python scripts_joint/run_joint.py --run H_D3b --device cuda`) finished the full
20/80/80 schedule, 180/180 epochs, exit 0, Upsilon never opened. Preflight (dry-run
+ smoke) re-run after the config edit and green, then cleaned with
`clean_preflight.py --apply`. Stage selection scores 4.337 / 4.551 / 3.886; selected
checkpoint = global epoch 180 (stage-3 last). Final evaluation at decoder 1.0/0.25,
encoder 0/0.

**artifact-measured (the mechanism check passed).** Zero-noise decoded J/psi mass
spread is 17.6-19.3 MeV at every scored checkpoint (prior std 14.93, CMS x std 28.03,
robust prior width 0.65 MeV), i.e. 0.63-0.69x the data width and 1.18-1.29x the prior.
The A2frozen/D3zcycle stage-1 construction (29.2 MeV = 1.04x data) does **not** appear.
Measured with `scripts_joint/d3b_readout_probe.py --device cuda`; payload
`outputs/cms_Joint/d3b_readout_probe/d3b_final/d3b_readout.json`, write-up
`outputs/cms_Joint/d3b_readout_probe/REPORT.md` (with the pre-D3b baseline table).

**artifact-measured (J/psi z->x at the native operating point).** decoded mass std
46.7 -> 38.8 MeV (1.66x -> 1.39x data); `width_rel_error` 0.753 -> 0.152;
`width_rel_error_vs_identity` 1.5996 -> 0.322; direct mass KS 0.1134 -> 0.0725;
W1 0.0135 -> 0.0071 GeV. Native selection score 6.047 -> 3.963 (rescore:
zero 5.258, native 3.963).

**artifact-measured (A0.4 scorecard, `outputs/cms_Joint/scorecard/Run_H_D3b/`).** All
eight checkpoints are `identified`, **including stage-1** (within/required 1.05-1.12)
where A2frozen's stage-1 is rejected as degenerate. Axis C native slice median/max
18.6-23.4 / 33.6-51.4 against gates 3/10 (A2frozen 44.6 / 65.4). Axis C still fails.

**artifact-measured (Upsilon transfer, continuum-reweighted prior, 1M events).**
Inclusive mass KS 0.0375, W1 0.0372 GeV, mean +29 MeV. Per state at native: 1S/2S/3S
median-CMS **+39.8 / +27.9 / +30.9 MeV** (A2frozen stage-3 best: +21.6/+2.9/+6.3) and
std 236 / 278 / 290 MeV (first signal state); the added width over the prior is
209-221 MeV and is state-independent, as is the +15.5 MeV mean bias. Per-stage medians
(stage-1 / stage-2 / stage-3 best): +39/+21/+20, +63/+44/+45, +39/+26/+30 - the
position bias is already in **stage-1**. Artifacts:
`outputs/cms_Joint/Run_H_D3b/upsilon_transfer_continuumReweighted_best_model/`,
`upsilon_peak_medians_native/`, `upsilon_peak_medians_stages/`,
`upsilon_arm_comparison/`, plus 18 paperstyle plots under `quantitative_z_to_x/`.

**artifact-measured (zero vs native noise, same events; Upsilon).** Zero-noise 1S std
214 MeV vs native 235; medians +65.5 vs +39.5. The channel **centres** the peak and
widens it slightly; the width is already in the mean map at zero noise (214 vs prior
110). Same test on A2frozen: 280 -> 296, medians +18.2 -> +20.5. Driver
`scripts_joint/upsilon_noise_channel_test.py`; artifacts `upsilon_noise_test_d3b/`,
`upsilon_noise_test_a2frozen/`.

**artifact-measured (paperstyle, both noise conditions).**
`plot_joint_paperstyle.py` gained `--noise native|zero|both` (default native, old
behaviour); D3b produced 44 plots at
`outputs/cms_Joint/Run_H_D3b/paperstyle_best_model/{native,zero}/{jpsi,z}/`. At native
the J/psi x-space mass ratios sit at 1 (W2 2.4e-7 data cycle, 1.2e-6 decoded prior);
at zero noise the Z peak overshoots the data (ratio ~1.2, W2 0.926/0.815 vs native
0.429/0.332), which is the visual form of "zero-noise closure is not defensible".

**artifact-measured (gauge / negative-control study).** New read-only
`scripts_joint/marginal_vs_scoring_rule.py`: D3b best_model decoded at zero and native
noise, scored against the locked CMS x sample with mass W1, mass KS, an energy distance
and a Gaussian-kernel score. **All four gauges move together** (J/psi KS 0.232 -> 0.075,
energy distance +0.00358 -> +0.00108; Z 0.070 -> 0.048, +0.101 -> +0.080), so the naive
contrast is confounded. The clean control is the **shuffled** decode (same values,
z -> x assignment permuted): every gauge reproduces bit-for-bit (J/psi KS 0.23198,
energy distance +0.00358; Z 0.06998, +0.10057). **Consequence: no marginal gauge can
measure the map, because a permutation leaves it exactly invariant.** Payload
`outputs/cms_Joint/d3b_readout_probe/identity_vs_conditional{,_shuffle}/`.

**source-verified (literature).** Four parallel deep reads of the arXiv literature are
consolidated in `docs/literature_2026-10-04.md`: (i) our "learned sigma collapses"
failure is published in the paired setting and traced to mean-predicting losses
(arXiv:2006.06685), with the same paper measuring "too narrow" posteriors after the fix;
(ii) identifiability of the split is **not** stated anywhere, but arXiv:2603.20903 gives
uniqueness **iff** an injectivity condition on the marginal forward map, which fails for
a non-injective (smearing/collapsed) kernel - so even the truth-level marginal is not
identified; (iii) the cleanest demonstration that marginal matching leaves the noise
width free is arXiv:2411.02495 (unpaired coupling = OT coupling; posterior width set by
the SDE noise scale, verified over four orders of magnitude); (iv) **no** paper does an
unpaired, real-data, zero-shot transfer to an excluded resonance with a position/width
decomposition - our result appears new; (v) primary-source CMS numbers: Y(1S) resolution
96 +/- 2 MeV (all eta) and 69 +/- 2 MeV (|eta| < 1), J/psi ~1% (~31 MeV), line shape
Crystal Ball + FSR tail, response tracker-dominated below pT 200 GeV - our 84 MeV is the
acceptance-weighted interpolation and should be reported as such, while our 235-300 MeV
is 2.4-3.6x every primary-source value.

**correction (citation hygiene).** `docs/project_tree.md` cites arXiv:1803.01718/1803.01720
as unfolding reviews; both are unrelated mathematics papers. Use Blobel hep-ex/0208022,
Kuusela & Panaretos 1401.8274, Volobouev 1408.6500, Stanley-Patil-Kuusela 2111.01091.
The scoring-rule review also corrects three ids used in earlier notes (Pacchiardi & Dutta
is 2205.15784; Gneiting & Katzfuss has no arXiv version; Wang-Blei-Cunningham is NeurIPS
2021 proceedings only).

**proposal (next, one controlled variable).** The residual 1.39x is now attributable to
the kernel, not the map: at the D3b best checkpoint the two quadrature terms are 17.6 MeV
(mean map) and ~34 MeV (kernel) against 28.0 MeV of data, and the kernel's own calibration
already records `achieved/target` 1.21 for J/psi. Before rescaling it, pin the calibration
target (28.1 MeV std vs ~24 MeV quadrature requirement vs 84 MeV fitted peak width - the
three differ by 1.2-2.5x). **[CORRECTED 2026-10-04: the 84 MeV number is the Upsilon(1S)
reference, not a J/psi number - no artifact uses it for J/psi. The J/psi readings are 28.06
(data std), 23.81 (std quadrature) and 30.02 (robust quadrature). See Session 74 and
`docs/calibration_target_2026-10-04.md`.]** Two untried, literature-backed mechanisms: a strictly proper
scoring rule on the conditional (arXiv:2205.15784), and an architectural constraint that
stops the mean map carrying width. Not started; no training authorisation.

### 2026-10-04 - Session 71, Phase 0 of the scoring-rule programme: what is identifiable

**approval.** The user approved the plan (an experiment design + paper outline, folded
into `docs/literature_2026-10-04.md` is the evidence base). Phase 0 is a CPU-only toy
study; no training authorisation was given for Phase 1.

**source-verified + artifact-measured (the instrument was wrong three times before it was
right).** `scripts_joint/scoring_rules.py` now ships exactly one rule, the Gaussian log
score `(x - mu)^2 / (2 sigma^2) + log sigma` (lower is better, strictly proper, minimised
at `mu -> x` and `sigma -> |x - mu|`). Rejected after measurement:
- a **Gauss-kernel score** with plug-in bandwidth is a *similarity*, not a divergence: a
  too-narrow model (c = 0.2) scored 0.216 against 0.286 for the correct model, and a
  shuffled target scored 0.097, i.e. its minimum is reached by *disagreeing* with the data;
- the **energy score** `E|X~ - x| - (1/2)E|X~ - X~'|` has the right minimum over a scale
  family (0.567 at c = sigma vs 0.699/0.657 either side) but returns **exactly 0.0 when the
  draws are identical** - better than the correct value - so minimising it drives
  `sigma -> 0`. Any rule whose within-model term collapses with the spread has this trap;
- the energy score's "second independent observation" variant has its minimum at
  `c = sigma/sqrt(2)`, an underdispersion bias.

**artifact-measured (the identifiability limit, `scripts_joint/single_observation_limits.py`,
`outputs/cms_Joint/single_observation_limits/`).** With ONE observation per event and a mean
map able to hit that observation, the argmin over `sigma_hat` is 0.01 for **all three**
rules - log score, CRPS and energy score - because `mu = x` leaves only `log sigma`. With a
correctly specified mean the same rules recover the truth exactly (argmin 1.0). **Repeated
observations do not rescue it** (2 obs/event with a flexible mean still gives argmin 0.01);
only a correctly specified or structurally constrained mean does. Consequence: the
degeneracy is in the **mean map**, not in the loss, and no scoring rule can fix it.

**artifact-measured (1-D toy with planted ground truth,
`scripts_joint/toy_identifiability.py`, `outputs/cms_Joint/toy_identifiability/`, 5 seeds,
4000 steps).** Recovered spread as the median ratio `k_hat/k_true` over the central domain:
| arm | median ratio | in band [0.75, 1.25] |
|---|---|---|
| flexible mean + marginal/reconstruction loss (the objective we train with) | **0.314 +- 0.000** | **0/5** |
| flexible mean + log score | **1.033 +- 0.020** | **5/5** |
| fixed mean + log score | 1.002 +- 0.011 | 5/5 |
So the marginal objective alone under-spreads by 3x on a problem where the truth is known,
and the log score recovers it even with a flexible mean map - because that toy network is far
too small to interpolate 512 events. The real model has 830k parameters against 3.0M/4.2M
training events, and the degenerate regime is a question of capacity versus event count.

**source-verified (wiring, all opt-in).** `scripts/loss.py` gained
`DualSpaceFeatureOTLoss.energy_score_loss` (log score over the decoded conditional of a
batch subset, per-event moments from the draws); `joint_trainer.train_joint_epoch` gained
`kappa` (per stage, mirroring `zeta`), `scoring_rule.{draws,score_batch}` config keys, the
draws decoded from the same `z` slice as the realised `x`, and component registration only
when `kappa > 0` so every earlier config keeps a byte-identical history row. Tests:
`tests/test_scoring_rules.py` (19) and `tests/test_energy_score_term.py` (10), including a
regression pin for the removed kernel score and a pin for the interpolating-mean limit.
Full suite **478 tests, OK (4 skipped)**.

**proposal (Phase 1, needs authorisation).** Add `kappa > 0` to the D3b configuration as the
single controlled change, with an explicit monitor on the learned `core_sigma`/`tail_sigma`:
the stage-1 checkpoint at g20 (about 25 minutes) decides whether the log score drives the
amplitude up (proceed) or down (the degenerate regime; stop). The frozen physics kernel is
what makes the mean map structurally unable to absorb the width, so it stays on. No run
started.

### 2026-10-04 - Session 72, Phase 1 run and readout: the log score is refuted

**artifact-measured (`Run_H_SR`, 180/180 epochs, exit 0, 4.6 h, 92.4 s/epoch = 1.25x D3b).**
Config `configs_joint/cms_Joint_runH_scoringRule.yaml`. Full write-up:
`outputs/cms_Joint/Run_H_SR/VERDICT.md`. Versus D3b the arm changed two coupled things as
one intervention - the decoder amplitude was **unfrozen** (`freeze_noise_amplitude: false`,
so the kernel becomes a floor) and the strictly proper **Gaussian log score** was added at
`kappa = 0.004` on a 2048-event subset with 2 decodes per event. Edit (1) was forced: the
first preflight kept the freeze, and inspecting that probe showed `freeze_noise_amplitude:
true` **discards the learned scale parameters** (byte-identical to D3b), so the score term
would have had no amplitude to act on.

**artifact-measured (pre-declared readout; four of five criteria fail).**
| criterion | target | measured | verdict |
|---|---|---|---|
| zero-noise mean-map spread | <= 20 MeV | 11.7 MeV | pass |
| native J/psi `width_rel_error` | <= 0.15 | 0.3849 (D3b 0.3818) | **fail** |
| native `width_rel_error_vs_identity` | <= 0.5 | 0.8169 (D3b 0.8105) | **fail** |
| Axis C native slice median | <= 12 | 31.0 (D3b 23.4) | **fail** |
| calibration: the score must move toward the measured spread | - | **pinned at 91.5 from epoch 1 to 180** | **fail** |

A log score of 91.5 implies an effective per-coordinate scale of `sqrt(0.5/91.5) = 0.074`
against the relative floor of 0.1: **the term sits on its floor and carries no gradient
toward the spread** while the model loss fell 4.97 -> 1.55.

**artifact-measured (mechanism).** Decoder core sigma (median over real kinematics) fell
0.0025 (D3b stage-1) -> 0.0010 (SR best), tail 0.00081 -> 0.00016. The kernel evaluates to
~0.008 on J/psi kinematics, so `sigma = max(learned, kernel)` leaves the **effective**
amplitude equal to the kernel and the learned path inert - which is why the physics readout
is identical to D3b to two digits. The score drove the amplitude **down**; the kernel is the
only reason the physics held. The zero-noise mean-map spread fell monotonically
17.3 -> 11.7 MeV, i.e. *below* the prior width of 14.8 MeV (over-contraction, by the mean
map). This is the Phase 0 degeneracy in the real model: a 2048-event batch out of 2.9M
J/psi events is interpolable by an 830k-parameter mean map, so `mu = x` leaves only
`log sigma` and every proper rule is minimised at `sigma -> 0`.

**conclusion.** The Phase 1 hypothesis is **refuted by its own pre-declared criteria**: a
strictly proper score of the conditional does not make the detector resolution learnable in
this model on real unpaired data. The failure is the structural degeneracy between the mean
map and the channel, not the loss or its weight, and no choice of `kappa` can fix it because
the gradient points to collapse. Two positive by-products: this is the first non-frozen arm
in eleven where the amplitude did not vanish to the floor during training, and it confirms
the frozen kernel is load-bearing.

**artifact-measured (OOD, report-only; Upsilon transfer, continuum-reweighted prior, native
multipliers).** Per-state medians minus the CMS fit 1S/2S/3S: **+122 / +103 / +111 MeV**
against D3b's +40 / +28 / +31 and A2frozen stage-3's +20.5 / +4.1 / +6.0. Inclusive mass KS
0.0752 (D3b 0.0375), W1 0.0683 GeV (D3b 0.0372), decoded 1S std 0.4329 GeV (D3b 0.2357),
inclusive mean minus data +66 MeV (D3b +29). The drift is the same state-independent mass
shift the project has been chasing, now ~3x larger. So the arm is worse than its baseline on
three of five pre-declared criteria plus the OOD readout. Artifacts:
`Run_H_SR/upsilon_transfer_continuumReweighted_best_model/`, `upsilon_peak_medians_native/`.

**mechanism (why the drift is in that direction).** With the kernel already saturating the
data width, the mean map's spread is a *residual* that the log score penalises as error, so
contracting the mean map lowers the score while the kernel holds the marginal width. The
score therefore actively prefers a narrower mean map: measured zero-noise spread 29.2 MeV
(D3b channel-off stage-1) -> 11.7 MeV here, and the OOD peak shifts grow. Identifiability and
decorrelation are the same failure. **This qualifies the Phase 0 existence proof**: the toy's
mean map was learned from scratch with no kernel, so a small map could not interpolate and
the spread was identified; under a kernel floor, even a constrained mean map may still be
pushed to contract because the residual the score penalises is the spread it wants. That must
be tested on the toy (CPU) before any further GPU arm.

**source-verified (side fix).** `scripts/loss.py` now imports `scoring_rules` from
`scripts_joint`, which was not on `sys.path` for `scripts_joint/upsilon/evaluate_z_to_x.py`;
the module adds it defensively. The first Upsilon evaluation of `Run_H_SR` failed with
`ModuleNotFoundError` until that was fixed.

**proposal (only lever left, needs authorisation).** Constrain the mean map so it cannot
carry the spread - cap its capacity or its per-coordinate residual scale - keep the score
term, and first extend the Phase 0 toy to include a kernel floor to check that the lever works
there. Re-run the same pre-declared readout only if the toy recovers the spread.

### 2026-10-04 - Session 73, the P0-P4 programme is recorded, and the control cell that was missing

**source-verified (the plan is now durable).** The experiment design approved on
2026-10-04 (phases **P0-P4**, criteria S1-S5, exit conditions) existed only in the planning
conversation. It is now written into `docs/project_tree.md`: branch **P** in the section-4
tree, a full leaf section 5 "P", rows 12-16 of the section-6 ordering table, and gate
**G-P**. Three similar numbering systems are now explicitly disambiguated in that file's
header: `P-A`/`P-B` (section 3 factorization), `P1`-`P6` (Session 59 blockers), and
**P0-P4** (the programme phases). Also recorded there: the name collision between the D3
*acceptance* leaf, `Run_H_D3zcycle` and `Run_H_D3b`.

**artifact-measured (new; the missing control cell, `outputs/cms_Joint/Run_H_A2floor/final_readout/`,
`scripts_joint/d3b_readout_probe.py`, 7 s per checkpoint on CUDA).** `Run_H_A2floor` is a
kernel floor with a learned amplitude and **no** score term - exactly `Run_H_SR`'s amplitude
configuration minus the instrument. Zero-noise mean-map spread: **29.1 -> 32.8 -> 33.0 ->
15.1 MeV** (stage-1 g20 -> stage-2 g40 -> stage-3 g140 -> last g180), `kernel_only`
38.6-45.1 MeV once the channel is on. End-of-training comparison, same locked split and
probe:

| arm | amplitude | score | zero-noise map spread [MeV] |
|---|---|---|---|
| A2frozen | frozen to kernel | no | 19.1 |
| A2floor | learned, kernel floor | no | 15.1 |
| D3b | frozen to kernel | no | 17.6 |
| SR | learned, kernel floor | yes | **11.7** |

**Consequence - Session 72 is qualified, not withdrawn.** Contraction of the mean map
happens **with or without** the score (A2floor wanders up to 33.0 MeV and ends at 15.1 with
no score at all), so the Session 72 statement that "the score actively prefers a narrower
mean map" is **downgraded from a measured mechanism to a hypothesis**: the direction is
consistent and SR ends narrowest, but the sole-cause attribution is not measured. What
survives unchanged: the score's own value is floor-pinned from epoch 1 to 180 and therefore
carries no upward gradient, and every live arm is **kernel-limited** (kernel_only 34-45 MeV
against 28.0 MeV of data). The refutation of the P1 hypothesis does not depend on the
downgraded sentence.

**The P0-P4 impact, stated as what each phase can still deliver.** This is the answer to
"P0-P4 有什么影响" in the log's own terms:

- **P0 - partially invalidated, still usable.** S1's positive half passed (log score
  1.033 +/- 0.020, 5/5 seeds) and its **negative half failed as declared** (the marginal
  objective collapsed to 0.314, 0/5 seeds below the declared 0.20). Two plan deviations are
  recorded: the shipped rule is the Gaussian **log** score, not the energy/kernel score the
  plan named (both were measured and rejected first), and the toy has **no kernel floor**,
  which is precisely the production condition. So P0's existence proof is conditional, and
  P0 must be extended with a kernel floor before it justifies any GPU arm.
- **P1 - done and refuted.** S2 pass (11.7 MeV), S3 fail x3, S4 **not measurable** until P2
  exists, S5 report-only and it fails badly (+122 MeV vs the declared +/-30). Cost 1.25x
  D3b, inside the declared <=2x budget. The executed arm also changed two coupled variables
  instead of the declared single `kappa`; the confound is covered by A2floor (above), which
  fails too.
- **P2 - untouched by the refutation and now the highest-value item.** It is the paper's
  claim 1 (a conditional-spread criterion with identity/floor references and a permutation
  control) and the only instrument that can measure S4. CPU, read-only, not started.
- **P3 - trigger not met, and the failure inverted.** Its declared trigger was a mean-map
  residual > 25 MeV; the measured residual is 11.7 MeV and the map now **over-contracts**
  (0.79x the prior std, 0.42x the CMS width). A cap on the map's spread is aimed the wrong
  way, so P3 must be redesigned (a scheduled floor on the map's spread, or a weak width
  penalty at the prior width) and only after the toy gains a kernel floor. Not authorised,
  not started.
- **P4 - unchanged, and now more blocking.** Three inconsistent J/psi targets (28.1 / ~24 /
  84 MeV) still have to be resolved before any kernel rescale, and the kernel is what sets
  the width in every live arm. Writing only, so it is the cheapest unblocking step.
  **[CORRECTED 2026-10-04: 84 MeV is not a J/psi number. The J/psi readings are 28.06
  (data std), 23.81 (std quadrature, adopted) and 30.02 (robust quadrature); 84 MeV is the
  Upsilon(1S) reference (our own fit, 84.42 +/- 3.12 MeV). See Session 74.]**

**Paper consequence.** The plan's claim-ladder item 5 ("a proper scoring rule makes the
conditional spread learnable without pairs") - the item that would have made this a PRD
paper - is **withdrawn as a claim and kept as a measured negative result**. Items 1-4 and 6
are untouched, so the paper falls back to MLST / JINST / EPJC / Comput. Softw. Big Sci
unless P2 or P4 produces a positive physics result. The productive inversion: the plan
assumed the missing object was a *loss*; the programme proved it is the **mean map** plus
the **kernel's calibration target**, with five refuted mechanisms behind that statement.

**Still stale, not touched here (proposal).** `docs/project_tree.md` sections 1 and 3 and
`paper/FRAMING.md` still carry pre-D3b framing: FRAMING's RQ1 row says "D3 (running)" for an
arm that has since been read out and refuted, and its Upsilon expectation still cites 84 MeV
where the primary sources give 96 +/- 2 MeV (all eta) and 69 +/- 2 MeV (|eta| < 1). Fixing
those is the P4 work item.

### 2026-10-04 - Session 74, P2 and P4 delivered, and the Phase 1 refutation re-attributed

**The three items the user asked for, in order: P2 (both halves), P4, and the P0 toy
extension with a kernel floor.** All three are done, and two of them changed conclusions
recorded in Sessions 71-73.

**source-verified (P2, unpaired half).** `scripts_joint/conditional_spread.py`
(`tests/test_conditional_spread.py`, 26 tests). Five legs: the exact split; the claimed
per-event scale against the data-driven requirement under **both** denominators; 68%/95%
coverage of the cycle residual with a same-n/same-D finite-draw null; a permutation control
on the **explained variance**; the identity rail and the no-information floor. Two design
corrections were forced by measurement: the coverage must be centred on **zero**, not on the
empirical median (recentring lets a biased mean map look calibrated), and the control leg
must be the explained-variance separation, not a coverage difference (coverage saturates and
has too little power when the mean-map spread is small against the channel noise).

**artifact-measured (P2 unpaired, 8 checkpoints, J/psi, 6000 events, 48 draws, 16 s,
`outputs/cms_Joint/conditional_spread/arms_2026-10-04/`).** The two deterministic stage-1
warmups (A2frozen, A2floor) claim **exactly zero** per-event spread: coverage 0.000, claim /
required 0.00, pairing sensitivity 1.16 -- rejected. Every stochastic checkpoint is
*coherent* (coverage 0.65-0.81 against a 0.66-0.68 finite-draw null; pairing sensitivity
0.22-0.79 against a 0.10 floor) but **off-band on the adopted std requirement**: A2frozen
stage-3 1.51, A2floor stage-2 1.70, A2floor last 1.47, SR 1.44, D3zcycle 1.40, D3b 1.30.
Against the **robust** denominator the same arms read 1.00-1.31. So the width failure is a
failure of the claim's **size**, and the verdict is denominator-dependent -- P4's point,
measured by an independent instrument.

**artifact-measured (P2 paired half, ppzee, all 160,000 held-out pairs, 32 encoder draws;
`scripts_joint/paired_quantile_calibration.py`, 44 tests,
`outputs/cms_Joint/ppzee/quantile_calibration/`).** 68% coverage **0.0565** [0.0554, 0.0577]
against a finite-draw null of **0.6387**; 95% coverage 0.1034 against 0.8988; pull width
**20.40** against a null of 1.0499; PIT max deviation 692 binomial sd (chi2 p = 0), strongly
U-shaped. The posterior mean is unbiased (residual rms 2.6065 vs the canonical 2.6100 GeV) --
**the encoder is centred and far too narrow** (median inferred width 0.137 GeV). The shuffled
control moves the truths 13.2 GeV rms and makes the reading *worse*, not better.

**artifact-measured (method result worth carrying).** The PIT/rank statistic is exactly
uniform under calibration for any continuous conditional, so it needs no finite-draw
correction; central-interval coverage and the pull width do. At D = 32 a perfectly calibrated
conditional reads 0.6387 at nominal 0.68 and 0.8988 at 0.95, and the pull reference is
`sqrt(1+1/D)*sqrt((D-1)/(D-3)) = 1.0499`, not 1.0. Both scripts now evaluate against a
same-n/same-D calibrated null; the unpaired one uses a t-based null (0.648 at D = 8, 0.674 at
D = 32, 0.678 at D = 512).

**source-verified + artifact-measured (P4, `docs/calibration_target_2026-10-04.md`).**
Adopted, explicitly changeable: **the per-region data-driven quadrature requirement, std
estimator -- J/psi 23.81 MeV (= sqrt(28.0605^2 - 14.8408^2), 0.01088 in log-pT), Z
2.884 GeV**, with the robust reading (30.02 MeV, the one the A0.4 R gate already uses), the
kernel's own fitting target (28.1) and the primary source (~31 MeV, arXiv:2502.14036) as
systematics. **Correction to Sessions 71-73: 84 MeV is NOT a J/psi number.** No artifact uses
it for J/psi; it is the **Upsilon(1S)** reference, and it is our own fit: 84.42 +/- 3.12 MeV
from the three-Gaussian fit to CMS Open Data record 5206
(`experiments/cms_upsilon/figures/cms_upsilon_fit_results.csv`), with 88.24 +/- 3.05 MeV for
the 8.5-11.5 GeV range -- a 4.5% analysis systematic on the reference itself. **Upsilon has no
quadrature target at all** (the region prior's std is 110.29 MeV against 84.42, so the
variance is negative); keep 84.4 primary with 96 +/- 2 / 69 +/- 2 as the acceptance systematic.
The kernel as shipped implies **33.97 MeV** at J/psi (independently 34.1 in the D3b probe,
35.4 in SR) = **1.43x the adopted target**; the rescale factor is 0.701 (E) / 0.827 (A) /
0.884 (F), a 1.26x span, so no rescale should run before the choice. D3b's 38.9 MeV native
width is 1.63x required under E, and its within/required 0.95-1.12 against F becomes
1.23-1.42 against E, i.e. **2 of its 4 stochastic checkpoints leave the [0.8, 1.25] R band:
the identification verdict is target-dependent and every identification number must be quoted
with its denominator.** Finally, the mean-map variance budget `target^2 - kernel^2` is
**negative under all three readings** (-587 / -364 / -253 MeV^2): **P3 is blocked on the
kernel rescale, not on architecture.**

**artifact-measured (the P0 extension, `scripts_joint/toy_unpaired_score.py`,
`outputs/cms_Joint/toy_unpaired_score*/`, 5 seeds, 4000 steps).** The item above was "add a
kernel floor to the toy and check that a constrained mean map plus the score lifts the
spread". It does not, and the reason is more interesting than the question:

| arm | k_hat/k_true | in band | mean-map spread ratio | scale z-dependence |
|---|---|---|---|---|
| paired target, no kernel (the Phase 0 control) | 1.101 +/- 0.044 | 5/5 | 0.94 | 0.611 (= truth) |
| **unpaired target, no floor** | 1.399 +/- 0.040 | **0/5** | **0.052** | **0.097** |
| unpaired target + the shipped relative floor | 1.341 +/- 0.055 | **0/5** | **0.011** | 0.090 |
| unpaired target + kernel floor | 1.024 +/- 0.122 | 5/5 | 0.907 | 0.614 |
| unpaired target + kernel floor + linear mean | 1.036 +/- 0.143 | 4/5 | 0.672 | 0.618 |
| **kappa = 0 control (marginal objective only)** | **1.022** | **5/5** | 1.063 | 0.614 |

1. **An unpaired target destroys the conditional structure**: the mean map is pulled onto the
   marginal mean (0.052 of its true spread) and the scale loses its z-dependence (0.097 vs
   0.611). The shipped relative floor makes it worse (0.011).
2. **The kernel floor stops the collapse but does not make the score work**: the kappa = 0
   control -- no score at all -- recovers the spread just as well (1.022 vs 1.024) on this
   planted problem, so the kernel arm's apparent success is the marginal objective's doing.
3. **The P3 lever fails on the toy too** (linear mean + score: 4/5, mean map still 0.672).
   Caveats recorded: the kappa = 0 control used the bounded mean head only, and the toy's
   marginal arm succeeds partly because that head cannot carry the whole marginal spread (an
   unbounded head reproduces the same ordering: unpaired 1.290 / mean-map 0.090, kernel arm
   1.003 / 0.844).

**artifact-measured (why the Phase 1 score sat at 91.5 -- the instrument defect, now measured
rather than hypothesised; `outputs/cms_Joint/conditional_spread/score_anatomy.json`).** The
trainer scores `x[:n]` against draws at `z[:n]` where **x and z are independently sampled
batches** (`joint_trainer.py`: `x = _sample_batch(x_train)`, `z = _sample_batch(z_train)`), so
the term is the *marginal* score, not the paired conditional score the toy validated. And the
per-coordinate anatomy at D3b/SR: the draws' spread is **0.0144-0.0306 of the reference std in
all eight coordinates** against a 0.1 relative floor -- the floor is 3-7x larger than the claim
everywhere, so `max(claimed, 0.1*ref)` clamped the term in every coordinate. The measured
value (99-101 on real kinematics; 91.5 during training) is entirely the clamped term, and its
gradient with respect to the amplitude was **exactly zero**. Without the floor the same term
reads 3,264 (D3b) / 3,464 (SR). Cause: the physics kernel is calibrated in log-pT (~0.8% of
pT), so its absolute smearing is 1.5-3% of the cartesian reference stds.

**conclusion, and a correction to Session 72.** The P1 hypothesis is **refuted as
implemented**: an unpaired target plus a floor that clamped every coordinate. Session 72's
statement that "the score drove the amplitude down" is **withdrawn** -- the score had no
amplitude gradient at all, and the pre-existing `x_reco` term (F1) drove the amplitude down.
What survives with a mechanism: the term degenerated into a pure "pull the mean map onto an
unpaired target" loss whose optimum is the marginal mean, which is the measured contraction
29.2 -> 11.7 MeV. The toy's clean statement, which is the paper's claim: **a strictly proper
score of a conditional identifies the spread only against a paired target; with an unpaired
target it is actively harmful, collapsing the mean map and removing the scale's dependence on
the condition.** The paired bench (ppzee) is the only place in this project where that target
exists, and there the posterior is far too narrow (coverage 0.0565 vs 0.6387).

**proposal (next, needs authorisation).** The live blocker is the **kernel amplitude**, not
the mean map: adopt P4's target, rescale the kernel by 0.701 (one controlled change, one
config, the physics kernel's `scale`), and re-run the same pre-declared readout. Expected if
the analysis is right: decoded J/psi native width 38.9 -> ~27 MeV and the four D3b stochastic
checkpoints back inside the R band. Do not spend a GPU arm on P3 until that is measured.
Cheap parallel items: (a) a fitted J/psi peak sigma on the locked split (Crystal Ball +
exponential, as CMS does) to decide A vs E empirically -- read-only; (b) an Upsilon region
prior restricted to its signal component so an Upsilon quadrature target becomes definable.

### 2026-10-04 - Session 75, decision documents consolidated to the latest state (read-only)

**source-verified (documents).** At the user's request the decision-document set
was rewritten to reflect the post-Session-74 state, with no training, no
checkpoint changes and no `outputs/` modification:

- `docs/project_tree.md` **v2** — all inline updates/corrections folded into
  final statements; branch P closed (P0 done + extended, P1 refuted with the two
  measured instrument defects, P2 done = claim 1, P3 blocked on P4, P4 done);
  new section 6.0 "benchmark board" (identity/floor references, Axis R band
  0.8-1.25, calibration targets, Axis C gates 3/10, ppzee oracle 0.8099,
  coverage/pull nulls, G-D criteria, PRD bar); new branch **I** with six
  innovation candidates (I1-I6, all proposal-labelled, none authorised); the
  retired/parked table updated.
- `paper/FRAMING.md` **v2** — final RQ answers (RQ1 negative with workaround,
  RQ2 split, RQ3 negative), claim ladder item 5 withdrawn, PRD assessment
  unchanged in its verdict (not PRD; needs M2 + M3), I3 added as the theory
  spine.
- `docs/calibration_target_2026-10-04.md` — re-laid out, **no number changed**;
  S2c cross-linked as the empirical settler of E vs A.
- `docs/literature_2026-10-04.md` — section 4.1 addendum records that
  recommendation 1 (proper scoring rule) was implemented and refuted the same
  day; the recommendation itself is marked closed.
- This file: section 1.7 added; history untouched (append-only).

**proposal (unchanged by the rewrite, needs authorisation).** S2 kernel rescale
(0.701, D3b config, pre-declared readout) remains the live next run; S2c and I2
are the cheap read-only parallel items. The P-branch working-tree changes are
still uncommitted; per section 1.1 they should be committed before any new run.

### 2026-10-04 - Session 76, read-only items executed and the knee-kernel run launched

**What the user asked for, and what was delivered.** Execute the four read-only
items (S2c, I2, I6, Upsilon target), the I3 write-up, then start the training
run. All done; the run is in flight.

**artifact-measured (I2).** See section 1.8. The decisive number is the
two-parameter solve on J/psi + Z predicting Upsilon(1S) at 0.367x, which refutes
S2 as written and produced the knee law instead.

**source-verified + artifact-measured (the knee kernel).** `knee_pt_gev` added
to the linear kernel form with 5 focused tests; the fit script sweeps the knee
grid and solves the two remaining parameters exactly per knee. Adopted knee
20 GeV by physical argument (plateau middle), not by the score argmax — the score
keeps improving to 24 only because it is chasing the Upsilon reference, which
would be calibration on the held-out region. The config header records the whole
scan and the disclosure.

**artifact-measured (I6).** The Upsilon zero-noise width is pT-stratified:
27.5 MeV locally at low pair-pT against 372.7 MeV in the top sextile, bias
flipping sign near 4 GeV. This is the actionable form of the old "the map carries
214 MeV" statement and it is what P3's redesign should target.

**artifact-measured (S2c, inconclusive and recorded as such).** The Crystal Ball
fit fails to describe the data (chi2/dof 1233, tails railed); no usable number.
E stays adopted by decision. A proper fit (wider window, psi(2S), FSR tail,
bin-width scan) remains open.

**source-verified (I3).** The re-mixing proposition is written to
`docs/non_identifiability_2026-10-04.md` with a proof sketch, the measured
instances (shuffled decode bit-for-bit invariance; the toy's 0.052 collapse), and
the boundary table of what restores identification.

**RUNNING (needs monitoring, no further authorisation assumed).**
`Run_H_kneeKernel`: `python scripts_joint/run_joint.py --run H_kneeKernel
--device cuda`, background job id `pwsh-408`, log `logs/Run_H_kneeKernel.log`,
outputs `outputs/cms_Joint/Run_H_kneeKernel/`. Launched 20:04, ~100 s/epoch,
ETA ~5 h. Stage-1 train loss 4.41 -> 3.40 over the first four logged epochs.
**Next session: read `history.json` and run the pre-declared readout; do not
start a second GPU job while it runs.**

