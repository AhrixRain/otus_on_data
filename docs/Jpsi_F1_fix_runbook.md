# J/psi F1 fix — runbook (implementation complete, runs deferred to the user)

Last updated: 2026-08-15 (session 5). Everything below is implemented, unit-tested,
dry-run-validated, and committed. No training run is in progress.

## 0. What the fix is

E1 measurements (memory.md §4.5) established that the J/psi training failure has two
drivers: (a) the vanilla paper objective's degenerate-prior/composition mismatch and
(b) the prior/data kinematic support mismatch (trigger-matched prior is 2.5x softer
than the parked-B data, with a junk tail in the skim). The implemented fix:

- **Data scope:** signal region [3.0369, 3.1569] GeV (paper restricted-decoder scope);
  muon_pt_max: 100 removes the skim's misreconstructed multi-TeV junk
  (scripts/cms_data.py, optional key; E1: hard 3 GeV trigger floor confirmed).
- **Prior:** theory_prior_selection filters the MG5 signal file through the
  trigger-equivalent selection (pT>3, |eta|<2.4, mass window) — 5,995 of 1M events
  survive (scripts/cms_data.filter_theory_prior; cache-keyed; unit-tested).
- **Two ready configs:**
  - configs/archive/cms_JpsiDoubleMuons_v3.10_staged_restricted.yaml — **recommended fix**:
    v3.5's staged physics loss (the only objective that ever produced a good
    generator, because its x_sim terms constrain D(z~prior) directly) + the new
    scope + a re-weighted checkpoint-selection score so the cycle is no longer
    masked (memory.md §6.3).
  - configs/archive/cms_JpsiDoubleMuons_v3.9_F1_restricted.yaml — the single-factor
    diagnostic: vanilla two-term objective + the new scope (extends the v3.8
    paper config verbatim). Run this only if you want the controlled comparison.

## 1. Environment

On this machine use the conda env (NOT .venv, which does not exist here):

    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python

Python 3.10.20, torch 2.12.1 (MPS available), numpy 2.2.5, h5py, uproot, awkward.
Verify:

    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"

## 2. Pre-flight (optional but recommended, ~2 min)

    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python scripts/preflight.py         --config configs/archive/cms_JpsiDoubleMuons_v3.10_staged_restricted.yaml --device mps

Unit tests (10 v3.9 data-path tests + full suite, all green as of 2026-08-15):

    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python -m unittest discover -s tests

## 3. Run the fix (v3.10)

    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python scripts/train.py         --config configs/archive/cms_JpsiDoubleMuons_v3.10_staged_restricted.yaml --device mps

- ~200 epochs (50 anchor warmup / 50 joint / 100 decoder-response), ~605 steps/epoch
  at batch 4796 (auto-capped to the 4,796-event filtered prior), ~0.25 s/step on MPS:
  **expect ~12-15 h** including per-epoch validation.
- Run dir: outputs/cms_JpsiDoubleMuons/archive/Jpsi_v3.10_staged_restricted (train.py refuses
  to overwrite an existing dir — pick a new --run-name for reruns).
- Monitor: tail -f <run_dir>/train_log.csv (rows every 10 epochs) or status.json.
  Watch the stage-3 phase: the cycle term is now weighted in checkpoint selection.
- Expected-good signs: train loss falls through the stages; eval selection score
  tracks; z->x generator quality (offline) should approach v3.5's level.

If you also want the vanilla diagnostic (v3.9, ~13-15 h):

    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python scripts/train.py         --config configs/archive/cms_JpsiDoubleMuons_v3.9_F1_restricted.yaml         --run-name Jpsi_v3.9_F1_restricted_ptmax100_run2 --device mps

(NOTE: Jpsi_v3.9_F1_restricted_ptmax100 already exists with ~20 epochs of a run the
user terminated — use the --run-name above, or delete that directory first if you
prefer the original name. Nothing is deleted by these scripts automatically.)

## 4. Evaluate after training

    # Full three-path evaluation (simulation / reconstruction / unfolding)
    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python scripts/eval.py         --config <the same config used for training>         --checkpoint outputs/cms_JpsiDoubleMuons/archive/<run_dir>/best_model.pt         --device mps

    # Also evaluate the final checkpoint (checkpoint selection can be noisy with
    # the 600-event z_val; memory.md §4.6)
    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python scripts/eval.py         --config <config> --checkpoint outputs/cms_JpsiDoubleMuons/archive/<run_dir>/checkpoint_final.pt         --device mps --output-dir outputs/cms_JpsiDoubleMuons/archive/<run_dir>/eval_final

    # Verdict table (W1/KS + shape stats)
    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python scripts/eval_verdict.py         --eval-dir outputs/cms_JpsiDoubleMuons/archive/<run_dir>/eval

    # Generator-quality check with well-populated sampling (20 draws x full prior)
    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python scripts/sample_generator.py         --config <config> --checkpoint outputs/cms_JpsiDoubleMuons/archive/<run_dir>/best_model.pt         --device mps --draws 20

    # Out-of-scope diagnostic (full-window data fed to the signal-region model)
    /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python scripts/eval.py         --config configs/archive/cms_JpsiDoubleMuons_v3.9_out_of_scope_eval.yaml         --checkpoint outputs/cms_JpsiDoubleMuons/archive/<run_dir>/best_model.pt         --device mps --output-dir outputs/cms_JpsiDoubleMuons/archive/<run_dir>/eval_out_of_scope

## 5. Baselines for comparison (memory.md §4.3b, same harness)

| checkpoint | sim W1 [GeV] | sim KS | reco W1 | reco KS | unfold W1 | unfold KS |
|---|---|---|---|---|---|---|
| v3.5 best (ep160) | 0.0086 | 0.045 | 0.1244 | 0.520 | 0.0103 | 0.440 |
| v3.8 best (ep280) | 20.90 | 0.896 | 1.594 | 0.594 | 5.086 | 0.618 |

Scope caveat: v3.5/v3.8 were evaluated on the full [2.6,3.5] window; the v3.10/v3.9
models are signal-region models, so their W1/KS are structurally smaller — compare
shape stats (peak mean/std vs signal-region data: mean 3.086, std 28 MeV p-based)
and use the out-of-scope diagnostic to quantify continuum handling.

## 6. Guardrails

- Do not delete data/, outputs/, or the partial run dirs without deciding to.
- The MG5 priors are frozen inputs; the inclusive file is unusable for this fix
  (pure 2->1, see memory.md §4.5).
- One-factor discipline: v3.10 already bundles scope + selection-score re-weight
  (justified in its header); any further change should be its own config.
