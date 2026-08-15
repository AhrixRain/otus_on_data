# J/psi prior rebuild — production runbook (MG5 on macOS arm64)

Last updated: 2026-08-14 (session 6/7, goal round 1). Documents the complete
chain that regenerates a CMS-matched J/psi prior on this Mac and writes the
mixed HDF5 into data/.

## 0. Why this prior

memory.md §4.7 quantified the CMS-data/MG5-prior mismatch that broke J/psi
training: delta-function mass prior (0.2 MeV vs data 27.9 MeV), 0% continuum
(vs ~15% inside the signal region), 2.6x too-soft pT spectrum, and a 0.60%
pass rate of the trigger-matched filter. The new prior fixes all four:

1. Signal p p > jpsiv j (sm_onia, effective vector J/psi) with generation
   cuts matched to the fiducial region (ptl 3.0, etal 2.5, mmll [2.9,3.3],
   ptj 10, cut_decays=True) -> ~92% post-filter efficiency.
2. Continuum p p > mu+mu- j (stock sm with massive muons) mixed at
   ~15% of post-filter counts (signal-region scope).
3. Resolution smearing pT' = pT*(1+N(0,a)), a=0.0127 calibrated to the
   data's 27.9 MeV mass std (scripts/smear_prior.py).
4. Massive daughters everywhere (MM = 0.105658 in both restrict cards,
   E = sqrt(p^2 + m_mu^2) in the postprocessor) so z-space E-based and
   x-space p-based mass conventions agree.

## 1. Environment (verified on this Mac)

- MG5_aMC 3.7.0 extracted at ~/MG5_aMC_v3_7_0 (same version that made the
  original prior; tarball /tmp/MG5_aMC_v3.7.0.tar.gz).
- Python: use the conda cms env (has six):
  /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/python.
  System /usr/bin/python3 (3.9.6) lacks six.
- gfortran: brew gcc 16.1.0 -> /opt/homebrew/bin/gfortran (auto-detected).
- LHAPDF 6.5.6: conda-forge package in the cms env; PDF grid
  NNPDF31_lo_as_0130 (lhaid 315200) downloaded from lhapdfsets.web.cern.ch
  into /opt/homebrew/Caskroom/miniforge/base/envs/cms/share/LHAPDF/.
- mg5_configuration.txt addition:
  lhapdf_py3 = /opt/homebrew/Caskroom/miniforge/base/envs/cms/bin/lhapdf-config

### macOS-specific fixes (both REQUIRED)

1. rpath for libLHAPDF.dylib. The compiled madevent/gensym binaries link
   @rpath/libLHAPDF.dylib and crash at runtime with the dyld 'Reason:' error
   (MG5 parses the survey stdout and dies with
   ValueError: could not convert string to float: 'Reason:').
   Fixed by adding -Wl,-rpath,/opt/homebrew/Caskroom/miniforge/base/envs/cms/lib
   to the llhapdf line of Source/make_opts in each process dir AND in
   Template/LO/Source/make_opts + .make_opts (survives re-output). Belt and
   braces: export DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/cms/lib
   when launching.
2. systematics off (run_card: False = use_syst): the default LO
   systematics computation requests the NNPDF23_lo_as_0130_qed errorset,
   which is not installed.

Every launch also needs:
    export LHAPATH=/opt/homebrew/Caskroom/miniforge/base/envs/cms/share/LHAPDF
    export DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/cms/lib
    export PATH=/opt/homebrew/Caskroom/miniforge/base/envs/cms/bin:$PATH

## 2. Models

- sm_onia(-c_mass): stock sm + jpsiv particle/vertices (recipe in
  docs/sm_onia_rebuild.md) + massive muons (MM = 0.105658 in
  restrict_c_mass.dat MASS block; MASS 443 = 3.0969, DECAY 443 = 9.29e-05).
- sm_mumass(-c_mass): stock sm + massive muons only (continuum).

## 3. Cards (committed in scripts/mg5_cards/)

- signal_proc_card.dat / signal_run_card.dat: nevents 200000, ebeam 4000,
  lhapdf 315200, ptj 10, ptl 3.0, etal 2.5, drll 0.0, mmll [2.9,3.3],
  cut_decays True, use_syst False.
- continuum_proc_card.dat / continuum_run_card.dat: same, nevents 50000.

## 4. Production (running 2026-08-14)

    cd ~/MG5_aMC_v3_7_0/<proc_dir>
    printf 'launch ...' | ./bin/madevent    # or via mg5_aMC 'launch <dir>'

Outputs: <proc_dir>/Events/run_NN/unweighted_events.lhe.gz

## 5. Post-processing (scripts/)

    python scripts/lhe_to_prior_hdf5.py --lhe <signal>.lhe --out sig.hdf5 --label signal
    python scripts/lhe_to_prior_hdf5.py --lhe <cont>.lhe --out con.hdf5 --label continuum
    python scripts/smear_prior.py --in sig.hdf5 --out sig_smear.hdf5 --a 0.0127 --seed 0
    python scripts/smear_prior.py --in con.hdf5 --out con_smear.hdf5 --a 0.0127 --seed 0
    python scripts/mix_prior.py --signal sig_smear.hdf5 --continuum con_smear.hdf5 \
        --out data/cms_jpsi_mumu_mg5_8tev_mixed.hdf5 --frac-signal 0.85 --seed 0

pT hardening (DONE 2026-08-14, closes the last kinematic gap):

    REF=outputs/cms_JpsiDoubleMuons/archive/.plot_cache/selected_split_0ed04817d72bb34f84ed.npz
    python scripts/reweight_prior.py --in sig_smear.hdf5 --ref-cache $REF \
        --out sig_rw.hdf5 --n 80648 --max-weight 20 --seed 0
    python scripts/reweight_prior.py --in con_smear.hdf5 --ref-cache $REF \
        --out con_rw.hdf5 --n 14232 --max-weight 20 --seed 0
    python scripts/mix_prior.py --signal sig_rw.hdf5 --continuum con_rw.hdf5 \
        --out data/cms_jpsi_mumu_mg5_8tev_mixed.hdf5 --frac-signal 0.85 --seed 0

## 6. Acceptance targets (data numbers from memory.md §4.7 / §4.5)

    trigger-matched pass rate        >= 80%     (was 0.60%)
    mass mean / std (massive conv.)  3.0978 / 25-30 MeV  (was 3.0969 / 0.2 MeV)
    muon pT median                  ~12.6 GeV  (was 4.8)
    pair pT median                  ~26 GeV    (was 10.1)
    continuum fraction in window    ~15%       (was 0%)
    support coverage (signal-window data)  >= 90%  (was 69.8%)

Validate with:
    python scripts/prior_kinematics.py --file data/cms_jpsi_mumu_mg5_8tev_mixed.hdf5
    python scripts/jpsi_mismatch_check.py   (edit PRIOR path) or
    a direct data-vs-new-prior comparison using the version-3 split caches.

Known remaining limitations: (a) non-prompt B-decay component is NOT
included (future work; the reweighting compensates statistically but does not
model the physics); (b) the reweighting up-weights the pT tail, so the
high-pT statistics are effectively thinner than the raw counts suggest.
