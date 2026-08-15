# sm_onia UFO model — rebuild reference (from Virat's recipe, assessed 2026-08-14)

The `sm_onia` UFO model used to generate the J/psi prior
(data/cms_jpsi_mumu_mg5_8tev_1M.hdf5) lives on the Linux machine
(/home/ziqinl12/Documents/MadGraph5_onia/models/sm_onia, patched as
sm_onia-c_mass; model_patch_checksum 433f9f73... recorded in the HDF5 attrs).
It is NOT in this repo and NOT on this Mac. The recipe below (from Virat's
email) reconstructs it on top of the stock `sm` model. Verified against the
prior's HDF5 attrs and against MG5 3.7.0's stock sm (extracted at
~/MG5_aMC_v3_7_0 on this Mac): all edits match the file layout, and the UFO
objects used (FFV1 lorentz, Identity(1,2) color) exist in stock sm.

## 0. The recipe (verbatim, lightly annotated)

Precondition: a copy of the stock model first:

    cp -r ~/MG5_aMC_v3_7_0/models/sm ~/MG5_aMC_v3_7_0/models/sm_onia

Then append to the model files:

    cd models/sm_onia/
    cat >> parameters.py << 'EOF'
    MJPSI = Parameter(name = 'MJPSI', nature = 'external', type = 'real', value = 3.0969, texname = '\text{MJPSI}', lhablock = 'MASS', lhacode = [ 443 ])
    WJPSI = Parameter(name = 'WJPSI', nature = 'external', type = 'real', value = 0.0000929, texname = '\text{WJPSI}', lhablock = 'DECAY', lhacode = [ 443 ])
    gJpsi = Parameter(name = 'gJpsi', nature = 'internal', type = 'real', value = '0.01', texname = 'g_{J/\psi}')
    EOF
    cat >> particles.py << 'EOF'
    jpsiv = Particle(pdg_code = 443, name = 'jpsiv', antiname = 'jpsiv', spin = 3, color = 1, mass = Param.MJPSI, width = Param.WJPSI, texname = 'jpsiv', antitexname = 'jpsiv', charge = 0, GhostNumber = 0, LeptonNumber = 0, Y = 0)
    EOF
    cat >> couplings.py << 'EOF'
    GC_JPSI = Coupling(name = 'GC_JPSI', value = 'gJpsi', order = {'QED':1})
    EOF
    cat >> vertices.py << 'EOF'
    V_jpsi_mu = Vertex(name = 'V_jpsi_mu', particles = [ P.mu__plus__, P.mu__minus__, P.jpsiv ], color = [ '1' ], lorentz = [ L.FFV1 ], couplings = {(0,0):C.GC_JPSI})
    V_jpsi_u = Vertex(name = 'V_jpsi_u', particles = [ P.u__tilde__, P.u, P.jpsiv ], color = [ 'Identity(1,2)' ], lorentz = [ L.FFV1 ], couplings = {(0,0):C.GC_JPSI})
    V_jpsi_d = Vertex(name = 'V_jpsi_d', particles = [ P.d__tilde__, P.d, P.jpsiv ], color = [ 'Identity(1,2)' ], lorentz = [ L.FFV1 ], couplings = {(0,0):C.GC_JPSI})
    V_jpsi_s = Vertex(name = 'V_jpsi_s', particles = [ P.s__tilde__, P.s, P.jpsiv ], color = [ 'Identity(1,2)' ], lorentz = [ L.FFV1 ], couplings = {(0,0):C.GC_JPSI})
    V_jpsi_c = Vertex(name = 'V_jpsi_c', particles = [ P.c__tilde__, P.c, P.jpsiv ], color = [ 'Identity(1,2)' ], lorentz = [ L.FFV1 ], couplings = {(0,0):C.GC_JPSI})
    EOF
    rm -rf __pycache__ && rm -f py3_model.pkl

Edit restrict_c_mass.dat: add `443 3.096916e+00 # MJpsi` to Block MASS
(between 25 and the end of the block) and a DECAY line
`DECAY 443 9.340000e-05`; then re-clear __pycache__/py3_model.pkl and
`import sm_onia-c_mass`.

## 1. Verified consistency with the existing prior (HDF5 attrs)

| attrs in cms_jpsi_mumu_mg5_8tev_1M.hdf5 | recipe |
|---|---|
| effective_coupling_gJpsi = 0.01 | gJpsi = 0.01 ✓ |
| jpsi_mass_GeV = 3.0969 | MJPSI 3.0969 ✓ (restrict line has 3.096916, see §2) |
| jpsi_width_GeV = 9.29e-05 (92.9 keV) | WJPSI default 92.9 keV ✓ (restrict line 93.4 keV, see §2) |
| process p p > jpsiv j, jpsiv > mu+ mu- | quark vertices u/d/s/c enable q qbar / q g production ✓ |
| particle_order mu- first | decay vertex V_jpsi_mu + post-processing ✓ |

## 2. Corrections / caveats

1. **Missing step 0** — the recipe assumes models/sm_onia already exists;
   copy stock sm first (see §0). The stock-sm layout in MG5 3.7.0 was verified
   on this Mac (parameters/particles/couplings/vertices/restrict_c_mass.dat).
2. **Width mismatch inside the recipe**: WJPSI default = 92.9 keV but the
   restrict_c_mass.dat DECAY line = 93.4 keV. The restrict value wins on
   import. The ORIGINAL prior used 92.9 keV -> if reproducing the original,
   write `DECAY 443 9.290000e-05`.
3. **Mass mismatch**: restrict line 3.096916 (PDG-2024 value) vs the original
   prior's 3.0969. 16 keV difference; irrelevant for training, matters only
   for bit-level reproducibility.
4. **No gluon-fusion vertex**: the recipe defines only q qbar - jpsiv
   couplings (mirroring the Z-quark vertex, color Identity(1,2)). There is NO
   g g jpsiv vertex. This matches the original process card's jet definition
   (j = g u c d s ...) — the original prior was generated WITHOUT gluon
   fusion. Reproduce as-is; do not "improve" silently (one-factor discipline).
5. **The final `generate p p > mu+ mu-` is only a model smoke test.** The
   production process was `p p > jpsiv j, jpsiv > mu+ mu-` (2->2 with a jet;
   the 2->1 pure s-channel variant is the unusable inclusive file, memory §4.5).
6. **model_patch_checksum will change** after rebuild; record the new checksum
   in the new HDF5 attrs.

## 3. Still missing for the prior rebuild (not covered by the recipe)

- production run_card (8 TeV, mmll [2.8,3.4], ptl 0, ptj 1, drll 0, etal -1,
  PDF choice, 1M events, 10 batches, seeds 240001-240010) — on the Linux box
- the LHE -> HDF5 post-processing script (FDL/zData writer, exactly-one-mu
  pair, finite/positive checks, float32) — on the Linux box
- for the NEW prior: resolution smearing (~28 MeV), harder cuts (ptl_min 2.5,
  |etal|max 2.5, higher ptj_min), continuum component (generate with stock sm:
  p p > mu+mu- j — do NOT use sm_onia for the continuum, or the J/psi
  s-channel pollutes it), post-filter mixing ~85/15, massive-daughters E
  convention. See the prior-build plan (session 6, memory §4.7 follow-up).

## 4. Verification commands after rebuild (on this Mac)

    cd ~/MG5_aMC_v3_7_0
    ./bin/mg5_aMC   # PATH must include the conda python (has six); gfortran is at /opt/homebrew/bin/gfortran
    import model sm_onia-c_mass --modelname
    generate p p > jpsiv j, jpsiv > mu+ mu-
    output jpsi_priortest
    # then launch with the production run_card
