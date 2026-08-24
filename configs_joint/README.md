# cms_Joint run configurations

`cms_Joint_runA.yaml` is the locked first baseline in the joint-resonance
series. Its scientific question is narrow: can one shared response model,
without an explicit parent-mass feature or resonance label, close both the
J/psi and Z dimuon regions before seeing Upsilon?

Run A uses the available J/psi dimuon MG5 prior and the available massless
Drell-Yan lepton prior for the Z region. The latter has a historical `dyee`
filename but stores only truth four-vectors; a dedicated DY-to-dimuon prior is
the controlled replacement made in Run B.

`cms_Joint_runB.yaml` inherits the complete Run A contract and changes only
the run identity and the Z theory prior to
`cms_dymumu_mg5_8tev_dy1j_ptj5_fiducial_70_110_1M.hdf5`. This makes Run B a
direct prior-species comparison rather than a retune. The inherited stage
names intentionally remain unchanged so the optimization schedule is visibly
identical in both runs.

`cms_Joint_runC.yaml` keeps Run B's dedicated dimuon prior and introduces the
next controlled training profile: RMS-balanced checkpoint selection, a
six-layer 512-unit MLP, 20% more epochs in every stage, and a 24,576-event
training batch with four steps per epoch. Evaluation uses batches of 4,096.
Run C remains mass-blind and keeps Upsilon locked.

`cms_Joint_runC_fullScale.yaml` removes every split cap while preserving the
80/10/10 holdout contract. Its deterministic cycling sampler visits every row
of each training pool before starting that pool's next shuffled cycle. This
uses all selected training-partition events without contaminating validation
or test data.

`cms_Joint_runD.yaml` inherits the full-scale Run C contract and replaces only
the Z truth prior with
`cms_dymumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_70_110_1M.hdf5`.
The new prior contains post-shower stable muons from merged 0/1-parton
MG5+Pythia8 CKKW-L production, includes lepton QED FSR, and removes the old
hard pair-recoil floor. Upsilon remains excluded from training and selection;
because its original first look has already occurred, any later Run D Upsilon
evaluation is post-unblinding held-out transfer rather than a new zero-shot
claim.

`cms_Joint_runE.yaml` inherits Run D and changes only the epoch definition.
One Run E epoch deterministically visits every selected J/psi-data,
J/psi-prior, Z-data, and Z-prior training row exactly once. The largest
partition determines the number of optimizer updates, and shorter partitions
are divided into the same number of balanced batches without oversampling.

Do not modify a completed run in place after producing a full checkpoint. Copy the
configuration to a new run name for broad-mass support, alternative priors,
region-weight changes, or architecture changes. This preserves each run's
identity and the meaning of the original locked Upsilon test.
