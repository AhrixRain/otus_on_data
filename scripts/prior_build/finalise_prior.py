#!/usr/bin/env python3
"""Assemble a region prior from converted per-component HDF5s, with full
method book section 10 provenance, and write its JSON manifest.

Why this is a separate step from the converter.  A region prior is several
labelled components generated in separate MG5 runs (J/psi: signal + continuum;
Upsilon: 1S + 2S + 3S + continuum), and the HepMC of each run is deleted as soon
as that run's HDF5 is written and checksummed -- disk discipline, since the
Upsilon and J/psi HepMC alone run to tens of GB.  So the region file is built
from the per-component HDF5s, not by re-reading HepMC.

**No selection is applied here, deliberately.**  The generation cuts were applied
by MG5 at matrix-element level and are recorded in the attributes; the parton
shower then moves events across every one of those edges, and those migrated
events are the whole point of generating loose (method book section 1, reason 2:
"a sample generated exactly at the analysis window is missing the events that
smear into it").  Re-applying the generation cuts offline would throw the
migration away.  The analysis selection is the consumer's choice and is never
baked in.

Usage:
    python scripts/prior_build/finalise_prior.py --region z \\
        --component continuum:3:~/mg5work/converted/prod_z_continuum.hdf5 \\
        --out data/priors/z_unified_bare_tms22p8.hdf5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

MUON_MASS = 0.105658
REPO = Path(__file__).resolve().parents[2]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats_block(z, nominal=None):
    z = np.asarray(z, dtype=np.float64)
    e = (np.sqrt((z[:, 0:3] ** 2).sum(1) + MUON_MASS ** 2)
         + np.sqrt((z[:, 4:7] ** 2).sum(1) + MUON_MASS ** 2))
    p = z[:, 0:3] + z[:, 4:7]
    mass = np.sqrt(np.clip(e ** 2 - (p ** 2).sum(1), 0.0, None))
    mass_stored = np.sqrt(np.clip((z[:, 3] + z[:, 7]) ** 2 - (p ** 2).sum(1), 0.0, None))
    muon_pt = np.concatenate([np.hypot(z[:, 0], z[:, 1]), np.hypot(z[:, 4], z[:, 5])])
    pair_pt = np.hypot(z[:, 0] + z[:, 4], z[:, 1] + z[:, 5])
    q75, q25 = np.percentile(mass, [75, 25])
    out = dict(
        events=int(len(z)),
        mass_mean_gev=float(mass.mean()), mass_std_mev=float(mass.std() * 1000),
        mass_median_gev=float(np.median(mass)),
        mass_iqr_width_mev=float((q75 - q25) / 1.349 * 1000),
        mass_std_stored_energy_mev=float(mass_stored.std() * 1000),
        muon_pt_median_gev=float(np.median(muon_pt)),
        muon_pt_p99_gev=float(np.percentile(muon_pt, 99)),
        pair_pt_median_gev=float(np.median(pair_pt)),
        pair_pt_p99_gev=float(np.percentile(pair_pt, 99)),
        pair_pt_min_gev=float(pair_pt.min()),
        mass_min_gev=float(mass.min()), mass_max_gev=float(mass.max()),
        muon_pt_min_gev=float(muon_pt.min()))
    if nominal:
        for w in (5.0, 10.0):
            out[f"frac_within_{int(w)}mev"] = float(np.mean(np.abs(mass - nominal) < w / 1000))
    return out


def parse_component(spec):
    name, cid, path = spec.split(":", 2)
    return dict(name=name, component_id=int(cid), path=str(Path(path).expanduser()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True)
    ap.add_argument("--component", action="append", required=True, type=parse_component,
                    metavar="NAME:ID:PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--truth-variant", default="bare")
    ap.add_argument("--truth-variant-verified", default="true")
    ap.add_argument("--tms", type=float, required=True)
    ap.add_argument("--ptj", type=float, default=2.5)
    ap.add_argument("--nominal-mass", type=float, default=None)
    ap.add_argument("--scan-report", default=None)
    ap.add_argument("--status", default="ACCEPTED")
    ap.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    idx_path = REPO / "scripts" / "mg5_cards" / "unified" / "index_production.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}

    zs, cids, ws, per_component = [], [], [], {}
    for comp in args.component:
        with h5py.File(comp["path"], "r") as fh:
            z = fh["FDL/zData"][:]
            w = fh["FDL/weight"][:] if "weight" in fh["FDL"] else np.ones(len(z))
            src_attrs = {k: fh.attrs[k] for k in fh.attrs}
        zs.append(z.astype(np.float32))
        ws.append(w)
        cids.append(np.full(len(z), comp["component_id"], dtype=np.int8))
        meta = index.get(f"{args.region}/{comp['name']}/production", {})
        side = Path(comp["path"]).with_suffix(".json")
        per_component[comp["name"]] = dict(
            component_id=comp["component_id"],
            source_hdf5=comp["path"], source_sha256=sha256_of(comp["path"]),
            source_manifest=json.loads(side.read_text()) if side.exists() else None,
            card_metadata=meta,
            prior_stats=stats_block(z, meta.get("nominal_mass") or args.nominal_mass))
        print(f"  {comp['name']:>12} (id {comp['component_id']}): {len(z)} events")

    z = np.concatenate(zs)
    cid = np.concatenate(cids)
    weight = np.concatenate(ws)

    any_meta = next((v["card_metadata"] for v in per_component.values()
                     if v["card_metadata"]), {})
    attrs = {
        "schema_version": "unified-prior-1",
        "region": args.region,
        "status": args.status,
        "truth_variant": args.truth_variant,
        "truth_variant_verified": args.truth_variant_verified,
        "truth_variant_definition": "status-1 muon after final-state QED radiation",
        "merging_scale_TMS_GeV": args.tms,
        "matrix_element_ptj_GeV": args.ptj,
        "merging_settings": json.dumps({
            "scheme": "CKKW-L", "Merging:doKTMerging": "on",
            "Merging:Dparameter": 0.4, "Merging:nJetMax": 1,
            "Merging:TMS_GeV": args.tms,
            "note": "TMS is set by the run card's ktdurham; the Pythia input card "
                    "carries Merging:TMS = -1.0, so the merging scale and the "
                    "matrix-element cut cannot disagree."}),
        "generation_cuts": json.dumps({
            "lepton_pt_min_GeV": any_meta.get("ptl"),
            "lepton_abs_eta_max": any_meta.get("etal"),
            "mass_window_GeV": any_meta.get("generation_window"),
            "parton_pt_min_GeV": args.ptj}),
        "analysis_window_new_GeV": json.dumps(any_meta.get("analysis_window_new")),
        "analysis_window_legacy_GeV": json.dumps(any_meta.get("analysis_window_legacy")),
        "analysis_muon_pt_GeV": any_meta.get("analysis_pt"),
        "selection": "NONE. The analysis selection is an offline choice "
                     "(method book section 1). Generation cuts were applied by MG5 "
                     "at matrix-element level and are recorded above; the shower "
                     "migrates events across them and those events are KEPT.",
        "columns": "mu_minus_px,mu_minus_py,mu_minus_pz,mu_minus_E,"
                   "mu_plus_px,mu_plus_py,mu_plus_pz,mu_plus_E",
        "particle_order": "mu- (PDG 13) first; mu+ (PDG -13) second",
        "four_vector_convention": "(px, py, pz, E)",
        "units": "GeV",
        "daughter_muon_mass_GeV": MUON_MASS,
        "component_names": json.dumps([c["name"] for c in args.component]),
        "component_id_mapping": json.dumps({c["name"]: c["component_id"]
                                            for c in args.component}),
        "component_counts": json.dumps({n: v["prior_stats"]["events"]
                                        for n, v in per_component.items()}),
        "components_are_mixed": "false",
        "mixture_fraction_note":
            "Components are shipped LABELLED and UNMIXED. The mixture fraction is a "
            "declared analysis parameter to be fitted from a mass sideband by the "
            "consumer and varied as a systematic; it is deliberately not in this "
            "file. NOTE: cms_data.load_theory_prior_z returns z_data[:, :8] and "
            "never reads FDL/component_id, so applying a fraction at load time has "
            "to be BUILT before this file can be trained on.",
        "number_of_events": int(len(z)),
        "artificial_smearing_applied": "false",
        "reweighting_applied": "false",
        "weights_stored_but_not_applied": "true",
        "weight_note": "FDL/weight holds the nominal per-event generator weight "
                       "including the CKKW-L alpha_s and PDF reweighting. It is NOT "
                       "applied to zData. The shipped 2026-08 priors dropped these "
                       "weights entirely and were therefore unweighted.",
        "beam_energy_GeV": 4000.0, "collider_energy_GeV": 8000.0,
        "pdf": "NNPDF31_lo_as_0130", "lhaid": 315200, "pdf_member": 0,
        "MadGraph_version": "3.7.0", "Pythia_version": "8.317",
        "LHAPDF_version": "6.5.6", "HepMC_version": "2.06.09",
        "shower_settings": json.dumps({
            "PartonLevel:ISR": "on", "PartonLevel:FSR": "on",
            "TimeShower:QEDshowerByL": "on", "PartonLevel:MPI": "off",
            "HadronLevel:all": "off"}),
        "pdg_edition": "constants as carried by the recovered UFO restrict cards; "
                       "Gamma(J/psi) = 92.9 keV identifies PDG 2020 "
                       "(PDG 2022+ gives 92.6 keV). UNVERIFIED: no PDG source on "
                       "this machine to check against.",
        "converter": "scripts/prior_build/hepmc_to_prior_hdf5.py",
        "finaliser": "scripts/prior_build/finalise_prior.py",
        "generated_on": f"{platform.node()} {platform.platform()}",
        "production_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for item in args.attr:
        k, _, v = item.partition("=")
        attrs[k] = v

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w", libver="earliest") as fh:
        g = fh.create_group("FDL")
        g.create_dataset("zData", data=z)
        g.create_dataset("component_id", data=cid)
        g.create_dataset("weight", data=weight)
        for k, v in attrs.items():
            fh.attrs[k] = v
            g.attrs[k] = v
    print(f"wrote {out}  zData {z.shape}  components "
          f"{ {int(c): int((cid == c).sum()) for c in np.unique(cid)} }")

    card_sha = {}
    card_root = REPO / "scripts" / "mg5_cards" / "unified" / args.region
    for p in sorted(card_root.rglob("*.dat")):
        card_sha[str(p.relative_to(REPO))] = sha256_of(p)
    model_sha = {}
    for model in ("sm_onia", "sm_mumass", "sm_upsilon_family"):
        d = REPO / "scripts" / "mg5_cards" / "0j1j" / "models" / model
        for p in sorted(d.glob("*.py")) + sorted(d.glob("restrict_c_mass.dat")):
            model_sha[str(p.relative_to(REPO))] = sha256_of(p)

    manifest = dict(
        hdf5=str(out), hdf5_sha256=sha256_of(out), status=args.status,
        attrs=attrs, per_component=per_component,
        card_sha256=card_sha, model_sha256=model_sha,
        scan_report=(json.loads(Path(args.scan_report).expanduser().read_text())
                     if args.scan_report and Path(args.scan_report).expanduser().exists()
                     else None))
    mpath = Path(args.manifest).expanduser() if args.manifest else out.with_suffix(".json")
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
