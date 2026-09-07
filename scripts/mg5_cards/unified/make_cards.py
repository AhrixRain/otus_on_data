#!/usr/bin/env python3
"""Emit the unified MG5 / Pythia8 cards from the method book parameters.

Every number this writes comes from `docs/unified_prior_method.md` v2.1 --
section 6 (run card and Pythia card), section 7 (models), section 8 (windows and
thresholds) and section 12 (per-region specs).  The cards are generated rather
than hand-written so that the method book stays the single source of truth and a
reader can diff the output against it.

The run card is produced by substituting into a **recovered original** run card
(`scripts/mg5_cards/0j1j/<sample>/run_card.dat`), so every field this method does
not deliberately change keeps the value the original priors were made with, and
`diff` against the original shows exactly the intended changes and nothing else.

What varies per region (method book section 2: only three things genuinely do):
the model, the mass window and the muon pT threshold.  Everything else is one
recipe.

Usage:
    python scripts/mg5_cards/unified/make_cards.py --out scripts/mg5_cards/unified
    python scripts/mg5_cards/unified/make_cards.py --out ... --tms 10.0 --region jpsi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RECOVERED = REPO / "scripts" / "mg5_cards" / "0j1j"

# --------------------------------------------------------------------------
# Method book section 8: windows and thresholds.
# --------------------------------------------------------------------------
REGIONS = {
    "jpsi": dict(
        generation_window=(2.787, 3.407),
        analysis_window_new=(2.942, 3.252),
        analysis_window_legacy=(3.0369, 3.1569),
        analysis_pt=3.0, generation_ptl=2.0,
        tms_scan=[5.0, 10.0, 20.0], tms_rule=10.0,
        components=[
            dict(name="jpsi", component_id=0, kind="signal",
                 model="sm_onia-c_mass", resonance="jpsiv", mass=3.0969),
            dict(name="continuum", component_id=3, kind="continuum",
                 model="sm_mumass-c_mass", resonance=None, mass=None),
        ],
    ),
    "z": dict(
        generation_window=(65.0, 115.0),
        analysis_window_new=(83.70, 98.68),
        analysis_window_legacy=(70.0, 110.0),
        analysis_pt=25.0, generation_ptl=15.0,
        tms_scan=[5.0, 11.4, 22.8, 45.6], tms_rule=22.8,
        components=[
            dict(name="continuum", component_id=3, kind="continuum",
                 model="sm_mumass-c_mass", resonance=None, mass=None),
        ],
    ),
    "upsilon": dict(
        generation_window=(8.4, 11.6),
        analysis_window_new=(8.987, 10.873),
        analysis_window_legacy=(8.5, 11.5),
        analysis_pt=3.0, generation_ptl=2.0,
        tms_scan=[5.0, 10.0, 20.0], tms_rule=10.0,
        components=[
            dict(name="upsilon1s", component_id=0, kind="signal",
                 model="sm_upsilon_family-c_mass", resonance="upsilonv", mass=9.4604),
            dict(name="upsilon2s", component_id=1, kind="signal",
                 model="sm_upsilon_family-c_mass", resonance="upsilon2v", mass=10.0234),
            dict(name="upsilon3s", component_id=2, kind="signal",
                 model="sm_upsilon_family-c_mass", resonance="upsilon3v", mass=10.3551),
            dict(name="continuum", component_id=3, kind="continuum",
                 model="sm_mumass-c_mass", resonance=None, mass=None),
        ],
    ),
}

# Method book section 6.  `ptj` is fixed at 2.5 so it sits below the smallest
# TMS in every scan; `ktdurham` tracks TMS, because ktdurham is the cut that
# actually restricts the 1-jet matrix element to the region above the merging
# scale.  Setting ptj below ktdurham is what makes the merging variable, and not
# the generation cut, define the boundary -- section 6.1 records that the hole
# this is insurance against was looked for and not found.
PTJ = 2.5
ETAL = 2.6
ETAJ = 5.0
DRJL = 0.0
DRLL = 0.0

# Template run card for each component: the region's OWN recovered original.
# This matters.  Using one template for everything would silently import the
# J/psi card's layout into the Z card -- the recovered DY card has no
# `cut_decays` line and carries `asrwgtflavor 4` where every other card carries
# 5 -- and the diff against the original would then show layout churn mixed in
# with the intended changes.  Templating each region from its own original means
# `diff <recovered> <unified>` shows exactly the method book's changes and
# nothing else, per region, which is the only form in which that diff is
# evidence of anything.
TEMPLATE = {
    ("jpsi", "jpsi"): "jpsi_signal",
    ("jpsi", "continuum"): "jpsi_continuum",
    ("z", "continuum"): "dy",
    ("upsilon", "upsilon1s"): "upsilon/upsilon1s",
    ("upsilon", "upsilon2s"): "upsilon/upsilon2s",
    ("upsilon", "upsilon3s"): "upsilon/upsilon3s",
    ("upsilon", "continuum"): "upsilon/continuum",
}

RUN_CARD_SUBSTITUTIONS = {
    "ptj": PTJ, "ptl": None, "etal": ETAL, "etaj": ETAJ,
    "drjl": DRJL, "drll": DRLL, "mmll": None, "mmllmax": None,
    "ktdurham": None, "nevents": None, "iseed": None,
    "ickkw": 0, "xqcut": -1.0, "maxjetflavor": 4,
}


def substitute_run_card(text, values):
    """Replace `  <value>\t= <key> ! comment` lines, leaving everything else."""
    out, seen = [], set()
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            rhs = stripped.split("=", 1)[1]
            key = rhs.split("!", 1)[0].strip()
            if key in values and values[key] is not None:
                comment = ("!" + rhs.split("!", 1)[1]) if "!" in rhs else ""
                out.append(f"  {values[key]}\t= {key} {comment}\n")
                seen.add(key)
                continue
        out.append(line)
    missing = {k for k, v in values.items() if v is not None} - seen
    if missing:
        raise SystemExit(f"run-card keys not found in template: {sorted(missing)}")
    return "".join(out)


PROC_CARD = """\
set group_subprocesses Auto
set ignore_six_quark_processes False
set low_mem_multicore_nlo_generation False
set complex_mass_scheme False
set include_lepton_initiated_processes False
set gauge unitary
set loop_optimized_output True
set loop_color_flows False
set max_npoint_for_channel 0
set default_unset_couplings 99
set max_t_for_channel 99
set zerowidth_tchannel True
set nlo_mixed_expansion True
import model {model} --modelname
define j = g u c d s u~ c~ d~ s~
define p = g u c d s u~ c~ d~ s~
{generate}
output {output_dir}
"""

PYTHIA_CARD = """\
! Unified prior Pythia8 INPUT card, method book section 6.
! Region {region}, component {component}.
!
! This is the card MG5 reads, NOT the resolved `tag_1_pythia8.cmd` MG5 writes.
! The two are easy to confuse and the confusion is fatal: the resolved card
! carries MG5-owned parameters such as `JetMatching:setMad` and `Beams:LHEF`,
! and feeding those back in makes MG5 abort with
!   "The parameter JetMatching:setMad is already set to False."
! artifact-measured 2026-09-06 -- that is exactly how the first shakedown run
! failed.  So this card carries only what a user is allowed to set, and matches
! the recovered originals field for field.
!
! Merging:TMS = -1.0 means "take it from the run card's ktdurham".  The merging
! scan is therefore driven entirely by `ktdurham` in the run card, and TMS
! cannot silently disagree with the matrix-element cut.  Merging:doKTMerging and
! Merging:Dparameter are set by MG5 from ktdurham and dparameter.
!
! MPI-off and hadronisation-off are inherited from the recovered original for
! continuity and are declared DELIBERATELY, not silently.  MPI affects pair pT,
! which is a conditioning feature of the response, so MPI-on is a systematic
! variation to run once -- not a setting to leave undocumented.
!
! Merging:Process = guess is kept because it is what the original used.  What
! Pythia actually resolves must be read out of the run log and recorded; "guess"
! is not a provenance record.  artifact-measured 2026-09-06 on the J/psi signal
! reshower: Pythia resolved it to "Les Houches User Process(es) code 9999, 2 -> 2"
! with m3Hat = 3.097 (the resonance) and m4Hat = 0 (the light parton).
!
Main:numberOfEvents       = -1
HEPMCoutput:file          = hepmc.gz
Merging:TMS               = -1.0
Merging:Process           = guess
Merging:nJetMax           = 1
PartonLevel:ISR           = on
PartonLevel:FSR           = on
TimeShower:QEDshowerByL   = on
PartonLevel:MPI           = off
HadronLevel:all           = off
Random:setSeed            = on
Random:seed               = {seed}
"""


def generate_lines(component):
    if component["kind"] == "signal":
        r = component["resonance"]
        return (f"generate    p p > {r}, {r} > mu+ mu- @0\n"
                f"add process p p > {r} j, {r} > mu+ mu- @1")
    return ("generate    p p > mu+ mu- @0\n"
            "add process p p > mu+ mu- j @1")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", action="append", choices=sorted(REGIONS))
    ap.add_argument("--tms", type=float, default=None,
                    help="override the region's rule TMS (used for scan points)")
    ap.add_argument("--nevents", type=int, default=100000)
    ap.add_argument("--seed-base", type=int, default=20260906)
    ap.add_argument("--tag", default="production")
    args = ap.parse_args()

    out_root = Path(args.out)
    index = {}
    for region in (args.region or sorted(REGIONS)):
        spec = REGIONS[region]
        tms = args.tms if args.tms is not None else spec["tms_rule"]
        for i, comp in enumerate(spec["components"]):
            d = out_root / region / comp["name"] / args.tag
            d.mkdir(parents=True, exist_ok=True)
            seed = args.seed_base + 1000 * list(REGIONS).index(region) + 10 * i
            template = (RECOVERED / TEMPLATE[(region, comp["name"])]
                        / "run_card.dat").read_text()
            values = dict(RUN_CARD_SUBSTITUTIONS)
            values.update({
                "ptl": spec["generation_ptl"],
                "mmll": spec["generation_window"][0],
                "mmllmax": spec["generation_window"][1],
                "ktdurham": tms,
                "nevents": args.nevents,
                "iseed": seed,
            })
            (d / "run_card.dat").write_text(substitute_run_card(template, values))
            (d / "proc_card_mg5.dat").write_text(PROC_CARD.format(
                model=comp["model"], generate=generate_lines(comp),
                output_dir=f"~/mg5work/proc/{region}_{comp['name']}"))
            (d / "pythia8_card.dat").write_text(PYTHIA_CARD.format(
                region=region, component=comp["name"], seed=seed))
            index[f"{region}/{comp['name']}/{args.tag}"] = dict(
                region=region, component=comp["name"],
                component_id=comp["component_id"], kind=comp["kind"],
                model=comp["model"], resonance=comp["resonance"],
                nominal_mass=comp["mass"], tms=tms, ptj=PTJ, ptl=spec["generation_ptl"],
                etal=ETAL, generation_window=list(spec["generation_window"]),
                analysis_window_new=list(spec["analysis_window_new"]),
                analysis_window_legacy=list(spec["analysis_window_legacy"]),
                analysis_pt=spec["analysis_pt"], nevents=args.nevents, seed=seed,
                path=str(d))
            print(f"wrote {d}")
    idx = out_root / f"index_{args.tag}.json"
    idx.write_text(json.dumps(index, indent=2))
    print(f"wrote {idx}")


if __name__ == "__main__":
    main()
