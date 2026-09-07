#!/usr/bin/env python3
"""HepMC2 ASCII -> prior HDF5 converter (the ``FDL/zData`` writer).

Why this exists
---------------
`lhe_to_prior_hdf5.py` is **not** a substitute for this program.  CKKW-L merging
is applied by Pythia, so the LHE file is the unmerged union of the 0-jet and
1-jet matrix elements: converting there gives a double-counted sample whose pair
pT is matrix-element-only.  The shipped priors record `source_hepmc` and
`event_level = "post-parton-shower stable muons with lepton QED FSR"`, i.e. they
came from showered HepMC.  The program that made them was never committed
(memory.md section 11, Session 28); this is its replacement, written to the
specification in `docs/unified_prior_method.md` sections 9 and 10.

What it does
------------
Streams HepMC2 ASCII (plain or gzipped), reconstructs the per-event particle
graph, extracts one mu-/mu+ pair per event under one of three truth definitions,
applies the **generation** selection only, and writes the method book section 10
schema:

    FDL/zData         [N, 8] float32   mu-(px,py,pz,E), mu+(px,py,pz,E)  GeV
    FDL/component_id  [N]    int8
    FDL/weight        [N]    float64   nominal per-event generator weight

Truth variants (method book section 9), selected with ``--truth-variant``:

``born``     the muon from the hard process, before final-state QED radiation.
             Found by walking back from the status-1 muon through production
             vertices for as long as the vertex has an incoming particle of the
             same PDG id.  The walk therefore stops at the muon that came
             straight out of the resonance or the hard vertex.
``bare``     the status-1 muon after radiation.  This is what the shipped
             priors' own ``event_level`` attribute declares.
``dressed``  the status-1 muon with every status-1 photon inside ``--dressed-dr``
             (default 0.1) recombined.  Hadronisation is off in this recipe, so
             every such photon is a shower photon; there are no hadron-decay
             photons to exclude.

Why ``FDL/weight`` exists, and why you must look at it
------------------------------------------------------
The samples are **not** unweighted after merging.  artifact-measured on the
recovered Upsilon(1S) sample: the nominal weight takes 823 distinct values in
the first 4000 events, with ~79% of events sharing one value.  That is the
CKKW-L alpha_s and PDF reweighting, which is analytic and multiplicative, on top
of the Sudakov part that Pythia applies by veto.  The original converter dropped
these weights -- the shipped priors carry integer component counts and no weight
dataset -- so every distribution in them is the unweighted one.  This converter
stores the weights and ``--report-weighted`` prints both the weighted and the
unweighted statistics, so the size of that approximation is a measured number
rather than an inherited assumption.  Nothing here silently reweights anything.

Mass-convention check (method book section 11 gate 2)
------------------------------------------------------
For every accepted pair the program compares the pair mass computed from the
stored energies against the pair mass computed from the momenta with the muon
mass put back by hand.  A relative disagreement above ``--mass-check-tol``
(default 1e-6) aborts the conversion.  That is the massive-muon check: a model
whose restrict card lost ``13 1.056580e-01 # MM`` produces massless muons in the
record and fails here loudly instead of producing a subtly wrong prior.

Usage
-----
    python scripts/prior_build/hepmc_to_prior_hdf5.py \
        --component jpsi:0:run_a/events.hepmc.gz,run_b/events.hepmc.gz \
        --component continuum:3:cont/events.hepmc.gz \
        --truth-variant bare \
        --gen-pt-min 2.0 --gen-eta-max 2.6 \
        --gen-mass-min 2.787 --gen-mass-max 3.407 \
        --out data/priors/jpsi_unified.hdf5 \
        --manifest data/priors/jpsi_unified.json \
        --attr region=jpsi --attr merging_tms_gev=10.0
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

MUON_PDG = 13
PHOTON_PDG = 22
DEFAULT_MUON_MASS = 0.105658  # the restrict_c_mass.dat value, method book s7
TRUTH_VARIANTS = ("born", "bare", "dressed")


# --------------------------------------------------------------------------
# HepMC2 ASCII streaming reader
# --------------------------------------------------------------------------

class Particle:
    __slots__ = ("barcode", "pdg", "px", "py", "pz", "e", "mass",
                 "status", "end_vtx", "prod_vtx")

    def __init__(self, barcode, pdg, px, py, pz, e, mass, status, end_vtx, prod_vtx):
        self.barcode = barcode
        self.pdg = pdg
        self.px = px
        self.py = py
        self.pz = pz
        self.e = e
        self.mass = mass
        self.status = status
        self.end_vtx = end_vtx
        self.prod_vtx = prod_vtx


class Event:
    __slots__ = ("number", "weight", "weights", "cross_section", "particles")

    def __init__(self):
        self.number = -1
        self.weight = 1.0
        self.weights = []
        self.cross_section = None
        self.particles = {}


def _open_maybe_gzip(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _parse_e_line(fields):
    """Return (event_number, weights).

    HepMC2 IO_GenEvent 'E' line:
      E num mpi scale aQCD aQED sig_proc_id sig_vtx n_vertices beam1 beam2
        n_random [random...] n_weights [weights...]
    """
    number = int(fields[1])
    n_random = int(fields[11])
    idx = 12 + n_random
    n_weights = int(fields[idx])
    weights = [float(v) for v in fields[idx + 1: idx + 1 + n_weights]]
    return number, weights


def _parse_weight_names(line):
    """The 'N' line: N n "name" "name" ... -- return the list of names."""
    out, buf, inside = [], [], False
    for ch in line:
        if ch == '"':
            if inside:
                out.append("".join(buf))
                buf = []
            inside = not inside
        elif inside:
            buf.append(ch)
    return out


def stream_events(path, weight_name="Weight"):
    """Yield one fully parsed Event at a time.  The file is never held in memory."""
    event = None
    current_vertex = None
    pending_in = 0
    pending_out = 0
    weight_index = None
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if not line:
                continue
            tag = line[0]
            if tag == "E" and line[1] == " ":
                if event is not None:
                    yield event
                fields = line.split()
                event = Event()
                event.number, event.weights = _parse_e_line(fields)
                current_vertex = None
                pending_in = pending_out = 0
            elif event is None:
                continue
            elif tag == "N" and line[1] == " ":
                names = _parse_weight_names(line)
                if weight_index is None:
                    weight_index = names.index(weight_name) if weight_name in names else 0
            elif tag == "C" and line[1] == " ":
                fields = line.split()
                event.cross_section = (float(fields[1]), float(fields[2]))
            elif tag == "V" and line[1] == " ":
                fields = line.split()
                current_vertex = int(fields[1])
                pending_in = int(fields[7])
                pending_out = int(fields[8])
            elif tag == "P" and line[1] == " ":
                fields = line.split()
                barcode = int(fields[1])
                # An 'orphan' incoming particle has no production vertex here;
                # only outgoing particles are produced at the current vertex.
                if pending_in > 0:
                    prod = 0
                    pending_in -= 1
                else:
                    prod = current_vertex
                    pending_out -= 1
                event.particles[barcode] = Particle(
                    barcode=barcode,
                    pdg=int(fields[2]),
                    px=float(fields[3]), py=float(fields[4]),
                    pz=float(fields[5]), e=float(fields[6]),
                    mass=float(fields[7]), status=int(fields[8]),
                    end_vtx=int(fields[11]), prod_vtx=prod,
                )
            elif line.startswith("HepMC::IO_GenEvent-END_EVENT_LISTING"):
                break
    if event is not None:
        if weight_index is None:
            weight_index = 0
        yield event
    # note: weight assignment happens in extract_pairs, which knows the index
    return


# --------------------------------------------------------------------------
# Truth-muon extraction
# --------------------------------------------------------------------------

def _delta_r(a, b):
    def eta(p):
        pt = math.hypot(p.px, p.py)
        mag = math.sqrt(pt * pt + p.pz * p.pz)
        if mag - abs(p.pz) <= 0:
            return math.copysign(1e9, p.pz)
        return 0.5 * math.log((mag + p.pz) / (mag - p.pz))
    dphi = math.atan2(a.py, a.px) - math.atan2(b.py, b.px)
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi < -math.pi:
        dphi += 2 * math.pi
    deta = eta(a) - eta(b)
    return math.hypot(deta, dphi)


def _walk_to_born(particle, event, incoming_by_vertex):
    """Walk back through production vertices while the parent is the same PDG."""
    current = particle
    for _ in range(200):  # generous guard against a pathological cycle
        vertex = current.prod_vtx
        if not vertex:
            return current
        parents = incoming_by_vertex.get(vertex, ())
        same = [p for p in parents if p.pdg == current.pdg]
        if not same:
            return current
        current = same[0]
    return current


def extract_pair(event, variant, dressed_dr):
    """Return ((variant pair), (bare pair)) or (None, reason).

    The **bare** pair is always returned alongside the requested variant,
    because the method book section 11 gate 2 mass check has to be applied to
    it and not to the output.  Dressing a muon adds a photon four-vector to it
    and therefore takes it off the muon mass shell **by construction**: the
    dressed object has m > m_mu and an E-vs-p check on it would fail for a
    perfectly correct file.  The question gate 2 actually asks -- did the model
    carry a massive muon, or did the restrict card lose `MM`? -- is a question
    about the status-1 muons in the record, so that is what gets checked.
    """
    stable_muons = [p for p in event.particles.values()
                    if p.status == 1 and abs(p.pdg) == MUON_PDG]
    minus = [p for p in stable_muons if p.pdg == MUON_PDG]
    plus = [p for p in stable_muons if p.pdg == -MUON_PDG]
    if len(minus) != 1 or len(plus) != 1:
        return None, "bad_stable_muon_multiplicity"
    mu_minus, mu_plus = minus[0], plus[0]
    bare = ((mu_minus.px, mu_minus.py, mu_minus.pz, mu_minus.e),
            (mu_plus.px, mu_plus.py, mu_plus.pz, mu_plus.e))

    if variant == "bare":
        return (bare, bare), None

    if variant == "born":
        incoming = {}
        for p in event.particles.values():
            if p.end_vtx:
                incoming.setdefault(p.end_vtx, []).append(p)
        b_minus = _walk_to_born(mu_minus, event, incoming)
        b_plus = _walk_to_born(mu_plus, event, incoming)
        return (((b_minus.px, b_minus.py, b_minus.pz, b_minus.e),
                 (b_plus.px, b_plus.py, b_plus.pz, b_plus.e)), bare), None

    if variant == "dressed":
        photons = [p for p in event.particles.values()
                   if p.status == 1 and p.pdg == PHOTON_PDG]
        out = []
        for mu in (mu_minus, mu_plus):
            px, py, pz, e = mu.px, mu.py, mu.pz, mu.e
            for ph in photons:
                if _delta_r(mu, ph) < dressed_dr:
                    px += ph.px
                    py += ph.py
                    pz += ph.pz
                    e += ph.e
            out.append((px, py, pz, e))
        return ((out[0], out[1]), bare), None

    raise ValueError(f"unknown truth variant {variant!r}")


# --------------------------------------------------------------------------
# Kinematics and checks
# --------------------------------------------------------------------------

def pair_mass_stored(z):
    """Pair mass from the stored energies."""
    e = z[:, 3] + z[:, 7]
    px = z[:, 0] + z[:, 4]
    py = z[:, 1] + z[:, 5]
    pz = z[:, 2] + z[:, 6]
    return np.sqrt(np.clip(e * e - px * px - py * py - pz * pz, 0.0, None))


def pair_mass_momentum(z, muon_mass):
    """Pair mass from the momenta, with the muon mass put back by hand."""
    e1 = np.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2 + z[:, 2] ** 2 + muon_mass ** 2)
    e2 = np.sqrt(z[:, 4] ** 2 + z[:, 5] ** 2 + z[:, 6] ** 2 + muon_mass ** 2)
    e = e1 + e2
    px = z[:, 0] + z[:, 4]
    py = z[:, 1] + z[:, 5]
    pz = z[:, 2] + z[:, 6]
    return np.sqrt(np.clip(e * e - px * px - py * py - pz * pz, 0.0, None))


def transverse_momentum(z, first):
    return np.hypot(z[:, first], z[:, first + 1])


def abs_pseudorapidity(z, first):
    mag = np.sqrt(z[:, first] ** 2 + z[:, first + 1] ** 2 + z[:, first + 2] ** 2)
    ratio = np.clip(z[:, first + 2] / np.maximum(mag, 1e-12), -1 + 1e-7, 1 - 1e-7)
    return np.abs(np.arctanh(ratio))


def stats_block(z, weights=None, nominal_mass=None, muon_mass=DEFAULT_MUON_MASS):
    """The prior_stats.py block, recomputed without the torch dependency.

    Definitions match scripts/prior_build/prior_stats.py exactly: `mass` is the
    'stable' convention (momenta plus the muon mass), `mass_stored` is the
    'stored' convention (stored energies, daughters massless)."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0:
        return {"events": 0}
    mass = pair_mass_momentum(z, muon_mass)
    mass_stored = pair_mass_stored(z)
    muon_pt = np.concatenate([transverse_momentum(z, 0), transverse_momentum(z, 4)])
    abs_eta = np.concatenate([abs_pseudorapidity(z, 0), abs_pseudorapidity(z, 4)])
    pair_pt = np.hypot(z[:, 0] + z[:, 4], z[:, 1] + z[:, 5])
    q75, q25 = np.percentile(mass, [75, 25])
    block = {
        "events": int(len(z)),
        "mass_mean_gev": float(mass.mean()),
        "mass_std_mev": float(mass.std() * 1000.0),
        "mass_median_gev": float(np.median(mass)),
        "mass_iqr_width_mev": float((q75 - q25) / 1.349 * 1000.0),
        "mass_std_stored_energy_mev": float(mass_stored.std() * 1000.0),
        "muon_pt_median_gev": float(np.median(muon_pt)),
        "muon_pt_p99_gev": float(np.percentile(muon_pt, 99)),
        "pair_pt_median_gev": float(np.median(pair_pt)),
        "pair_pt_p99_gev": float(np.percentile(pair_pt, 99)),
        "abs_eta_median": float(np.median(abs_eta)),
        "pair_pt_min_gev": float(pair_pt.min()),
    }
    if nominal_mass is not None:
        for window_mev in (5.0, 10.0):
            block[f"frac_within_{int(window_mev)}mev"] = float(
                np.mean(np.abs(mass - nominal_mass) < window_mev / 1000.0))
    if weights is not None and len(weights) == len(z):
        w = np.asarray(weights, dtype=np.float64)
        if w.sum() > 0:
            block["weighted"] = {
                "sum_of_weights": float(w.sum()),
                "weight_min": float(w.min()),
                "weight_max": float(w.max()),
                "n_distinct_weights": int(np.unique(w).size),
                "mass_mean_gev": float(np.average(mass, weights=w)),
                "pair_pt_median_gev": float(_weighted_quantile(pair_pt, w, 0.5)),
                "muon_pt_median_gev": float(
                    _weighted_quantile(muon_pt, np.concatenate([w, w]), 0.5)),
            }
    return block


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    cdf = np.cumsum(w)
    if cdf[-1] <= 0:
        return float("nan")
    cdf = cdf / cdf[-1]
    return float(np.interp(q, cdf, v))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def convert_component(paths, variant, args, log=print):
    rows, bare_rows, weights = [], [], []
    cutflow = {
        "events_processed": 0,
        "events_bad_stable_muon_multiplicity": 0,
        "events_rejected_by_generation_selection": 0,
        "events_accepted": 0,
    }
    last_cross_section = None
    weight_index_used = None
    for path in paths:
        log(f"    reading {path}")
        n_before = cutflow["events_processed"]
        # Resolve the nominal weight index from the first N line of this file.
        weight_index = _resolve_weight_index(path, args.weight_name)
        if weight_index_used is None:
            weight_index_used = weight_index
        for event in stream_events(path, args.weight_name):
            cutflow["events_processed"] += 1
            if event.cross_section is not None:
                last_cross_section = event.cross_section
            pair, reason = extract_pair(event, variant, args.dressed_dr)
            if pair is None:
                cutflow[f"events_{reason}"] += 1
                continue
            variant_pair, bare_pair = pair
            rows.append(list(variant_pair[0]) + list(variant_pair[1]))
            bare_rows.append(list(bare_pair[0]) + list(bare_pair[1]))
            if event.weights and weight_index < len(event.weights):
                weights.append(event.weights[weight_index])
            else:
                weights.append(1.0)
            if args.max_events and len(rows) >= args.max_events:
                break
        log(f"      {cutflow['events_processed'] - n_before} events read from this file")
        if args.max_events and len(rows) >= args.max_events:
            break

    if not rows:
        raise SystemExit(f"no muon pairs extracted from {paths}")

    z = np.asarray(rows, dtype=np.float64)
    z_bare = np.asarray(bare_rows, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    # ---- gate 2: the mass-convention / massive-muon check, before any cut ----
    # Applied to the BARE muons whatever the output variant is; see extract_pair.
    m_stored = pair_mass_stored(z_bare)
    m_momentum_bare = pair_mass_momentum(z_bare, args.muon_mass)
    m_momentum = pair_mass_momentum(z, args.muon_mass)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(m_stored - m_momentum_bare) / np.maximum(m_stored, 1e-12)
    worst = float(np.nanmax(rel))
    if worst > args.mass_check_tol:
        bad = int(np.sum(rel > args.mass_check_tol))
        raise SystemExit(
            "MASS CONVENTION CHECK FAILED (method book section 11 gate 2).\n"
            f"  worst relative disagreement between the E-based and p-based pair\n"
            f"  mass is {worst:.3e}, tolerance {args.mass_check_tol:.1e}; "
            f"{bad} of {len(z)} pairs exceed it.\n"
            f"  assumed muon mass {args.muon_mass}.  A model whose restrict card\n"
            "  lost `13 1.056580e-01 # MM` produces exactly this failure.\n"
            "  (The check is applied to the BARE muons, so it is independent of\n"
            "   --truth-variant and a dressed run cannot mask or cause it.)")

    # ---- generation selection only.  Never the analysis selection. ----
    keep = np.ones(len(z), dtype=bool)
    if args.gen_pt_min is not None:
        keep &= (transverse_momentum(z, 0) > args.gen_pt_min)
        keep &= (transverse_momentum(z, 4) > args.gen_pt_min)
    if args.gen_eta_max is not None:
        keep &= (abs_pseudorapidity(z, 0) < args.gen_eta_max)
        keep &= (abs_pseudorapidity(z, 4) < args.gen_eta_max)
    if args.gen_mass_min is not None:
        keep &= (m_momentum > args.gen_mass_min)
    if args.gen_mass_max is not None:
        keep &= (m_momentum < args.gen_mass_max)
    cutflow["events_rejected_by_generation_selection"] = int((~keep).sum())
    cutflow["events_accepted"] = int(keep.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_out = np.abs(pair_mass_stored(z) - m_momentum) / np.maximum(
            pair_mass_stored(z), 1e-12)
    return dict(z=z[keep], weights=w[keep], cutflow=cutflow,
                mass_check_worst_relative=worst,
                output_variant_offshell_worst_relative=float(np.nanmax(rel_out)),
                cross_section_last_c_line=last_cross_section,
                weight_index=weight_index_used)


def _resolve_weight_index(path, weight_name):
    """Read just far enough into the file to find the first 'N' line."""
    with _open_maybe_gzip(path) as handle:
        for line in handle:
            if line.startswith("N "):
                names = _parse_weight_names(line)
                return names.index(weight_name) if weight_name in names else 0
            if line.startswith("P "):
                break
    return 0


def parse_component_spec(spec):
    """NAME:ID:PATH[,PATH...]"""
    try:
        name, cid, paths = spec.split(":", 2)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--component must be NAME:ID:PATH[,PATH...], got {spec!r}")
    return dict(name=name, component_id=int(cid),
                paths=[p for p in paths.split(",") if p])


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--component", action="append", required=True,
                   type=parse_component_spec, metavar="NAME:ID:PATH[,PATH...]",
                   help="a labelled component and the HepMC files that make it. "
                        "Repeat for each component; they are written unmixed.")
    p.add_argument("--truth-variant", choices=TRUTH_VARIANTS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest")
    p.add_argument("--dressed-dr", type=float, default=0.1)
    p.add_argument("--muon-mass", type=float, default=DEFAULT_MUON_MASS)
    p.add_argument("--mass-check-tol", type=float, default=1e-6)
    p.add_argument("--weight-name", default="Weight",
                   help="name of the nominal weight in the HepMC 'N' line")
    p.add_argument("--gen-pt-min", type=float, default=None,
                   help="GENERATION muon pT cut.  Not the analysis cut.")
    p.add_argument("--gen-eta-max", type=float, default=None)
    p.add_argument("--gen-mass-min", type=float, default=None)
    p.add_argument("--gen-mass-max", type=float, default=None)
    p.add_argument("--nominal-mass", type=float, default=None,
                   help="resonance mass, only for the frac_within_*mev stats")
    p.add_argument("--max-events", type=int, default=None,
                   help="stop after this many accepted pairs per component (testing)")
    p.add_argument("--component-xsec", action="append", default=[],
                   metavar="NAME=VALUE", help="cross section in pb, per component")
    p.add_argument("--attr", action="append", default=[], metavar="KEY=VALUE",
                   help="extra provenance attribute written to the HDF5 root")
    p.add_argument("--cards-dir", action="append", default=[],
                   help="directory whose files are sha256'd into the manifest")
    p.add_argument("--report-weighted", action="store_true",
                   help="also print weighted statistics, to size the effect of "
                        "dropping the CKKW-L weights")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    xsec = {}
    for item in args.component_xsec:
        k, _, v = item.partition("=")
        xsec[k] = float(v)
    extra = {}
    for item in args.attr:
        k, _, v = item.partition("=")
        extra[k] = v

    print(f"truth variant: {args.truth_variant}")
    all_z, all_id, all_w = [], [], []
    per_component = {}
    for comp in args.component:
        print(f"  component {comp['name']} (id {comp['component_id']})")
        result = convert_component(comp["paths"], args.truth_variant, args)
        n = len(result["z"])
        all_z.append(result["z"])
        all_w.append(result["weights"])
        all_id.append(np.full(n, comp["component_id"], dtype=np.int8))
        per_component[comp["name"]] = dict(
            component_id=comp["component_id"],
            sources=[str(Path(p)) for p in comp["paths"]],
            source_sha256={str(Path(p)): sha256_of(p) for p in comp["paths"]},
            cutflow=result["cutflow"],
            mass_check_worst_relative=result["mass_check_worst_relative"],
            mass_check_applied_to="bare muons (see extract_pair docstring)",
            output_variant_offshell_worst_relative=result[
                "output_variant_offshell_worst_relative"],
            cross_section_pb=xsec.get(comp["name"]),
            cross_section_last_c_line=result["cross_section_last_c_line"],
            weight_index=result["weight_index"],
            prior_stats=stats_block(result["z"], result["weights"],
                                    args.nominal_mass, args.muon_mass),
        )
        cf = result["cutflow"]
        print(f"    processed {cf['events_processed']}, accepted {cf['events_accepted']}, "
              f"bad multiplicity {cf['events_bad_stable_muon_multiplicity']}, "
              f"rejected by generation selection "
              f"{cf['events_rejected_by_generation_selection']}")
        print(f"    gate 2 mass-convention check (on bare muons): worst relative "
              f"deviation {result['mass_check_worst_relative']:.3e} "
              f"(tolerance {args.mass_check_tol:.1e}) PASS")
        print(f"    output-variant off-shellness (information, not a gate): "
              f"{result['output_variant_offshell_worst_relative']:.3e}")
        if args.report_weighted:
            block = per_component[comp["name"]]["prior_stats"].get("weighted")
            if block:
                print(f"    unweighted pair-pT median "
                      f"{per_component[comp['name']]['prior_stats']['pair_pt_median_gev']:.4f}"
                      f"   weighted {block['pair_pt_median_gev']:.4f}"
                      f"   ({block['n_distinct_weights']} distinct weights)")

    z = np.concatenate(all_z).astype(np.float32)
    cid = np.concatenate(all_id)
    weights = np.concatenate(all_w)

    attrs = {
        "schema_version": "unified-prior-1",
        "truth_variant": args.truth_variant,
        "truth_variant_definition": {
            "born": "muon from the hard process, before final-state QED radiation",
            "bare": "status-1 muon after radiation",
            "dressed": f"status-1 muon with status-1 photons inside dR < {args.dressed_dr}",
        }[args.truth_variant],
        "dressed_dr": args.dressed_dr,
        "columns": "mu_minus_px,mu_minus_py,mu_minus_pz,mu_minus_E,"
                   "mu_plus_px,mu_plus_py,mu_plus_pz,mu_plus_E",
        "particle_order": "mu- (PDG 13) first; mu+ (PDG -13) second",
        "four_vector_convention": "(px, py, pz, E)",
        "units": "GeV",
        "daughter_muon_mass_GeV": args.muon_mass,
        "generation_cuts": json.dumps({
            "lepton_pt_min_GeV": args.gen_pt_min,
            "lepton_abs_eta_max": args.gen_eta_max,
            "mass_window_GeV": [args.gen_mass_min, args.gen_mass_max],
        }),
        "selection": "NONE - the analysis selection is an offline choice "
                     "(method book section 1). Only the generation selection is applied.",
        "components_are_mixed": "false",
        "component_names": json.dumps([c["name"] for c in args.component]),
        "component_id_mapping": json.dumps(
            {c["name"]: c["component_id"] for c in args.component}),
        "component_counts": json.dumps(
            {name: int(v["cutflow"]["events_accepted"]) for name, v in per_component.items()}),
        "component_cross_sections_pb": json.dumps(
            {name: v["cross_section_pb"] for name, v in per_component.items()}),
        "number_of_events": int(len(z)),
        "artificial_smearing_applied": "false",
        "reweighting_applied": "false",
        "weights_stored_but_not_applied": "true",
        "weight_note": "FDL/weight holds the nominal per-event generator weight, "
                       "including the CKKW-L alpha_s and PDF reweighting. It is "
                       "NOT applied to zData. The shipped 2026-08 priors dropped "
                       "these weights entirely; see the manifest for the size.",
        "converter": "scripts/prior_build/hepmc_to_prior_hdf5.py",
        "production_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    attrs.update(extra)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # libver="earliest" pins the most backward-compatible on-disk format.  This
    # VM has h5py 3.16.0 on HDF5 2.0.0; the Windows `cms` environment that will
    # train on these files has an older HDF5, and a file it cannot open is a
    # file that does not exist.  data/priors/_compat_probe.hdf5 is the check.
    with h5py.File(out, "w", libver="earliest") as fh:
        group = fh.create_group("FDL")
        group.create_dataset("zData", data=z)
        group.create_dataset("component_id", data=cid)
        group.create_dataset("weight", data=weights)
        for key, value in attrs.items():
            fh.attrs[key] = value
            group.attrs[key] = value
    print(f"wrote {out}  zData {z.shape}  component_id {cid.shape}")

    manifest = dict(
        hdf5=str(out), hdf5_sha256=sha256_of(out), attrs=attrs,
        per_component=per_component,
        card_sha256={}, argv=sys.argv[1:],
    )
    for directory in args.cards_dir:
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file():
                manifest["card_sha256"][str(path)] = sha256_of(path)
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2, default=str))
        print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
