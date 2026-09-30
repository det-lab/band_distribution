"""
Choose the node counts of a normalization table from the problem itself.

How many Chebyshev nodes an axis needs depends on the fit region (the same
box needed 5 nodes in q0 for Eq 4-100 and ~16 for Eq 0.75-200), so instead of
carrying hand-derived counts, measure them:

  points    write a "study" spec: for each axis, at a few baseline points in
            the box, 33 Chebyshev-Lobatto nodes along that axis (the others
            held fixed).  Evaluate it like any other spec (normgrid.py run).
  analyze   read the study results and, per axis, find the fewest nodes whose
            Chebyshev series is converged to --tol; write the grid spec
            (normgrid.py make-spec's output) plus a report.

Why 33 nodes and why it is enough: Lobatto nodes are nested (the 3, 5, 9 and
17 node sets are subsets of the 33 node set), so one evaluation per node
covers every coarser grid.  The Chebyshev coefficients c_j of the 33 node
interpolant say how well degree m suffices: the truncation error is at most
sum_{j>m} |c_j| (|T_j| <= 1).  An axis needs n = m + 1 nodes for the smallest
m whose tail is below tol / safety.  An axis that is not resolved by 33 nodes
(rough or nearly singular in the box) is an error, not a guess.

This sees one axis at a time, so it cannot see cross terms between axes;
`normgrid.py validate` against held-out points remains the final check.
slurm/build_table.sh chains all of it.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from numpy.polynomial import chebyshev as cheb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import normgrid as ng

STUDY_NODES = 33            # 2**5 + 1: nested Lobatto grids of 3, 5, 9, 17, 33 nodes


class NotConverged(RuntimeError):
    """An axis is not resolved to the requested tolerance by STUDY_NODES nodes."""


def baseline_fractions(axes, n_baselines):
    """Where each axis sits in its box for each baseline: the centre, then
    staggered 20%/80% points (no axis is ever at a symmetry point of all the
    others), then reproducible random ones if more are asked for."""
    fr = [[0.5] * len(axes)]
    for b in range(1, n_baselines):
        if b < 3:
            fr.append([0.2 if (i + b) % 2 == 0 else 0.8 for i in range(len(axes))])
        else:
            fr.append(list(np.random.default_rng(b).uniform(0.1, 0.9, len(axes))))
    return fr[:n_baselines]


def make_study_spec(band, region, epsrel, *, box=None, fixed=ng.DEFAULT_FIXED, n_baselines=3):
    """A spec of kind "points": for axis i, baseline b, node j the point with
    axis i at its j-th node and every other axis at the baseline.  Point
    index = (i * n_baselines + b) * STUDY_NODES + j."""
    spec = ng._base_spec("points", band, region, fixed, ng.default_box(band) if box is None else box,
                         epsrel, 1e-13)
    axes = ng.BAND_AXES[band]
    fractions = baseline_fractions(axes, n_baselines)
    points = []
    for a in axes:
        lo, hi = spec["box"][a]
        for fr in fractions:
            base = {x: spec["box"][x][0] + f * (spec["box"][x][1] - spec["box"][x][0])
                    for x, f in zip(axes, fr)}
            for node in ng.chebyshev_lobatto(lo, hi, STUDY_NODES):
                points.append({**base, a: float(node)})
    spec["points"] = points
    spec["study"] = {"axes": list(axes), "n_baselines": n_baselines, "n_nodes": STUDY_NODES}
    return spec


def tails(values):
    """tail[m] = sum_{j>m} |c_j| / min|values| for the degree-32 interpolant
    of `values` at the 33 ascending Lobatto nodes.  Scaled by the smallest
    value because the table needs RELATIVE accuracy everywhere in the box,
    including where the normalization is smallest."""
    x = -np.cos(np.pi * np.arange(STUDY_NODES) / (STUDY_NODES - 1))
    c = np.abs(cheb.chebfit(x, values, STUDY_NODES - 1))
    return np.cumsum(c[::-1])[::-1][1:] / np.min(np.abs(values))      # tail[m] for m = 0 .. 31


def nodes_needed(values, target, min_nodes):
    """Fewest nodes n = m + 1 with tail[m] <= target (n >= min_nodes); also
    returns that tail.  Raises NotConverged if no degree below the top one
    qualifies."""
    t = tails(values)
    for m in range(min_nodes - 1, len(t)):
        if t[m] <= target:
            return m + 1, float(t[m])
    raise NotConverged(f"smallest tail {t[-1]:.1e} at degree {len(t) - 1} is still above {target:.1e}")


def analyze(study, result_paths, tol, safety=2.0, min_nodes=3, max_points=500_000):
    """Study results -> (grid spec, report dict)."""
    vals, _ = ng.gather_results(study, result_paths)
    bad = np.flatnonzero(np.isnan(vals))
    if len(bad):
        raise RuntimeError(f"{len(bad)} of {len(vals)} study points missing or failed (first: "
                           f"{bad[:10].tolist()}); re-run them (normgrid.py run --retry-failed)")
    axes, nb = study["study"]["axes"], study["study"]["n_baselines"]
    per_axis, report_axes, unresolved = {}, {}, []
    for i, a in enumerate(axes):
        ns, ts = [], []
        for b in range(nb):
            s = (i * nb + b) * STUDY_NODES
            try:
                n, t = nodes_needed(vals[s:s + STUDY_NODES], tol / safety, min_nodes)
            except NotConverged as e:
                unresolved.append(f"{a} (baseline {b}): {e}")
                continue
            ns.append(n)
            ts.append(t)
        if ns:
            per_axis[a] = max(ns)
        report_axes[a] = {"n": per_axis.get(a), "per_baseline": ns, "tail_at_n": ts}
    if unresolved:
        raise NotConverged(f"not resolved by {STUDY_NODES} nodes at tol {tol:.1e} / safety {safety}:\n  "
                           + "\n  ".join(unresolved)
                           + "\nNarrow that axis's box, loosen --tol, or raise epsrel's accuracy "
                             "(integration noise sets a floor near epsrel)")
    total = int(np.prod(list(per_axis.values())))
    report = {"tol": tol, "safety": safety, "min_nodes": min_nodes, "band": study["band"],
              "region": study["region"], "box": study["box"], "epsrel": study["epsrel"],
              "points": total, "axes": report_axes}
    if total > max_points:
        raise RuntimeError(f"the grid would have {total:,} points (> --max-points {max_points:,}): "
                           + ", ".join(f"{a}={n}" for a, n in per_axis.items()))
    spec = ng.make_grid_spec(study["band"], per_axis, region=tuple(study["region"]), fixed=study["fixed"],
                             box={a: tuple(v) for a, v in study["box"].items()}, epsrel=study["epsrel"],
                             epsabs=study["epsabs"])
    return spec, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("points", help="write the study spec")
    p.add_argument("--band", required=True, choices=["NR", "ER"])
    p.add_argument("--region", nargs=4, type=float, required=True, metavar=("EP_MIN", "EP_MAX", "EQ_MIN", "EQ_MAX"))
    p.add_argument("--epsrel", type=float, required=True, help="per-point quadrature tolerance, e.g. 1e-7")
    p.add_argument("--baselines", type=int, default=3)
    p.add_argument("--box", nargs="+", default=None, metavar="AXIS=LO:HI",
                   help="replace axes of the band's default box, e.g. V=2.5:4 dq=0:0.42")
    p.add_argument("--out", required=True)

    p = sub.add_parser("analyze", help="study results -> grid spec + report")
    p.add_argument("--study", required=True)
    p.add_argument("--results", nargs="+", required=True, help="result files or globs")
    p.add_argument("--tol", type=float, default=1e-7, help="per-axis interpolation error target (default 1e-7)")
    p.add_argument("--safety", type=float, default=2.0,
                   help="require the coefficient tail below tol/safety: the tail bounds the truncation "
                        "error of the 33 node series, not of the coarser interpolant (default 2)")
    p.add_argument("--min-nodes", type=int, default=3)
    p.add_argument("--max-points", type=int, default=500_000)
    p.add_argument("--out-spec", required=True)
    p.add_argument("--report", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "points":
        spec = make_study_spec(a.band, tuple(a.region), a.epsrel, n_baselines=a.baselines,
                               box=ng.parse_box(a.box, a.band) if a.box else None)
        ng.save_spec(spec, a.out)
        print(f"{a.out}: {ng.n_points(spec)} study points ({len(spec['study']['axes'])} axes x "
              f"{a.baselines} baselines x {STUDY_NODES} nodes)")
    else:
        paths = sorted({f for pat in a.results for f in (glob.glob(pat) or [pat])})
        spec, report = analyze(ng.load_spec(a.study), paths, a.tol, a.safety, a.min_nodes, a.max_points)
        ng.save_spec(spec, a.out_spec)
        with open(a.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"{spec['band']} region {spec['region']}: tol {a.tol:.0e} (tail <= tol/{a.safety:g}) -> "
              f"{report['points']:,} grid points")
        for ax, r in report["axes"].items():
            print(f"  {ax:3s} {r['n']:3d} nodes   (per baseline {r['per_baseline']}, tail at n "
                  f"{', '.join(f'{t:.0e}' for t in r['tail_at_n'])})")
        print(f"wrote {a.out_spec} and {a.report}")


if __name__ == "__main__":
    main()
