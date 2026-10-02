"""
Precomputed, interpolated region-normalization tables for MCMC.

The region-normalization integral (_ppqfort_bindings.ppqn_region/ppqg_region) is
~2 s per call -- fine for one fit, hopeless inside an MCMC.  But the region
is fixed for a whole run and the integral is a very smooth function of the
handful of physics parameters the MCMC varies, so it can be evaluated once
on a small tensor grid (embarrassingly parallel: one independent integral
per grid point, e.g. on the OSG) and interpolated in ~tens of microseconds
per step.  This module is the whole pipeline:

  make-spec / make-random-spec   describe a grid (or a set of held-out
                                 validation points) as a small JSON file
  chunks                         split it into (start, stop) ranges, one per
                                 batch job
  run                            worker: evaluate a range of points, crash-
                                 tolerant and resumable (Fortran `error stop`
                                 kills the process, so a supervisor restarts
                                 past the failing point and records it)
  merge                          combine result files into one HDF5 table
  validate                       compare the interpolant against directly
                                 computed held-out points
  NormInterpolator               load the table and evaluate it (numpy only)

Coordinates.  The physical parameters are mapped to axes that make the valid
region a rectangle: (k, F0, V, p0, dp = p10 - p0, q0, dq = q10 - q0).  F0 is used
*linearly*, not as log F0: the normalization is smooth in F0 (it enters as a
variance) but not in log F0, and measured on this problem 3 Chebyshev nodes in
F0 interpolate to ~3e-9 where log10 F0 needs 9+ nodes for 1e-7.  The
resolution model sigp^2 = p0^2 + (p10^2 - p0^2)(Ep/c)^2 is only defined for
p10 >= p0 (dp >= 0; likewise q10 >= q0, dq >= 0), so a plain (p0, p10) or
(q0, q10) box would contain unphysical corners (the q0 and q10 ranges
overlap, so this matters for q; for the default p ranges p10 > p0 anyway).  The ER band does not depend on k.

Interpolation is a tensor-product polynomial through Chebyshev-Lobatto nodes,
built and evaluated with numpy.polynomial.chebyshev (chebfit/chebval), exact
at the nodes; N nodes on an axis means a degree N-1 polynomial along it.  It never extrapolates: a query outside the
table's box raises OutOfBoxError.

Usage:  python python/cli/normgrid.py --help   (run with LD_LIBRARY_PATH=lib so the
Fortran library loads, as with the other python/cli/ entry points)
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from numpy.polynomial import chebyshev as cheb

# _ppqfort_bindings.py is a sibling package, not a sibling file, now that the
# CLI tools and the private ctypes layer live in their own subdirectories.
_INTERNAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "internal")

AXES = ("k", "F0", "V", "p0", "dp", "q0", "dq")
BAND_AXES = {"NR": AXES, "ER": tuple(a for a in AXES if a != "k")}

# Bounding rectangle, in axis coordinates, of the MCMC prior.  p0 and q0 have
# Gaussian priors centred on P0_MEAN/Q0_MEAN with a 20% (1 sigma) width, so
# their boxes are the +/-4 sigma range, i.e. mean * [0.2, 1.8] (the sampler
# truncates the prior there).  p10 in [0.3, 0.6] and q10 in [0.2, 0.4] give
# dp = p10 - p0 in [0.3 - 0.1156, 0.6 - 0.0128] and dq = q10 - q0 in
# [0, 0.4 - 0.0474]; q0 is capped at 0.4 (= max q10) since q10 >= q0.  These
# rectangles also contain valid points outside the prior (e.g. p10 = 0.19) --
# harmless, the interpolant is just not used there.
P0_MEAN, Q0_MEAN = 0.06421907, 0.23718488
DEFAULT_BOX = {
    "k": (0.13, 0.22),
    "F0": (1e-5, 1.0),
    "V": (2.7, 3.3),
    "p0": (0.2 * P0_MEAN, 1.8 * P0_MEAN),
    "dp": (0.18, 0.59),
    "q0": (0.2 * Q0_MEAN, 0.4),
    "dq": (0.0, 0.353),
}
DEFAULT_FIXED = {"Z": 32.0, "eps": 3.0e-3}

# Per-band changes to DEFAULT_BOX.  ER: the effective electron-recoil Fano
# factor at the low fields these detectors run at is ~0.2-0.3 (CDMSlite:
# 0.21-0.29), above the literature 0.13, so the ER box brackets that instead
# of the NR box's 1e-5..1.
BAND_BOX_OVERRIDES = {"ER": {"F0": (0.1, 0.35)}}


def default_box(band):
    """DEFAULT_BOX with the band's overrides applied."""
    return {**DEFAULT_BOX, **BAND_BOX_OVERRIDES.get(band, {})}


def x10_ranges(box):
    """The widest p10 and q10 ranges a box covers together with its full p0
    and q0 ranges, for a prior that enforces p10 >= p0 and q10 >= q0:
    ((p10_lo, p10_hi), (q10_lo, q10_hi)).  Every p10 in the range with every
    p0 must give dp = p10 - p0 inside the dp range: p10_lo >= p0_hi + dp_lo
    unless dp_lo == 0 (then p10 >= p0 is the only lower limit), and
    p10_hi <= p0_lo + dp_hi."""
    out = []
    for x0, d in (("p0", "dp"), ("q0", "dq")):
        (x0lo, x0hi), (dlo, dhi) = box[x0], box[d]
        out.append((float(x0hi + dlo if dlo > 0 else x0lo), float(x0lo + dhi)))
    return tuple(out)

# Node counts per axis that keep the worst-case interpolation error of the
# default box near 1e-7 per axis (~1e-6 total, i.e. ~0.02 in the log-
# likelihood at 20,000 events).  They depend on the REGION: these come from
# one-axis-at-a-time studies for the analysis ROI Ep 2.5-350 / Eq 0.75-200
# against directly computed points (Chebyshev-Lobatto nodes, epsrel=1e-7,
# worst of three baselines spanning the box).  Smallest n with error <= 1e-7:
#   NR  k 8, F0 3, V 4, p0 4, dp 3, q0 ~16, dq 4
#   ER  F0 2, V 8, p0 4, dp 6, q0 10, dq 8
# with a margin on F0, ER V/dq/q0 and NR q0.  NR q0 is the hard axis: the
# error falls only ~x4 per two nodes (1e-6 at n=12), so 16 is an
# extrapolation -- check it with `validate`.  (For the earlier 2-200 / 4-100
# region q0 needed only 5 nodes.)  One-axis-at-a-time studies cannot see
# cross terms, so validate any table against held-out points.  For another
# region do not reuse these: slurm/build_table.sh (python/cli/normplan.py) measures
# the counts for the region it is given -- it reproduces the NR counts below
# and the ER ones to within a node.
RECOMMENDED_NODES = {
    "NR": {"k": 8, "F0": 3, "V": 4, "p0": 4, "dp": 3, "q0": 16, "dq": 4},     # 73,728 points
    "ER": {"F0": 3, "V": 9, "p0": 4, "dp": 6, "q0": 11, "dq": 9},             # 64,152 points
}
DEFAULT_REGION = (2.0, 200.0, 4.0, 100.0)


class OutOfBoxError(ValueError):
    """A query lies outside the box a table was built for.  Carries the axis,
    its value and range, the full query and the table's file, so a crash in
    the middle of an MCMC run says everything needed to act on it."""

    def __init__(self, message, *, axis=None, value=None, box=None, query=None, source=""):
        super().__init__(message)
        self.axis, self.value, self.box, self.query, self.source = axis, value, box, query, source


class PriorNotCoveredError(ValueError):
    """An MCMC prior (or a set of starting points) reaches outside a table's
    box; raised at setup by NormInterpolator.check_prior / check_points."""


# ---------------------------------------------------------------------------
# Specs: a grid, or a set of random held-out points, as plain JSON-able dicts
# ---------------------------------------------------------------------------

def _base_spec(kind, band, region, fixed, box, epsrel, epsabs):
    if band not in BAND_AXES:
        raise ValueError(f"band must be 'NR' or 'ER', got {band!r}")
    box = {a: list(box[a]) for a in BAND_AXES[band]}
    fixed = dict(fixed)
    if band == "ER":
        fixed.pop("Z", None)
    return {"kind": kind, "band": band, "region": list(region), "fixed": fixed,
            "box": box, "epsrel": epsrel, "epsabs": epsabs}


def make_grid_spec(band, n_nodes, *, region=DEFAULT_REGION, fixed=DEFAULT_FIXED,
                   box=None, epsrel=1e-7, epsabs=1e-13):
    """n_nodes: {axis: number of Chebyshev-Lobatto nodes} for every axis of
    the band (1 pins an axis at its midpoint).  box: None = default_box(band)."""
    spec = _base_spec("grid", band, region, fixed, default_box(band) if box is None else box,
                      epsrel, epsabs)
    axes = BAND_AXES[band]
    if set(n_nodes) != set(axes):
        raise ValueError(f"n_nodes must give exactly the axes {axes}, got {sorted(n_nodes)}")
    if any(int(n_nodes[a]) < 1 for a in axes):
        raise ValueError("every axis needs at least 1 node")
    spec["n_nodes"] = {a: int(n_nodes[a]) for a in axes}
    return spec


def make_random_spec(band, n, seed, *, region=DEFAULT_REGION, fixed=DEFAULT_FIXED,
                     box=None, p10_range=(0.3, 0.6), q10_range=(0.2, 0.4),
                     epsrel=1e-7, epsabs=1e-13):
    """n held-out points, uniform over the axis box except F0, which
    alternates between log-uniform (even i; how a log-scale prior samples
    it) and uniform (odd i; the high-F0 end, where the dependence is
    strongest), restricted to the prior's p10 and q10 ranges (None = whole box).
    Point i depends only on (seed, i), so any subset can be computed
    independently."""
    spec = _base_spec("random", band, region, fixed, default_box(band) if box is None else box,
                      epsrel, epsabs)
    spec.update(n=int(n), seed=int(seed),
                p10_range=None if p10_range is None else list(p10_range),
                q10_range=None if q10_range is None else list(q10_range))
    return spec


def load_spec(path):
    with open(path) as f:
        return json.load(f)


def save_spec(spec, path):
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)


def axes_of(spec):
    return BAND_AXES[spec["band"]]


def chebyshev_lobatto(lo, hi, n):
    """n Chebyshev-Lobatto nodes on [lo, hi], ascending, endpoints included
    (n == 1: the midpoint)."""
    if n == 1:
        return np.array([0.5 * (lo + hi)])
    j = np.arange(n)
    return 0.5 * (lo + hi) - 0.5 * (hi - lo) * np.cos(np.pi * j / (n - 1))


def nodes_of(spec):
    return {a: chebyshev_lobatto(*spec["box"][a], spec["n_nodes"][a]) for a in axes_of(spec)}


def grid_shape(spec):
    return tuple(spec["n_nodes"][a] for a in axes_of(spec))


def n_points(spec):
    if spec["kind"] == "grid":
        return int(np.prod(grid_shape(spec)))
    if spec["kind"] == "points":                # an explicit list (normplan's study)
        return len(spec["points"])
    return spec["n"]


def coords_at(spec, i):
    """Axis coordinates of point i (flat C-order index for a grid, so the
    last axis varies fastest)."""
    axes = axes_of(spec)
    if not 0 <= i < n_points(spec):
        raise IndexError(f"point index {i} outside [0, {n_points(spec)})")
    if spec["kind"] == "grid":
        nodes = nodes_of(spec)
        idx = np.unravel_index(i, grid_shape(spec))
        return {a: float(nodes[a][j]) for a, j in zip(axes, idx)}
    if spec["kind"] == "points":
        return dict(spec["points"][i])
    rng = np.random.default_rng([spec["seed"], i])
    for _ in range(10000):
        c = {a: float(spec["box"][a][0] + rng.random() * (spec["box"][a][1] - spec["box"][a][0]))
             for a in axes}
        if i % 2 == 0:
            lo, hi = spec["box"]["F0"]
            c["F0"] = float(math.exp(math.log(lo) + rng.random() * (math.log(hi) - math.log(lo))))
        p_range, q_range = spec["p10_range"], spec.get("q10_range")   # q10_range absent in old specs
        if ((p_range is None or p_range[0] <= c["p0"] + c["dp"] <= p_range[1])
                and (q_range is None or q_range[0] <= c["q0"] + c["dq"] <= q_range[1])):
            return c
    raise RuntimeError("could not draw a point inside the p10/q10 ranges")


def physical_params(spec, coords):
    """Axis coordinates -> the keyword arguments of ppqn_region/ppqg_region
    (minus the region and tolerances)."""
    p = {"F0": coords["F0"], "eps": spec["fixed"]["eps"], "V": coords["V"],
         "p0": coords["p0"], "p10": coords["p0"] + coords["dp"],
         "q0": coords["q0"], "q10": coords["q0"] + coords["dq"]}
    if spec["band"] == "NR":
        p["k"] = coords["k"]
        p["Z"] = spec["fixed"]["Z"]
    return p


def _evaluate(spec, params, epsrel):
    """One region integral.  NORMGRID_FAKE=1 swaps in a cheap analytic stand-in
    (for the tests); NORMGRID_FAKE_CRASH=<idx> is handled by the worker."""
    fake = os.environ.get("NORMGRID_FAKE")
    if fake in ("smooth", "kink"):
        # analytic stand-ins with KNOWN convergence, in box-normalised coordinates u in [-1, 1]:
        # smooth = exp(3 u_V + 6 u_q0) (Chebyshev coefficients ~ Bessel I_j, needs ~10-20 nodes
        # on those two axes, the minimum elsewhere); kink = |u_q0 - 0.13| (algebraic decay, never converges)
        c = {**params, "dp": params["p10"] - params["p0"], "dq": params["q10"] - params["q0"]}
        u = {a: 2 * (c[a] - lo) / (hi - lo) - 1 for a, (lo, hi) in spec["box"].items()}
        if fake == "kink":
            return 1.0 + abs(u["q0"] - 0.13)
        return math.exp(3 * u["V"] + 6 * u["q0"]) * (1 + 0.05 * sum(v for a, v in u.items() if a not in ("V", "q0")))
    if fake:
        return 1.0 + sum(params[k] * (i + 1) for i, k in enumerate(sorted(params))) * 1e-3
    sys.path.insert(0, _INTERNAL_DIR)
    from _ppqfort_bindings import ppqg_region, ppqn_region
    fn = ppqn_region if spec["band"] == "NR" else ppqg_region
    return fn(*spec["region"], epsrel=epsrel, epsabs=spec["epsabs"], **params)


# ---------------------------------------------------------------------------
# Worker: results files are append-only lines "index value epsrel" (value
# nan = failed); the last line for an index wins.
# ---------------------------------------------------------------------------

def read_results(path):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and not line.startswith("#"):
                    out[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return out


def _append(path, i, value, epsrel, note=""):
    with open(path, "a") as f:
        f.write(f"{i} {value!r} {epsrel!r}{'  # ' + note if note else ''}\n")


def _worker(spec_path, out, todo_path, progress_path, epsrel):
    spec = load_spec(spec_path)
    todo = [int(x) for x in open(todo_path).read().split()]
    crash_at = os.environ.get("NORMGRID_FAKE_CRASH")
    for i in todo:
        with open(progress_path, "w") as f:
            f.write(str(i))
        if crash_at is not None and int(crash_at) == i:
            sys.exit(3)      # stand-in for a Fortran `error stop`
        v = _evaluate(spec, physical_params(spec, coords_at(spec, i)), epsrel)
        _append(out, i, float(v), epsrel)


def run_range(spec_path, start, stop, out, retry_failed=False, epsrel=None, quiet=False):
    """Evaluate points [start, stop) into `out`, skipping any already done.
    A point whose evaluation kills the process (Fortran `error stop`, e.g.
    the quadrature not certifying epsrel) is recorded as nan and the worker
    restarted after it; --retry-failed re-attempts recorded failures (use
    with a looser --epsrel)."""
    spec = load_spec(spec_path)
    stop = min(stop, n_points(spec))
    epsrel = spec["epsrel"] if epsrel is None else epsrel
    progress, todo_path = out + ".progress", out + ".todo"
    attempted = set()
    t0 = time.time()
    while True:
        done = read_results(out)
        todo = [i for i in range(start, stop) if i not in attempted and
                (i not in done or (retry_failed and math.isnan(done[i][0])))]
        if not todo:
            break
        with open(todo_path, "w") as f:
            f.write(" ".join(map(str, todo)))
        cmd = [sys.executable, os.path.abspath(__file__), "_worker", "--spec", spec_path, "--out", out,
               "--todo", todo_path, "--progress", progress, "--epsrel", repr(epsrel)]
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            break
        bad = int(open(progress).read())
        attempted.add(bad)
        _append(out, bad, float("nan"), epsrel, f"worker exited {rc}")
        if not quiet:
            print(f"point {bad} failed (worker exit {rc}); recorded as nan, continuing", flush=True)
    for p in (progress, todo_path):
        if os.path.exists(p):
            os.remove(p)
    done = read_results(out)
    n_ok = sum(1 for i in range(start, stop) if i in done and not math.isnan(done[i][0]))
    if not quiet:
        print(f"[{start}, {stop}): {n_ok}/{stop - start} ok in {time.time() - t0:.0f} s -> {out}")
    return n_ok == stop - start


# ---------------------------------------------------------------------------
# Merge to HDF5
# ---------------------------------------------------------------------------

def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError("normgrid tables are HDF5: install h5py (pip install h5py; it is in "
                          "environment.yaml)") from e
    return h5py


def gather_results(spec, paths):
    vals = np.full(n_points(spec), np.nan)
    eps = np.full(n_points(spec), np.nan)
    for p in paths:
        for i, (v, e) in read_results(p).items():
            if i < len(vals) and (math.isnan(vals[i]) or not math.isnan(v)):
                vals[i], eps[i] = v, e
    return vals, eps


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return ""


def merge(spec_path, result_paths, out_path, allow_missing=False):
    spec = load_spec(spec_path)
    if spec["kind"] != "grid":
        raise ValueError("merge builds a table from a grid spec")
    vals, eps = gather_results(spec, result_paths)
    missing = np.flatnonzero(np.isnan(vals))
    if len(missing) and not allow_missing:
        raise RuntimeError(f"{len(missing)} of {len(vals)} grid points missing or failed (first: "
                           f"{missing[:10].tolist()}); re-run them (run --retry-failed --epsrel ...) "
                           f"or pass --allow-missing")
    h5py = _h5py()
    shape = grid_shape(spec)
    lib_version = ""
    try:
        sys.path.insert(0, _INTERNAL_DIR)
        import _ppqfort_bindings
        lib_version = ".".join(map(str, _ppqfort_bindings.version()))
    except Exception:
        pass
    with h5py.File(out_path, "w") as f:
        f.create_dataset("values", data=vals.reshape(shape), compression="gzip")
        f.create_dataset("epsrel_used", data=eps.reshape(shape), compression="gzip")
        g = f.create_group("nodes")
        for a, n in nodes_of(spec).items():
            g.create_dataset(a, data=n)
        f.attrs["spec_json"] = json.dumps(spec)
        f.attrs["axes"] = json.dumps(list(axes_of(spec)))
        f.attrs["band"] = spec["band"]
        f.attrs["region"] = np.array(spec["region"])
        f.attrs["fixed_json"] = json.dumps(spec["fixed"])
        f.attrs["library_version"] = lib_version
        f.attrs["git_commit"] = _git_commit()
        f.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs["n_missing"] = len(missing)
    print(f"wrote {out_path}: shape {shape}, {len(missing)} missing")


# ---------------------------------------------------------------------------
# Interpolator
# ---------------------------------------------------------------------------

class NormInterpolator:
    """Tensor-product Chebyshev interpolant of a precomputed normalization
    table, built and evaluated with numpy.polynomial.chebyshev (no scipy).
    The table's values sit at Chebyshev-Lobatto nodes, so fitting a degree
    n-1 Chebyshev series along an n-node axis interpolates exactly through
    them.  Call with physical parameters:

        norm = table(k=..., F0=..., V=..., p0=..., p10=..., q0=..., q10=...)   # NR
        norm = table(F0=..., V=..., p0=..., p10=..., q0=..., q10=...)          # ER

    Z/eps may be passed too, but must match what the table was built with.
    Raises OutOfBoxError rather than ever extrapolating."""

    def __init__(self, axes, nodes, values, *, band, region, fixed, source=""):
        self.axes = tuple(axes)
        self.nodes = [np.asarray(nodes[a], dtype=float) for a in self.axes]
        values = np.asarray(values, dtype=float)
        if values.shape != tuple(len(n) for n in self.nodes):
            raise ValueError("values shape does not match node counts")
        if not np.all(np.isfinite(values)):
            raise ValueError("table contains missing/failed points; refusing to build an interpolant")
        self.band, self.region, self.fixed, self.source = band, tuple(region), dict(fixed), source
        self.box = {a: (n[0], n[-1]) for a, n in zip(self.axes, self.nodes)}
        # values at nodes -> Chebyshev coefficients, one axis at a time
        coef = values
        for d, x in enumerate(self.nodes):
            c = np.moveaxis(coef, d, 0)
            c = cheb.chebfit(self._to_unit(d, x), c.reshape(len(x), -1), len(x) - 1).reshape(c.shape)
            coef = np.moveaxis(c, 0, d)
        self._coef = coef

    def _to_unit(self, d, x):
        """Axis d coordinate -> [-1, 1] (a single node is a constant axis)."""
        lo, hi = self.nodes[d][0], self.nodes[d][-1]
        return 2 * (x - lo) / (hi - lo) - 1 if hi > lo else np.zeros_like(x)

    @classmethod
    def from_hdf5(cls, path):
        h5py = _h5py()
        with h5py.File(path, "r") as f:
            axes = json.loads(f.attrs["axes"])
            nodes = {a: f["nodes"][a][:] for a in axes}
            return cls(axes, nodes, f["values"][:], band=str(f.attrs["band"]),
                       region=f.attrs["region"], fixed=json.loads(f.attrs["fixed_json"]), source=path)

    def _coords(self, k, F0, V, p0, p10, q0, q10, Z, eps):
        for name, given in (("Z", Z), ("eps", eps)):
            if given is not None and name in self.fixed and abs(given - self.fixed[name]) > 1e-12 * max(1.0, abs(given)):
                raise ValueError(f"{name}={given} differs from the {self.fixed[name]} this table was built for")
        c = {"F0": F0, "V": V, "p0": p0, "dp": p10 - p0, "q0": q0, "dq": q10 - q0}
        if self.band == "NR":
            if k is None:
                raise ValueError("NR table needs k")
            c["k"] = k
        elif k is not None:
            raise ValueError("ER table does not depend on k; do not pass it")
        return c

    def __call__(self, *, F0, V, p0, p10, q0, q10, k=None, Z=None, eps=None):
        c = self._coords(k, F0, V, p0, p10, q0, q10, Z, eps)
        coef = self._coef
        for d, a in enumerate(self.axes):
            x = c[a]
            lo, hi = self.box[a]
            tol = 1e-12 * (hi - lo)
            if not (lo - tol <= x <= hi + tol):          # also catches NaN
                query = {n: v for n, v in (("k", k), ("F0", F0), ("V", V), ("p0", p0), ("p10", p10),
                                           ("q0", q0), ("q10", q10)) if v is not None}
                raise OutOfBoxError(
                    f"{a}={x!r} outside the range [{lo:.6g}, {hi:.6g}] of the {self.band} table "
                    f"{self.source or '(in memory)'}\n  query: "
                    + " ".join(f"{n}={v:.6g}" for n, v in query.items())
                    + "\n  The table covers the prior it was built for, so a query outside it means the "
                      "MCMC prior does not match the table, or a NaN/bug.  Check the prior at setup "
                      "(PpqPDF(prior_bounds=...) or table.check_prior), and have log_prob return -inf "
                      "for points outside the prior BEFORE calling the likelihood.",
                    axis=a, value=x, box=(lo, hi), query=query, source=self.source)
            coef = cheb.chebval(self._to_unit(d, np.float64(min(max(x, lo), hi))), coef)   # contracts the first remaining axis
        return float(coef)

    # ---- MCMC setup checks: is the prior (are the starting points) inside the box? ----

    def params(self):
        """The physical parameters this table varies, in PpqPDF's keyword names."""
        return (("k",) if self.band == "NR" else ()) + ("F0", "V", "p0", "p10", "q0", "q10")

    def table_bounds(self):
        """A physical-parameter prior box this table is guaranteed to cover,
        {param: (lo, hi)}, for a prior that also enforces p10 >= p0 and
        q10 >= q0 (the PDF is undefined otherwise).  Keeps the full p0/q0
        ranges and gives p10/q10 the widest ranges compatible with them, so
        it can be used directly as an MCMC's hard prior bounds.  Z and eps
        are fixed by the table (self.fixed), not ranges."""
        b = {a: self.box[a] for a in ("k", "F0", "V", "p0", "q0") if a in self.box}
        b["p10"], b["q10"] = x10_ranges(self.box)
        return {p: tuple(float(v) for v in b[p]) for p in self.params()}

    def required_box(self, bounds):
        """The axis box a prior with these physical bounds reaches, given that
        it enforces p10 >= p0 and q10 >= q0: {axis: (lo, hi)}.  bounds maps
        each of self.params() to (lo, hi); Z/eps may be given as the fixed
        value.  Raises PriorNotCoveredError for a missing or unbounded
        parameter, a range where Z/eps are fixed, or an empty prior."""
        problems = []
        for name, v in self.fixed.items():
            if name in bounds:
                lo, hi = (bounds[name], bounds[name]) if np.isscalar(bounds[name]) else bounds[name]
                if not (abs(lo - v) <= 1e-12 * max(1.0, abs(v)) and abs(hi - v) <= 1e-12 * max(1.0, abs(v))):
                    problems.append(f"{name}: the table fixes {name}={v}; the prior gives {bounds[name]}")
        extra = sorted(set(bounds) - set(self.params()) - set(self.fixed) - {"Z"})
        if extra:
            problems.append(f"the {self.band} table does not depend on {', '.join(extra)}"
                            + (" (the ER band has no k)" if "k" in extra else ""))
        b = {}
        for p in self.params():
            if p not in bounds or bounds[p] is None:
                problems.append(f"{p}: no prior bounds given (the table varies it)")
                continue
            lo, hi = (float(v) if v is not None else np.nan for v in bounds[p])
            if not (np.isfinite(lo) and np.isfinite(hi)):
                problems.append(f"{p}: prior [{lo}, {hi}] is unbounded; no table can cover it -- truncate "
                                "the prior (e.g. a Gaussian at +/-4 sigma) and reject outside it in log_prob")
            elif lo > hi:
                problems.append(f"{p}: prior lower bound {lo} is above the upper bound {hi}")
            else:
                b[p] = (lo, hi)
        if problems:
            raise PriorNotCoveredError(f"prior bounds unusable with the {self.band} table "
                                       f"{self.source or '(in memory)'}:\n  " + "\n  ".join(problems))
        req = {a: b[a] for a in ("k", "F0", "V") if a in b}
        for x, d in (("p", "dp"), ("q", "dq")):
            (x0lo, x0hi), (x1lo, x1hi) = b[x + "0"], b[x + "10"]
            x0hi_eff = min(x0hi, x1hi)            # x10 >= x0 caps x0 at the largest x10
            if x1hi < x0lo:
                raise PriorNotCoveredError(f"{x}10 <= {x1hi} < {x}0 >= {x0lo}: the prior is empty "
                                           f"under {x}10 >= {x}0")
            req[x + "0"] = (x0lo, x0hi_eff)
            req[d] = (max(0.0, x1lo - x0hi_eff), x1hi - x0lo)
        return {a: req[a] for a in self.axes}

    def check_prior(self, bounds, *, rebuild_hint=True):
        """Raise PriorNotCoveredError unless an MCMC prior with these physical
        bounds (see required_box) stays inside the table's box.  Call it once
        at MCMC setup; the message lists every axis that falls short and the
        box to rebuild the table with."""
        req = self.required_box(bounds)
        short = {a: r for a, r in req.items() if not self._covers(a, r)}
        if short:
            lines = [f"{a}: prior reaches [{r[0]:.6g}, {r[1]:.6g}], table covers "
                     f"[{self.box[a][0]:.6g}, {self.box[a][1]:.6g}]" for a, r in short.items()]
            msg = (f"the MCMC prior is not covered by the {self.band} table {self.source or '(in memory)'}"
                   " (dp = p10 - p0, dq = q10 - q0):\n  " + "\n  ".join(lines))
            if rebuild_hint:
                msg += ("\nRestrict the prior to the table (table_bounds() gives a covered prior box), or "
                        "rebuild it with:\n  " + self.rebuild_command(short))
            raise PriorNotCoveredError(msg)

    def check_points(self, points):
        """Raise PriorNotCoveredError unless every point is inside the box,
        e.g. an MCMC's starting walker positions.  points: {param: array}
        with every one of self.params() (Z/eps optional, checked against the
        table's fixed values)."""
        missing = [p for p in self.params() if p not in points]
        if missing:
            raise PriorNotCoveredError(f"points lack {', '.join(missing)}")
        v = {p: np.atleast_1d(np.asarray(points[p], dtype=float)) for p in self.params()}
        n = len(next(iter(v.values())))
        if any(len(a) != n for a in v.values()):
            raise PriorNotCoveredError("points: every parameter needs the same number of values")
        for name, fixed in self.fixed.items():
            if name in points and np.any(np.abs(np.asarray(points[name], dtype=float) - fixed)
                                         > 1e-12 * max(1.0, abs(fixed))):
                raise PriorNotCoveredError(f"points: {name} differs from the {fixed} this table was built for")
        c = {"F0": v["F0"], "V": v["V"], "p0": v["p0"], "dp": v["p10"] - v["p0"],
             "q0": v["q0"], "dq": v["q10"] - v["q0"]}
        if "k" in v:
            c["k"] = v["k"]
        bad, lines = np.zeros(n, dtype=bool), []
        for a in self.axes:
            lo, hi = self.box[a]
            tol = 1e-12 * (hi - lo)
            out = ~((c[a] >= lo - tol) & (c[a] <= hi + tol))      # NaN counts as out
            if out.any():
                bad |= out
                lines.append(f"{a}: {out.sum()} of {n} points outside [{lo:.6g}, {hi:.6g}] "
                             f"(they reach [{np.nanmin(c[a]):.6g}, {np.nanmax(c[a]):.6g}])")
        if bad.any():
            raise PriorNotCoveredError(
                f"{bad.sum()} of {n} points are outside the {self.band} table {self.source or '(in memory)'} "
                "(dp = p10 - p0, dq = q10 - q0):\n  " + "\n  ".join(lines)
                + f"\n  first: point {int(np.flatnonzero(bad)[0])}\n"
                  "Start the walkers inside the prior, and the prior inside the table (check_prior).")

    def _covers(self, a, r):
        lo, hi = self.box[a]
        tol = 1e-12 * (hi - lo)
        return lo - tol <= r[0] and r[1] <= hi + tol

    def rebuild_command(self, need):
        """The build_table command for a table whose box also covers `need`
        ({axis: (lo, hi)}): BOX lists every axis that then differs from the
        band's default box, rounded outward."""
        box = {a: (min(self.box[a][0], need[a][0]), max(self.box[a][1], need[a][1])) if a in need
               else self.box[a] for a in self.axes}
        default = default_box(self.band)
        spec = " ".join(f"{a}={format_box_range(*box[a])}" for a in self.axes
                        if not np.allclose(box[a], default[a], rtol=1e-9, atol=0))
        region = " ".join(f"{r:g}" for r in self.region)
        return f'BOX="{spec}" sbatch slurm/build_table.job {self.band} {region}'


def _round_out(x, up, sig=4):
    """x rounded to `sig` significant figures, up or down (never inward)."""
    if x == 0 or not np.isfinite(x):
        return float(x)
    scale = 10.0 ** (math.floor(math.log10(abs(x))) - sig + 1)
    return float((math.ceil if up else math.floor)(x / scale - (1e-9 if up else -1e-9)) * scale)


def format_box_range(lo, hi):
    """'lo:hi' with lo rounded down and hi up, as BOX= and --box take it."""
    return f"{_round_out(lo, False):.6g}:{_round_out(hi, True):.6g}"


def parse_box(items, band):
    """['q0=0.04:0.45', 'dq=0:0.42', ...] (or one space-separated string) ->
    the band's default box with those axes replaced."""
    if isinstance(items, str):
        items = items.split()
    box = dict(default_box(band))
    for item in items:
        try:
            axis, rng = item.split("=")
            lo, hi = (float(v) for v in rng.split(":"))
        except ValueError:
            raise ValueError(f"box entry {item!r} is not AXIS=LO:HI") from None
        if axis not in BAND_AXES[band]:
            raise ValueError(f"{axis!r} is not an axis of the {band} table ({', '.join(BAND_AXES[band])})")
        if not lo < hi:
            raise ValueError(f"box entry {item!r}: need LO < HI")
        box[axis] = (lo, hi)
    return box


# ---------------------------------------------------------------------------
# Validation against held-out, directly computed points
# ---------------------------------------------------------------------------

def validate(table_path, heldout_spec_path, result_paths, n_worst=5, accept=None):
    """Print the interpolation error against held-out points; returns the
    array of relative errors.  With `accept`, also prints PASS/FAIL against
    that worst-case relative error (the CLI turns FAIL into exit status 1)."""
    table = NormInterpolator.from_hdf5(table_path)
    spec = load_spec(heldout_spec_path)
    vals, _ = gather_results(spec, result_paths)
    errs, rows = [], []
    for i in np.flatnonzero(~np.isnan(vals)):
        c = coords_at(spec, int(i))
        pred = table(**{k: v for k, v in physical_params(spec, c).items() if k not in ("Z", "eps")})
        errs.append(abs(pred - vals[i]) / abs(vals[i]))
        rows.append((errs[-1], int(i), c))
    errs = np.array(errs)
    print(f"{len(errs)} held-out points ({np.isnan(vals).sum()} missing): relative interpolation error")
    print(f"  max {errs.max():.2e}   99% {np.percentile(errs, 99):.2e}   rms {np.sqrt(np.mean(errs**2)):.2e}"
          f"   median {np.median(errs):.2e}")
    print(f"  => log-likelihood error at 20,000 events: max {2e4 * errs.max():.3f}, rms {2e4 * np.sqrt(np.mean(errs**2)):.3f}")
    for e, i, c in sorted(rows, reverse=True)[:n_worst]:
        print(f"  worst: point {i} err {e:.2e}  " + " ".join(f"{a}={v:.4g}" for a, v in c.items()))
    if accept is not None:
        print(f"  {'PASS' if errs.max() <= accept else 'FAIL'}: worst-case error {errs.max():.2e} vs accepted {accept:.1e}")
    return errs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_kv_ints(items):
    return {kv.split("=")[0]: int(kv.split("=")[1]) for kv in items}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-spec", help="write a grid spec")
    p.add_argument("--band", required=True, choices=["NR", "ER"])
    p.add_argument("--nodes", nargs="+", default=None, metavar="AXIS=N",
                   help="nodes per axis, e.g. k=7 F0=3 V=4 p0=2 dp=3 q0=3 dq=3 (ER: no k); default: RECOMMENDED_NODES")
    # --region and --epsrel are required, not defaulted, for the same reason
    # ppqn_region takes no default epsrel/epsabs: a wrong value here produces a
    # table that is complete, internally consistent and wrong.  Neither merge
    # nor validate can see it -- they only ever compare a table against points
    # computed from the same spec -- so it would surface at the earliest in
    # PpqPDF's region check, and only if the fit gets that far.  (--nodes and
    # the box do default: validate measures node adequacy directly, and a bad
    # box raises OutOfBoxError at query time.  Both fail loudly.)
    p.add_argument("--region", nargs=4, type=float, required=True, metavar=("EP_MIN", "EP_MAX", "EQ_MIN", "EQ_MAX"),
                   help="fit region the table normalizes over; must match the region PpqPDF is built with")
    p.add_argument("--epsrel", type=float, required=True,
                   help="quadrature convergence tolerance per point, e.g. 1e-7")
    p.add_argument("--box", nargs="+", default=None, metavar="AXIS=LO:HI",
                   help="replace axes of the band's default box, e.g. V=2.5:4 dq=0:0.42")
    p.add_argument("--out", required=True)

    p = sub.add_parser("make-random-spec", help="write a held-out validation point set")
    p.add_argument("--band", required=True, choices=["NR", "ER"])
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--region", nargs=4, type=float, required=True, metavar=("EP_MIN", "EP_MAX", "EQ_MIN", "EQ_MAX"),
                   help="must match the region of the table these points validate")
    p.add_argument("--epsrel", type=float, required=True)
    p.add_argument("--box", nargs="+", default=None, metavar="AXIS=LO:HI",
                   help="as make-spec's; the points are then drawn over the p10/q10 ranges that box "
                        "covers (x10_ranges) instead of the default prior's")
    p.add_argument("--out", required=True)

    p = sub.add_parser("info", help="print a spec's size")
    p.add_argument("spec")

    p = sub.add_parser("chunks", help="print 'start stop' ranges of --size points")
    p.add_argument("spec"); p.add_argument("--size", type=int, default=200)

    p = sub.add_parser("run", help="evaluate points [start, stop) (worker)")
    p.add_argument("--spec", required=True); p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=10**12); p.add_argument("--out", required=True)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--epsrel", type=float, default=None, help="override the spec's epsrel")

    p = sub.add_parser("_worker"); p.add_argument("--spec"); p.add_argument("--out")
    p.add_argument("--todo"); p.add_argument("--progress"); p.add_argument("--epsrel", type=float)

    p = sub.add_parser("merge", help="combine result files into an HDF5 table")
    p.add_argument("--spec", required=True); p.add_argument("--results", nargs="+", required=True,
                   help="result files or globs"); p.add_argument("--out", required=True)
    p.add_argument("--allow-missing", action="store_true")

    p = sub.add_parser("validate", help="interpolation error against held-out points")
    p.add_argument("--table", required=True); p.add_argument("--heldout-spec", required=True)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--accept", type=float, default=None, metavar="REL_ERR",
                   help="exit status 1 if the worst held-out relative error exceeds this")

    a = ap.parse_args(argv)
    expand = lambda pats: sorted({f for pat in pats for f in (glob.glob(pat) or [pat])})

    if a.cmd == "make-spec":
        nodes = _parse_kv_ints(a.nodes) if a.nodes else RECOMMENDED_NODES[a.band]
        spec = make_grid_spec(a.band, nodes, region=tuple(a.region), epsrel=a.epsrel,
                              box=parse_box(a.box, a.band) if a.box else None)
        save_spec(spec, a.out)
        print(f"{a.out}: {n_points(spec)} points, shape {grid_shape(spec)}")
    elif a.cmd == "make-random-spec":
        if a.box:
            box = parse_box(a.box, a.band)
            p10, q10 = x10_ranges(box)
            spec = make_random_spec(a.band, a.n, a.seed, region=tuple(a.region), epsrel=a.epsrel,
                                    box=box, p10_range=p10, q10_range=q10)
        else:
            spec = make_random_spec(a.band, a.n, a.seed, region=tuple(a.region), epsrel=a.epsrel)
        save_spec(spec, a.out)
        print(f"{a.out}: {a.n} held-out points")
    elif a.cmd == "info":
        s = load_spec(a.spec)
        print(json.dumps({"kind": s["kind"], "band": s["band"], "points": n_points(s),
                          "shape": grid_shape(s) if s["kind"] == "grid" else None}))
    elif a.cmd == "chunks":
        n = n_points(load_spec(a.spec))
        for s in range(0, n, a.size):
            print(s, min(s + a.size, n))
    elif a.cmd == "run":
        ok = run_range(a.spec, a.start, a.stop, a.out, a.retry_failed, a.epsrel)
        sys.exit(0 if ok else 2)
    elif a.cmd == "_worker":
        _worker(a.spec, a.out, a.todo, a.progress, a.epsrel)
    elif a.cmd == "merge":
        merge(a.spec, expand(a.results), a.out, a.allow_missing)
    elif a.cmd == "validate":
        errs = validate(a.table, a.heldout_spec, expand(a.results), accept=a.accept)
        if a.accept is not None and errs.max() > a.accept:
            sys.exit(1)


if __name__ == "__main__":
    main()
