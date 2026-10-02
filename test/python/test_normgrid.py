"""
Tests for python/cli/normgrid.py that need no Fortran: node generation, index
<-> coordinate mapping, the physical-parameter mapping (p10 >= p0), the
tensor-Lagrange interpolant (exact for polynomials it can represent, exact at
nodes, never extrapolates), the crash-tolerant resumable worker (using the
cheap NORMGRID_FAKE evaluator), and the HDF5 round trip.  The real
Fortran-backed pipeline is exercised by hand; see the normgrid README section.

Run from the repository root:  python test/python/test_normgrid.py
"""

import itertools
import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "cli"))
import normgrid as ng

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def poly(c, degs):
    """A polynomial with a cross term in every axis, of degree degs[a]."""
    x = [c[a] for a in ng.BAND_AXES["NR"]]
    return (1.0 + 0.5 * x[0] ** degs[0] * x[1] ** degs[1] - 2.0 * x[2] ** degs[2]
            + 0.3 * x[3] ** degs[3] * x[4] ** degs[4] + x[5] ** degs[5] * x[6] ** degs[6])


# ---- nodes ----
n = ng.chebyshev_lobatto(2.0, 6.0, 5)
check("lobatto nodes: endpoints included, ascending", n[0] == 2.0 and n[-1] == 6.0 and np.all(np.diff(n) > 0))
check("lobatto n=1 is the midpoint", ng.chebyshev_lobatto(2.0, 6.0, 1)[0] == 4.0)
check("lobatto n=3 is [lo, mid, hi]", np.allclose(ng.chebyshev_lobatto(0, 1, 3), [0, 0.5, 1]))

# ---- specs / mapping ----
spec = ng.make_grid_spec("NR", dict(k=3, F0=4, V=2, p0=3, dp=2, q0=2, dq=3))
check("grid size", ng.n_points(spec) == 3 * 4 * 2 * 3 * 2 * 2 * 3)
seen = {tuple(sorted(ng.coords_at(spec, i).items())) for i in range(ng.n_points(spec))}
check("every grid index maps to a distinct point", len(seen) == ng.n_points(spec))
c0, c1 = ng.coords_at(spec, 0), ng.coords_at(spec, 1)
check("last axis varies fastest", c0["dq"] != c1["dq"] and all(c0[a] == c1[a] for a in ng.AXES[:-1]))
check("out-of-range index raises", raises(IndexError, lambda: ng.coords_at(spec, ng.n_points(spec))))
check("bad n_nodes rejected", raises(ValueError, lambda: ng.make_grid_spec("NR", dict(k=3))))

ok_p10 = all(ng.physical_params(spec, ng.coords_at(spec, i))["p10"] >= ng.physical_params(spec, ng.coords_at(spec, i))["p0"]
             for i in range(ng.n_points(spec)))
check("grid: p10 >= p0 at every point (rectangular box, no unphysical corners)", ok_p10)
ok_q10 = all(ng.physical_params(spec, ng.coords_at(spec, i))["q10"] >= ng.physical_params(spec, ng.coords_at(spec, i))["q0"]
             for i in range(ng.n_points(spec)))
check("grid: q10 >= q0 at every point", ok_q10)
rs = ng.make_random_spec("NR", 500, seed=7)
ps = [ng.physical_params(rs, ng.coords_at(rs, i)) for i in range(500)]
check("random: p10 within the prior range and >= p0", all(0.3 <= p["p10"] <= 0.6 and p["p10"] >= p["p0"] for p in ps))
check("random: q10 within the prior range and >= q0", all(0.2 <= p["q10"] <= 0.4 and p["q10"] >= p["q0"] for p in ps))
check("random points are reproducible per index", ng.coords_at(rs, 123) == ng.coords_at(rs, 123)
      and ng.coords_at(rs, 123) != ng.coords_at(rs, 124))
er = ng.make_grid_spec("ER", dict(F0=2, V=2, p0=2, dp=2, q0=2, dq=2))
check("ER spec has no k axis and no Z", "k" not in ng.axes_of(er) and "k" not in ng.physical_params(er, ng.coords_at(er, 0))
      and "Z" not in ng.physical_params(er, ng.coords_at(er, 0)))

# ---- interpolant ----
degs = (2, 3, 1, 2, 1, 1, 2)                       # nodes per axis = degree + 1
nodes = {a: ng.chebyshev_lobatto(*ng.DEFAULT_BOX[a], d + 1) for a, d in zip(ng.AXES, degs)}
grid = np.array([poly(dict(zip(ng.AXES, pt)), degs)
                 for pt in itertools.product(*[nodes[a] for a in ng.AXES])]).reshape([d + 1 for d in degs])
table = ng.NormInterpolator(ng.AXES, nodes, grid, band="NR", region=ng.DEFAULT_REGION, fixed=ng.DEFAULT_FIXED)
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(50):
    c = {a: rng.uniform(*ng.DEFAULT_BOX[a]) for a in ng.AXES}
    val = table(k=c["k"], F0=c["F0"], V=c["V"], p0=c["p0"], p10=c["p0"] + c["dp"], q0=c["q0"], q10=c["q0"] + c["dq"])
    worst = max(worst, abs(val - poly(c, degs)) / abs(poly(c, degs)))
check("interpolant reproduces a representable polynomial exactly", worst < 1e-12, f"worst rel err {worst:.1e}")
node_pt = {a: nodes[a][len(nodes[a]) // 2] for a in ng.AXES}
v = table(k=node_pt["k"], F0=node_pt["F0"], V=node_pt["V"], p0=node_pt["p0"],
          p10=node_pt["p0"] + node_pt["dp"], q0=node_pt["q0"], q10=node_pt["q0"] + node_pt["dq"])
check("exact at a node", abs(v - poly(node_pt, degs)) < 1e-13 * abs(v))
mid = dict(k=0.175, F0=1e-2, V=3.0, p0=0.064, p10=0.5, q0=0.2, q10=0.3)
check("out of box raises (k)", raises(ng.OutOfBoxError, lambda: table(**dict(mid, k=0.5))))
check("out of box raises (V)", raises(ng.OutOfBoxError, lambda: table(**dict(mid, V=2.0))))
check("out of box raises (F0 too large)", raises(ng.OutOfBoxError, lambda: table(**dict(mid, F0=10.0))))
check("out of box raises (F0 <= 0)", raises(ng.OutOfBoxError, lambda: table(**dict(mid, F0=0.0))))
check("p10 < p0 raises (dp < 0)", raises(ng.OutOfBoxError, lambda: table(**dict(mid, p10=0.05))))
check("q10 < q0 raises (dq < 0)", raises(ng.OutOfBoxError, lambda: table(**dict(mid, q10=0.15))))
check("NaN raises", raises(ng.OutOfBoxError, lambda: table(**dict(mid, q0=float("nan")))))
check("NR table requires k", raises(ValueError, lambda: table(**{k: v for k, v in mid.items() if k != "k"})))
check("mismatched Z rejected", raises(ValueError, lambda: table(**mid, Z=40.0)))
check("matching Z/eps accepted", np.isfinite(table(**mid, Z=32.0, eps=3.0e-3)))

# ---- PpqPDF uses a table when given one ----
from ppq_pdf import PpqPDF
reg = ng.DEFAULT_REGION
ep_d, eq_d = np.array([50.0, 60.0]), np.array([20.0, 25.0])
fit = PpqPDF(*reg, ep_d, eq_d, ppqn_table=table)
mid_full = dict(mid, Z=32.0, eps=3.0e-3)
check("PpqPDF.ppqn_integral interpolates the table", fit.ppqn_integral(**mid_full) == table(**mid, Z=32.0, eps=3.0e-3))
check("PpqPDF without table or tolerances refuses to guess",
      raises(ValueError, lambda: PpqPDF(*reg, ep_d, eq_d).ppqn_integral(**mid_full)))
check("PpqPDF rejects a table built for a different region",
      raises(ValueError, lambda: PpqPDF(2.0, 100.0, 4.0, 100.0, ep_d, eq_d, ppqn_table=table)))
check("PpqPDF rejects a table for the wrong band", raises(ValueError, lambda: PpqPDF(*reg, ep_d, eq_d, ppqg_table=table)))

# ---- worker: crash tolerance, resume, retry (fake evaluator) ----
os.environ["NORMGRID_FAKE"] = "1"
with tempfile.TemporaryDirectory() as d:
    er2 = ng.make_grid_spec("ER", dict(F0=2, V=2, p0=2, dp=2, q0=2, dq=2))     # 64 points
    sp, out = os.path.join(d, "spec.json"), os.path.join(d, "r.txt")
    ng.save_spec(er2, sp)
    os.environ["NORMGRID_FAKE_CRASH"] = "5"
    ok = ng.run_range(sp, 0, 64, out, quiet=True)
    res = ng.read_results(out)
    check("worker survives a crashing point and finishes the rest",
          (not ok) and len(res) == 64 and np.isnan(res[5][0]) and sum(np.isfinite(v[0]) for v in res.values()) == 63)
    ok2 = ng.run_range(sp, 0, 64, out, quiet=True)
    n_lines = sum(1 for _ in open(out))
    check("re-running is idempotent (no repeated work)", (not ok2) and n_lines == 64, f"{n_lines} lines")
    del os.environ["NORMGRID_FAKE_CRASH"]
    ok3 = ng.run_range(sp, 0, 64, out, retry_failed=True, quiet=True)
    check("--retry-failed recovers the failed point", ok3 and np.isfinite(ng.read_results(out)[5][0]))
    check("merge refuses missing points", raises(RuntimeError, lambda: ng.merge(sp, [os.path.join(d, "none.txt")], os.path.join(d, "x.h5"))))

    # ---- HDF5 round trip ----
    try:
        import h5py  # noqa: F401
        have_h5 = True
    except ImportError:
        have_h5 = False
        print("SKIP  HDF5 round trip (h5py not installed)")
    if have_h5:
        h5 = os.path.join(d, "t.h5")
        ng.merge(sp, [out], h5)
        t = ng.NormInterpolator.from_hdf5(h5)
        c = ng.coords_at(er2, 37)
        p = ng.physical_params(er2, c)
        v_direct = ng.read_results(out)[37][0]
        check("HDF5 table reproduces a stored grid point",
              abs(t(**{k: v for k, v in p.items() if k not in ("eps",)}, eps=p["eps"]) - v_direct) < 1e-12 * abs(v_direct))
        check("HDF5 table knows its band/region/fixed", t.band == "ER" and t.region == tuple(ng.DEFAULT_REGION))

# ---- per-band default boxes ------------------------------------------------
er_grid = ng.make_grid_spec("ER", ng.RECOMMENDED_NODES["ER"])
nr_grid = ng.make_grid_spec("NR", ng.RECOMMENDED_NODES["NR"])
check("ER default box narrows F0 to 0.1-0.35", er_grid["box"]["F0"] == [0.1, 0.35])
check("NR default box keeps F0 1e-5..1", nr_grid["box"]["F0"] == [1e-5, 1.0])
check("other axes are the same in both bands",
      all(er_grid["box"][a] == nr_grid["box"][a] for a in er_grid["box"] if a != "F0"))
er_held = ng.make_random_spec("ER", 40, seed=3)
check("ER held-out points stay inside the ER F0 box",
      all(0.1 <= ng.coords_at(er_held, i)["F0"] <= 0.35 for i in range(40)))

# ---- out-of-box errors carry what is needed to act on them ----
try:
    table(**dict(mid, V=3.9))
    e = None
except ng.OutOfBoxError as err:
    e = err
check("OutOfBoxError names the axis, value, box and query",
      e is not None and e.axis == "V" and e.value == 3.9 and e.box == table.box["V"] and e.query["V"] == 3.9)
check("OutOfBoxError message says to check the prior at setup", e is not None and "check_prior" in str(e))

# ---- MCMC setup: prior vs the table's box ----
design = dict(k=(0.13, 0.22), F0=(1e-5, 1.0), V=(2.7, 3.3), p0=(0.2 * ng.P0_MEAN, 1.8 * ng.P0_MEAN),
              p10=(0.3, 0.6), q0=(0.2 * ng.Q0_MEAN, 1.8 * ng.Q0_MEAN), q10=(0.2, 0.4))
table.check_prior(design)
check("the prior the default box was designed from is covered (q0 above 0.4 is cut by q10 >= q0)", True)
req = table.required_box(design)
check("required box: q0 capped at max q10", req["q0"][1] == 0.4)
check("required box: dp = [p10_lo - p0_hi, p10_hi - p0_lo]",
      np.allclose(req["dp"], (0.3 - 1.8 * ng.P0_MEAN, 0.6 - 0.2 * ng.P0_MEAN)))
check("required box: dq starts at 0 when q10 and q0 overlap", req["dq"][0] == 0.0)
tb = table.table_bounds()
table.check_prior(tb)
check("table_bounds() is itself covered", True)
check("table_bounds() contains the design prior's p10/q10",
      tb["p10"][0] <= 0.3 and tb["p10"][1] >= 0.6 and tb["q10"][0] <= 0.2 and tb["q10"][1] >= 0.4)


def prior_error(bounds):
    try:
        table.check_prior(bounds)
    except ng.PriorNotCoveredError as err:
        return str(err)
    return None


msg = prior_error(dict(design, V=(2.5, 4.0), q10=(0.2, 0.47)))
check("wide prior: every short axis is listed", msg is not None and "V: prior reaches" in msg and "dq: prior reaches" in msg
      and "k: prior" not in msg)
check("wide prior: q10 above 0.4 lifts the q0 cap, so q0 is short too", msg is not None and "q0: prior reaches" in msg)
check("wide prior: rebuild command with BOX rounded outward",
      msg is not None and 'BOX="V=2.5:4 q0=0.04743:0.427 dq=0:0.4226"' in msg and "sbatch slurm/build_table.job NR" in msg)
hint = msg.split("BOX=\"")[1].split("\"")[0] if msg else ""
wider = ng.parse_box(hint, "NR")
check("the suggested box covers the prior", wider["V"][0] <= 2.5 and wider["V"][1] >= 4.0
      and wider["dq"][1] >= 0.47 - 0.2 * ng.Q0_MEAN)
check("unbounded prior is an error of its own", "unbounded" in (prior_error(dict(design, V=(2.7, float("inf")))) or ""))
check("missing parameter is an error", "k: no prior bounds" in (prior_error({p: v for p, v in design.items() if p != "k"}) or ""))
check("a range for fixed Z is an error", "fixes Z" in (prior_error(dict(design, Z=(30.0, 34.0))) or ""))
check("the fixed Z value itself is fine", prior_error(dict(design, Z=32.0)) is None)
check("empty prior (q10 entirely below q0) is an error", "empty" in (prior_error(dict(design, q0=(0.3, 0.4), q10=(0.1, 0.2))) or ""))

# starting points (e.g. an emcee walker ball)
walkers = {p: rng.uniform(*tb[p], 64) for p in table.params()}
walkers["p10"] = np.maximum(walkers["p10"], walkers["p0"])
walkers["q10"] = np.maximum(walkers["q10"], walkers["q0"])
table.check_points(walkers)
check("points inside the box pass", True)
walkers["V"][[3, 7]] = 3.5
try:
    table.check_points(walkers)
    msg = None
except ng.PriorNotCoveredError as err:
    msg = str(err)
check("points outside: counted per axis, first one named", msg is not None and "2 of 64" in msg and "point 3" in msg)

# box specs
check("parse_box replaces only the given axes", ng.parse_box("V=2.5:4", "ER")["V"] == (2.5, 4.0)
      and ng.parse_box("V=2.5:4", "ER")["F0"] == (0.1, 0.35))
check("parse_box rejects an axis the band lacks", raises(ValueError, lambda: ng.parse_box("k=0.1:0.2", "ER")))
check("parse_box rejects LO >= HI", raises(ValueError, lambda: ng.parse_box("V=3:2", "NR")))
check("format_box_range rounds outward", ng.format_box_range(0.0474369, 0.4225631) == "0.04743:0.4226")
p10r, q10r = ng.x10_ranges(ng.default_box("NR"))
check("x10_ranges of the default box contain the design p10/q10", p10r[0] <= 0.3 and p10r[1] >= 0.6
      and q10r[0] <= 0.2 and q10r[1] >= 0.4)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
