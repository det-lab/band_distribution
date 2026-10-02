"""
Tests for python/cli/normplan.py (the node-count planner) that need no Fortran:
the study spec's layout, the coefficient-tail criterion on functions with
known convergence, and the whole study -> analyze path through the real
worker using normgrid's analytic stand-in evaluators (NORMGRID_FAKE=smooth
converges at a known rate; =kink never converges).

Run from the repository root:  python test/python/test_normplan.py
"""

import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "cli"))
import normgrid as ng
import normplan as npl

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


REGION, EPSREL = (2.5, 350.0, 0.75, 200.0), 1e-7
NODES = ng.chebyshev_lobatto(-1.0, 1.0, npl.STUDY_NODES)

# ---- study spec layout ----
nr = npl.make_study_spec("NR", REGION, EPSREL)
er = npl.make_study_spec("ER", REGION, EPSREL)
check("NR study: 7 axes x 3 baselines x 33 nodes", ng.n_points(nr) == 7 * 3 * 33)
check("ER study has no k axis: 6 x 3 x 33", ng.n_points(er) == 6 * 3 * 33 and "k" not in ng.axes_of(er))
i, b = 4, 1                                            # axis "dp", second baseline
block = [ng.coords_at(nr, (i * 3 + b) * 33 + j) for j in range(33)]
varying = [a for a in ng.axes_of(nr) if len({c[a] for c in block}) > 1]
check("one axis varies within a block, the rest sit at the baseline", varying == ["dp"])
check("the varying axis runs over the whole box, endpoints included",
      np.isclose(block[0]["dp"], nr["box"]["dp"][0], rtol=0, atol=1e-12)
      and np.isclose(block[-1]["dp"], nr["box"]["dp"][1], rtol=0, atol=1e-12))
check("baseline 0 is the box centre (here on the k axis block, a non-varying axis)",
      abs(ng.coords_at(nr, 16)["p0"] - 0.5 * sum(nr["box"]["p0"])) < 1e-12)
with tempfile.TemporaryDirectory() as d:
    ng.save_spec(nr, os.path.join(d, "s.json"))
    check("study spec round-trips through JSON",
          ng.load_spec(os.path.join(d, "s.json"))["region"] == list(REGION) and ng.n_points(ng.load_spec(os.path.join(d, "s.json"))) == ng.n_points(nr))

# ---- the criterion on functions with known degree / smoothness ----
poly5 = np.polynomial.chebyshev.chebval(NODES, [0.3, 1.0, -0.4, 0.2, 0.1, 0.05])
n, t = npl.nodes_needed(poly5, 1e-10, 3)
check("degree-5 polynomial needs exactly 6 nodes", n == 6, f"got {n}")
n_min, _ = npl.nodes_needed(np.full(33, 2.0) + 1e-3 * NODES, 1e-10, 3)
check("a linear function still gets min_nodes", n_min == 3, f"got {n_min}")
check("looser target => fewer nodes",
      npl.nodes_needed(np.exp(4 * NODES), 1e-3, 3)[0] < npl.nodes_needed(np.exp(4 * NODES), 1e-9, 3)[0])
check("a kink is not resolved: NotConverged", raises(npl.NotConverged, lambda: npl.nodes_needed(np.abs(NODES - 0.13), 1e-7, 3)))

# ---- whole path: study -> real worker -> analyze ----
os.environ["NORMGRID_FAKE"] = "smooth"
with tempfile.TemporaryDirectory() as d:
    sp, res = os.path.join(d, "study.json"), os.path.join(d, "res_study.txt")
    ng.save_spec(nr, sp)
    check("worker evaluates the whole study", ng.run_range(sp, 0, ng.n_points(nr), res, quiet=True))
    spec, rep = npl.analyze(nr, [res], tol=1e-7)
    n = {a: rep["axes"][a]["n"] for a in rep["axes"]}
    others = [a for a in n if a not in ("V", "q0")]
    check("smooth stand-in: untouched axes get min_nodes", all(n[a] == 3 for a in others), str(n))
    check("V (exp 3u) needs more, q0 (exp 6u) more still", 3 < n["V"] < n["q0"] <= 29, str(n))
    check("report total equals the grid spec's size", rep["points"] == ng.n_points(spec) == int(np.prod(list(n.values()))))
    check("grid spec keeps region, box and epsrel",
          spec["region"] == list(REGION) and spec["box"] == nr["box"] and spec["epsrel"] == EPSREL and spec["kind"] == "grid")

    # the chosen n really does deliver tol along that axis (the meaning of the criterion)
    from numpy.polynomial import chebyshev as cheb
    lo, hi = nr["box"]["q0"]
    base = ng.coords_at(nr, (ng.axes_of(nr).index("q0") * 3 + 0) * 33)     # baseline 0, q0 at its lowest node
    f = lambda q0: ng._evaluate(nr, ng.physical_params(nr, {**base, "q0": float(q0)}), EPSREL)
    xn = ng.chebyshev_lobatto(lo, hi, n["q0"])
    c = cheb.chebfit(2 * (xn - lo) / (hi - lo) - 1, [f(x) for x in xn], n["q0"] - 1)
    xt = np.linspace(lo, hi, 201)
    err = max(abs(cheb.chebval(2 * (x - lo) / (hi - lo) - 1, c) / f(x) - 1) for x in xt)
    check("q0 interpolant at the planned n meets tol", err <= 1e-7, f"max rel err {err:.1e}")

    _, rep_tight = npl.analyze(nr, [res], tol=1e-10)
    check("tighter tol => at least as many nodes on every axis",
          all(rep_tight["axes"][a]["n"] >= n[a] for a in n) and rep_tight["axes"]["q0"]["n"] > n["q0"])
    check("--max-points guards the grid size", raises(RuntimeError, lambda: npl.analyze(nr, [res], tol=1e-7, max_points=1000)))

    partial = os.path.join(d, "res_partial.txt")
    with open(res) as fin, open(partial, "w") as fout:
        fout.writelines(list(fin)[:-5])
    check("missing study points are reported, not guessed", raises(RuntimeError, lambda: npl.analyze(nr, [partial], tol=1e-7)))

os.environ["NORMGRID_FAKE"] = "kink"
with tempfile.TemporaryDirectory() as d:
    sp, res = os.path.join(d, "study.json"), os.path.join(d, "res_study.txt")
    ng.save_spec(nr, sp)
    ng.run_range(sp, 0, ng.n_points(nr), res, quiet=True)
    check("a rough axis stops the plan (NotConverged)", raises(npl.NotConverged, lambda: npl.analyze(nr, [res], tol=1e-7)))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
