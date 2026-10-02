"""
Tests for PpqPDF's MCMC setup checks with normalization tables: prior_bounds
(shared and band-specific keys), check_points, table_bounds, and table names
that are neither files nor registered.  Small in-memory tables stand in for
real ones, so no data files are needed -- but importing ppq_pdf loads the
Fortran library, so run it where that is built (e.g. in the container):

    LD_LIBRARY_PATH=lib python test/python/test_ppq_pdf_tables.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "cli"))
import normgrid as ng
from ppq_pdf import PpqPDF

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def error_of(exc, fn):
    try:
        fn()
    except exc as e:
        return str(e)
    return None


REGION = (2.5, 350.0, 0.75, 200.0)


def fake_table(band):
    """A 2-node-per-axis table over the band's default box (values are
    irrelevant here; only the box and metadata are checked)."""
    axes = ng.BAND_AXES[band]
    box = ng.default_box(band)
    nodes = {a: ng.chebyshev_lobatto(*box[a], 2) for a in axes}
    return ng.NormInterpolator(axes, nodes, np.ones([2] * len(axes)), band=band, region=REGION,
                               fixed=ng._base_spec("grid", band, REGION, ng.DEFAULT_FIXED, box, 1e-7, 1e-13)["fixed"])


nr, er = fake_table("NR"), fake_table("ER")
make = lambda **kw: PpqPDF(*REGION, [10.0], [5.0], ppqn_table=nr, ppqg_table=er, **kw)

shared = dict(V=(2.7, 3.3), p0=(0.2 * ng.P0_MEAN, 1.8 * ng.P0_MEAN), p10=(0.3, 0.6),
              q0=(0.2 * ng.Q0_MEAN, 1.8 * ng.Q0_MEAN), q10=(0.2, 0.4))
prior = dict(shared, NR={"k": (0.13, 0.22), "F0": (1e-5, 1.0)}, ER={"F0": (0.15, 0.3)})

make(prior_bounds=prior)
check("design prior with band-specific F0 passes both tables", True)
msg = error_of(ng.PriorNotCoveredError, lambda: make(prior_bounds=dict(prior, ER={"F0": (0.05, 0.3)})))
check("ER F0 below the ER box fails, naming the ER table", msg is not None and "ER table" in msg and "F0" in msg)
msg = error_of(ng.PriorNotCoveredError, lambda: make(prior_bounds=dict(shared, k=(0.13, 0.22), F0=(1e-5, 1.0))))
check("one shared F0 wider than the ER box fails (the bands' Fano factors differ)", msg is not None and "ER table" in msg)
msg = error_of(ng.PriorNotCoveredError, lambda: make(prior_bounds=dict(prior, V=(2.5, 4.0))))
check("shared V too wide fails", msg is not None and "V: prior reaches" in msg)
check("k at the top level is ignored for ER", error_of(Exception, lambda: make(prior_bounds=dict(
    shared, k=(0.13, 0.22), NR={"F0": (1e-5, 1.0)}, ER={"F0": (0.15, 0.3)}))) is None)
check("prior_bounds without a table is an error",
      error_of(ValueError, lambda: PpqPDF(*REGION, [10.0], [5.0], prior_bounds=prior)) is not None)

pdf = make()
tb = pdf.table_bounds()
check("table_bounds: one box per band", set(tb) == {"NR", "ER"} and "k" in tb["NR"] and "k" not in tb["ER"]
      and np.allclose(tb["ER"]["F0"], (0.1, 0.35)))
make(prior_bounds=dict({p: v for p, v in tb["NR"].items() if p not in ("k", "F0")},
                       NR={"k": tb["NR"]["k"], "F0": tb["NR"]["F0"]}, ER={"F0": tb["ER"]["F0"]}))
check("table_bounds used as the prior passes", True)

rng = np.random.default_rng(1)
n = 32
walkers = {p: rng.uniform(*shared[p], n) for p in ("V", "p0")}
# shared["q0"] = (0.2, 1.8) * Q0_MEAN is the *design prior*'s range; the table's
# actual q0 box is narrower (capped at max q10 = 0.4, see table_bounds() above),
# so draw q0 from there directly rather than from shared["q0"] and clamping --
# clamping against q10 doesn't help when the q0 draw itself already exceeds 0.4.
walkers["q0"] = rng.uniform(*tb["NR"]["q0"], n)
walkers["p10"] = rng.uniform(0.3, 0.6, n)
walkers["q10"] = np.maximum(rng.uniform(0.2, 0.4, n), walkers["q0"])
walkers["NR"] = {"k": rng.uniform(0.13, 0.22, n), "F0": rng.uniform(1e-5, 1.0, n)}
walkers["ER"] = {"F0": rng.uniform(0.15, 0.3, n)}
pdf.check_points(walkers)
check("walkers inside both tables pass", True)
walkers["ER"]["F0"][5] = 0.5
msg = error_of(ng.PriorNotCoveredError, lambda: pdf.check_points(walkers))
check("one walker outside the ER F0 range fails, naming it", msg is not None and "1 of 32" in msg and "point 5" in msg)

msg = error_of(KeyError, lambda: PpqPDF(*REGION, [10.0], [5.0], ppqn_table="no_such_table"))
check("a table that is neither a file nor registered: KeyError", msg is not None and "registered" in msg)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
