"""
Example: inspect the integration of a single bin in isolation.

Use this when expected_counts_from_pdf reports a ~0 (or suspiciously
small) expected count for some bin and you need to see why -- run
chisquare_harness.debug_bin on just that bin instead of the full
n_bins loop. See debug_bin's docstring for what each of the three
reported numbers means.

Edit BIN and PARAMS below to match the case you're debugging, then run:
    LD_LIBRARY_PATH=lib python test/python/debug_bin_example.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "internal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from band_breakpoints import make_ridge_breakpoints
# Unnormalized: debug_bin inspects the PDF's shape over one bin, not an
# absolute probability.  For a fit or MCMC, use PpqPDF (python/ppq_pdf.py).
from _ppqfort_bindings import make_ppqg_pdf_unnormalized
from chisquare_harness import debug_bin

# Same ER/PpqG parameters as test_chisquare_er_simulator.py.
PARAMS = dict(
    F0  = 0.122,
    eps = 3.0e-3,
    V   = 3.0,
    p0  = 0.06421907,
    p10 = 0.48998486,
    q0  = 0.23718488,
    q10 = 0.27093151,
)

# The bin reported as ~0: (Ep_min, Ep_max, Eq_min, Eq_max).
BIN = (64.17183111582968, 66.41909324688707, 33.49076000562251, 184.4618870920465)

# n_workers here only threads the PDF's internal batched grid evaluation
# (debug_bin's brute-force reference calls it with tens of thousands of
# points at once); the quad-driven calls are all single points and are
# unaffected, so this doesn't change what debug_bin demonstrates.
pdf_fort = make_ppqg_pdf_unnormalized(**PARAMS, n_workers=os.cpu_count())
ridge_breakpoints = make_ridge_breakpoints(
    "ER", k=None, Z=None, eps=PARAMS["eps"], V=PARAMS["V"],
    p0=PARAMS["p0"], p10=PARAMS["p10"],
    q0=PARAMS["q0"], q10=PARAMS["q10"],
    er_max=700.0, n_window_widths=10.0,
)

result = debug_bin(pdf_fort, BIN, ridge_breakpoints)
