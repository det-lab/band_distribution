"""
Chi-square distribution test: nuclear-recoil physics simulator vs Fortran PpqN.

Generates (Ep, Eq) pairs from the physics model
(Er ~ PErN, N | Er ~ TruncNormal float, Ep/Eq | Er,N ~ Normal) and checks
that chi-square values against the Fortran PpqN PDF follow chi2(dof).  This
validates that PpqN correctly describes the physics — unlike
test_chisquare_ppqn.py, which only checks internal self-consistency.

Observable range
----------------
  EP_RANGE = (3, 250) keV  — phonon channel; below the ~5 keV hardware
                             trigger threshold to ensure full coverage.
  EQ_RANGE = (0.5, 100) keV — charge channel; lower bound ~2 sigma above
                              zero (q0 = 0.237 keV).  Using 3 keV here would
                              exclude ~60% of NR events whose low-Er true Eq
                              sits near 1-2 keV.

Usage
-----
    LD_LIBRARY_PATH=lib python test/python/test_chisquare_nr_simulator.py [n_throws] [n_bins]

n_throws defaults to 1000 for a quick check; use 10000 for a thorough test.
n_bins defaults to 400; use a smaller value (e.g. 64) for a fast smoke test
of the wiring, since the one-time bin integration dominates the wall time.

The chi-square histogram is written to figures/chisquare_nr_simulator.png.

Approximate wall times (18 workers, x86 emulation on an Apple Silicon Mac;
native x86 hardware should be faster):

    n_throws=100,   n_bins=64   ~ 1 min   (smoke test)
    n_throws=10000, n_bins=400  ~ 4 min   (full validation)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist, kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from band_breakpoints import make_ridge_breakpoints
from generate_events import generate_NR_events
# Unnormalized is fine: this test only compares the PDF's *shape* against
# simulated events (chi-square binning), not absolute probability.  For a
# fit or MCMC, use PpqPDF (python/ppq_pdf.py) instead.
from _ppqfort_bindings import make_ppqn_pdf_unnormalized
from chisquare_harness import run_chisquare_test

# ---------------------------------------------------------------------------
# Physics parameters (from nrFanoII_paper2022/python/detector_params.txt)
# ---------------------------------------------------------------------------
PARAMS = dict(
    k   = 0.18,
    Z   = 32.0,
    F0  = 0.122,
    eps = 3.0e-3,
    V   = 3.0,
    p0  = 0.06421907,
    p10 = 0.48998486,
    q0  = 0.23718488,
    q10 = 0.27093151,
)

SPECTRUM_PARAMS = dict(
    PNa = 0.53693208,
    PNb = 6.41515782,
    PNd = 23.71789286,
)

EP_RANGE = (3.0,   250.0)   # keV — phonon channel
EQ_RANGE = (0.5,   100.0)   # keV — charge channel (see module docstring)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_EVENTS  = 10_000
N_THROWS  = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000
N_BINS    = int(sys.argv[2]) if len(sys.argv) > 2 else 400
N_WORKERS = os.cpu_count()

# ---------------------------------------------------------------------------
# Sampler: generate exactly n in-range NR events from the physics simulator.
#
# Events with Ep < EP_RANGE[0] (roughly Er < 2.5 keV) or Eq < EQ_RANGE[0]
# fall outside the observable region; oversample and keep the first n that
# pass the range cut.  The oversampling factor of 2 is conservative; a while
# loop handles the rare cases where a second pass is needed.
# ---------------------------------------------------------------------------
def nr_sampler(n, seed=None):
    rng = np.random.default_rng(seed)
    ep_acc, eq_acc = [], []
    n_acc = 0
    attempt_seed = int(rng.integers(0, 2**31))

    while n_acc < n:
        n_gen = max((n - n_acc) * 2, 1000)
        Ep, Eq, _, _ = generate_NR_events(
            n_gen, seed=attempt_seed, **PARAMS, **SPECTRUM_PARAMS
        )
        mask = (
            (Ep >= EP_RANGE[0]) & (Ep <= EP_RANGE[1]) &
            (Eq >= EQ_RANGE[0]) & (Eq <= EQ_RANGE[1])
        )
        ep_acc.append(Ep[mask])
        eq_acc.append(Eq[mask])
        n_acc += int(mask.sum())
        attempt_seed += 1

    Ep_all = np.concatenate(ep_acc)[:n]
    Eq_all = np.concatenate(eq_acc)[:n]
    return Ep_all, Eq_all

# ---------------------------------------------------------------------------
# Build the Fortran PDF wrapper
# ---------------------------------------------------------------------------
print(f"Loading Fortran PpqN PDF ({N_WORKERS} workers)...")
pdf_fort = make_ppqn_pdf_unnormalized(**PARAMS, n_workers=N_WORKERS)

# Quick sanity check on the sampler
print("Spot-checking sampler output range...")
t0 = time.time()
Ep_chk, Eq_chk = nr_sampler(500)
print(f"  Generated 500 events in {time.time()-t0:.2f} s")
print(f"  Ep: [{Ep_chk.min():.2f}, {Ep_chk.max():.2f}] keV")
print(f"  Eq: [{Eq_chk.min():.4f}, {Eq_chk.max():.2f}] keV")

# ---------------------------------------------------------------------------
# Run the chi-square test
# ---------------------------------------------------------------------------
t_start = time.time()
chi2_vals, dof, bins, expected = run_chisquare_test(
    pdf_fort, nr_sampler, EP_RANGE, EQ_RANGE,
    n_events   = N_EVENTS,
    n_throws   = N_THROWS,
    n_bins     = N_BINS,
    # band-ridge breakpoints keep the per-bin quadrature from missing
    # the narrow band inside wide (tail) bins; see band_breakpoints.py
    inner_points_func = make_ridge_breakpoints(
        "NR", k=PARAMS["k"], Z=PARAMS["Z"], eps=PARAMS["eps"],
        V=PARAMS["V"], p0=PARAMS["p0"], p10=PARAMS["p10"],
        q0=PARAMS["q0"], q10=PARAMS["q10"],
        er_max=700.0, n_window_widths=10.0),
    n_workers  = N_WORKERS,
    batch_size = 50,
    seed       = 42,
)
elapsed = time.time() - t_start

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
ks_stat, ks_p = kstest(chi2_vals, chi2_dist(dof).cdf)

print(f"\n{'='*60}")
print(f"Chi-square test: NR simulator vs Fortran PpqN  (dof = {dof})")
print(f"{'='*60}")
print(f"  n_events per throw : {N_EVENTS:,}")
print(f"  n_throws           : {N_THROWS:,}")
print(f"  n_bins             : {len(bins)}  (target {N_BINS})")
print(f"  min expected count : {expected.min():.2f}")
print(f"  total wall time    : {elapsed/60:.1f} min")
print()
print(f"  Mean chi2    : {chi2_vals.mean():.3f}   (expect {dof})")
print(f"  Var  chi2    : {chi2_vals.var():.3f}   (expect {2*dof:.0f})")
print(f"  KS statistic : {ks_stat:.4f}")
print(f"  KS p-value   : {ks_p:.4f}   (pass threshold: > 0.05)")
print()
passed = ks_p > 0.05 and abs(chi2_vals.mean() - dof) / dof < 0.02
print("PASS" if passed else "FAIL",
      "— chi-square distribution",
      "consistent with" if passed else "DEVIATES FROM",
      f"chi2({dof}).")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
os.makedirs(REPO_ROOT / "figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(chi2_dist(dof).ppf(1e-4), chi2_dist(dof).ppf(1 - 1e-4), 500)
ax.plot(x, chi2_dist(dof).pdf(x), color="C1", lw=2, label=rf"$\chi^2({dof})$ theory")
ax.hist(chi2_vals, bins=80, density=True, alpha=0.6, color="C0",
        label=f"{N_THROWS:,} throws")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel("Probability density")
ax.set_title("Chi-square: NR simulator vs Fortran PpqN")
stats_text = (
    f"mean = {chi2_vals.mean():.2f}  (expect {dof})\n"
    f"var  = {chi2_vals.var():.1f}  (expect {2*dof:.0f})\n"
    f"KS p = {ks_p:.3f}"
)
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.legend()
fig.tight_layout()
outfile = REPO_ROOT / "figures" / "chisquare_nr_simulator.png"
fig.savefig(outfile, dpi=150)
print(f"\nPlot saved to {outfile}")
