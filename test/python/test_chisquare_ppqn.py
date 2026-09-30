"""
Chi-square distribution test for the Fortran PpqN (nuclear-recoil band) PDF.

Uses chisquare_harness.run_chisquare_test (equibin bins, PDF-integrated
expected counts, batched chi-square throws; see that module's
docstring for the general workflow).  Specific to this test:

- The PDF comes from lib/libband_distribution.so via python/ppqfort_pdf.py
  (threaded chunked PpqN_vector calls).
- The lintsampler grid uses feature-scaled (non-uniform) edges: the band
  width tracks the detector resolutions sigp(Ep) and sigq(Eq), which grow
  roughly linearly with energy, so the edges are spaced proportionally to
  the local resolution.  This keeps a constant number of cells per band
  width with far fewer points than a uniform grid would need.
- The grid evaluation (~1e6 Fortran calls) is cached in PPQN_GRID_CACHE;
  delete the file to force re-evaluation.
- Expected counts integrate the *true* Fortran PDF per bin.  Because the
  band is a narrow ridge crossing wide tail bins, the inner (Eq)
  quadrature is given the ridge location as breakpoints so it cannot
  miss the peak.

Run from the repository root with:
  LD_LIBRARY_PATH=lib python test/python/test_chisquare_ppqn.py [n_throws] [n_bins]

n_throws defaults to 100,000 and n_bins to 64.
"""

import hashlib
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
import pq_dist_v10 as ppq
import sample_from_pdf as spdf
from band_breakpoints import make_ridge_breakpoints
from chisquare_harness import run_chisquare_test
from ppqfort_pdf import make_ppqn_pdf

# ---------------------------------------------------------------------------
# Physics parameters (all explicit — the Fortran wrapper takes no defaults)
# ---------------------------------------------------------------------------
PARAMS = dict(
    k   = 0.18,         # Lindhard ionization yield calibration constant
    Z   = 32.0,         # target atomic number (germanium)
    F0  = 0.122,        # Fano factor (constant)
    eps = 3.0e-3,       # keV per e/h pair
    V   = 3.0,          # bias voltage [V]
    p0  = 0.06421907,   # phonon resolution
    p10 = 0.48998486,
    q0  = 0.23718488,   # charge resolution
    q10 = 0.27093151,
)

EP_RANGE = (2.5, 350.0)
EQ_RANGE = (0.75, 250.0)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_EVENTS        = 1_000     # events per chi-square throw
N_THROWS        = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
N_BINS          = int(sys.argv[2]) if len(sys.argv) > 2 else 64   # equibin target bins
CELLS_PER_SIGMA = 6.0       # sampling-grid cells per local band width
N_WORKERS       = os.cpu_count()
PPQN_GRID_CACHE = REPO_ROOT / "ppqn_vertex_grid.npz"

# ---------------------------------------------------------------------------
# Detector resolutions (from the python reference implementation), used
# for the feature-scaled grid edges; band-ridge breakpoints for the
# per-bin quadrature come from band_breakpoints.py.
# ---------------------------------------------------------------------------
def sigp(ep):
    return ppq.sigp(ep, eps=PARAMS["eps"], V=PARAMS["V"],
                    p0=PARAMS["p0"], p10=PARAMS["p10"])

def sigq(eq):
    return ppq.sigq(eq, q0=PARAMS["q0"], q10=PARAMS["q10"])

ridge_breakpoints = make_ridge_breakpoints(
    "NR", k=PARAMS["k"], Z=PARAMS["Z"], eps=PARAMS["eps"], V=PARAMS["V"],
    p0=PARAMS["p0"], p10=PARAMS["p10"], q0=PARAMS["q0"], q10=PARAMS["q10"],
    er_max=700.0, n_window_widths=10.0)

# ---------------------------------------------------------------------------
# Build (or load) the sampling grid
# ---------------------------------------------------------------------------
print("Building Fortran PpqN pdf wrapper "
      f"({N_WORKERS} workers)...")
pdf_fort = make_ppqn_pdf(**PARAMS, n_workers=N_WORKERS)

ep_edges = spdf.make_feature_scaled_edges(sigp, *EP_RANGE, CELLS_PER_SIGMA)
eq_edges = spdf.make_feature_scaled_edges(sigq, *EQ_RANGE, CELLS_PER_SIGMA)
print(f"Feature-scaled grid: {ep_edges.size} x {eq_edges.size} vertices "
      f"({CELLS_PER_SIGMA} cells per local band width)")

# cache key includes the shared library hash so a rebuilt .so invalidates it
with open(REPO_ROOT / "lib" / "libband_distribution.so", "rb") as f:
    lib_hash = hashlib.sha256(f.read()).hexdigest()

vertex_values = None
if os.path.exists(PPQN_GRID_CACHE):
    cache = np.load(PPQN_GRID_CACHE)
    if (np.array_equal(cache["ep_edges"], ep_edges)
            and np.array_equal(cache["eq_edges"], eq_edges)
            and np.array_equal(cache["params"],
                               np.array(list(PARAMS.values())))
            and str(cache["lib_hash"]) == lib_hash):
        vertex_values = cache["values"]
        print(f"Loaded cached vertex grid from {PPQN_GRID_CACHE}")
    else:
        print("Cache exists but edges/params/library differ — re-evaluating.")

if vertex_values is None:
    print(f"Evaluating PpqN at {ep_edges.size * eq_edges.size:,} vertices "
          f"(cached afterwards in {PPQN_GRID_CACHE})...")
    t0 = time.time()
    Ep_g, Eq_g = np.meshgrid(ep_edges, eq_edges, indexing="ij")
    ep_flat, eq_flat = Ep_g.ravel(), Eq_g.ravel()
    vertex_values = np.empty(ep_flat.size)
    block = 50_000
    for start in range(0, ep_flat.size, block):
        stop = min(start + block, ep_flat.size)
        vertex_values[start:stop] = pdf_fort(ep_flat[start:stop],
                                             eq_flat[start:stop])
        done = stop / ep_flat.size
        elapsed = time.time() - t0
        eta = elapsed / done * (1 - done)
        print(f"  {stop:>9,} / {ep_flat.size:,}  ({100*done:.0f}%, "
              f"eta {eta/60:.1f} min)", end="\r" if stop < ep_flat.size else "\n",
              flush=True)
    vertex_values = vertex_values.reshape(ep_edges.size, eq_edges.size)
    np.savez_compressed(
        PPQN_GRID_CACHE,
        ep_edges=ep_edges, eq_edges=eq_edges, values=vertex_values,
        params=np.array(list(PARAMS.values())), lib_hash=lib_hash,
    )
    print(f"Grid evaluated in {(time.time()-t0)/60:.1f} min, "
          f"saved to {PPQN_GRID_CACHE}")

pdf_lookup = spdf.make_vertex_lookup_pdf(ep_edges, eq_edges, vertex_values)
sampler = spdf.build_sampler_from_edges(pdf_lookup, ep_edges, eq_edges)

# ---------------------------------------------------------------------------
# Run the chi-square test (expected counts use the TRUE Fortran PDF)
# ---------------------------------------------------------------------------
chi2_vals, dof, bins, expected = run_chisquare_test(
    pdf_fort, sampler, EP_RANGE, EQ_RANGE,
    n_events          = N_EVENTS,
    n_throws          = N_THROWS,
    n_bins            = N_BINS,
    inner_points_func = ridge_breakpoints,
    n_workers         = N_WORKERS,
    batch_size        = 5_000,
    seed              = 42,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Chi-square test: Fortran PpqN  (dof = {dof})")
print(f"{'='*55}")
print(f"  n_events per throw : {N_EVENTS:,}")
print(f"  n_throws           : {N_THROWS:,}")
print(f"  n_bins             : {len(bins)}  (target {N_BINS})")
print(f"  min expected count : {expected.min():.1f}")
print()
print(f"  Mean chi2   : {chi2_vals.mean():.3f}   (expect {dof})")
print(f"  Var  chi2   : {chi2_vals.var():.3f}   (expect {2*dof})")

ks_stat, ks_p = kstest(chi2_vals, chi2_dist(dof).cdf)
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  KS p-value  : {ks_p:.4f}   (expect >> 0.05)")
print()
passed = ks_p > 0.05 and abs(chi2_vals.mean() - dof) / dof < 0.02
if passed:
    print("PASS — chi-square distribution consistent with chi2(dof).")
else:
    print("FAIL — chi-square distribution deviates from chi2(dof).")

# ---------------------------------------------------------------------------
# Reference plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

x = np.linspace(chi2_dist(dof).ppf(1e-4), chi2_dist(dof).ppf(1 - 1e-4), 500)
ax.plot(x, chi2_dist(dof).pdf(x), color="C1", lw=2, label=rf"$\chi^2({dof})$ theory")
ax.hist(chi2_vals, bins=80, density=True, alpha=0.6, color="C0",
        label=f"{N_THROWS:,} throws")

ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel("Probability density")
ax.set_title("Chi-square distribution: Fortran PpqN")

stats_text = (
    f"mean = {chi2_vals.mean():.2f}  (expect {dof})\n"
    f"var  = {chi2_vals.var():.1f}  (expect {2*dof})\n"
    f"KS p = {ks_p:.3f}"
)
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax.legend()
fig.tight_layout()
os.makedirs(REPO_ROOT / "figures", exist_ok=True)
outfile = REPO_ROOT / "figures" / "chisquare_ppqn.png"
fig.savefig(outfile, dpi=150)
print(f"\nPlot saved to {outfile}")
