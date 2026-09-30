"""
Regression test for integrate_g_safe_inspect with multimodal integrands.

The safe integrator scans the integrand on a grid, groups the non-zero
samples into peaks, pads each group's range, and integrates each range
with adaptive quadrature.  When two padded ranges overlap they must be
*merged* -- integrating both would double-count the overlap, and
discarding one (as an older version did) silently drops the mass of an
entire peak.

Each case integrates a sum of unit-normalized Gaussians, so the exact
answer is the number of peaks.  The narrow-sigma cases produce exact
zeros between the peaks (the exponential underflows), which is what
splits the scan into multiple groups.

Run from the repository root:
  python test/python/test_safe_integrator_multimodal.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
from pq_dist_v10 import integrate_g_safe_inspect


def make_gaussian_sum(centers, sigma):
    """Sum of unit-normalized Gaussians; underflows to exact 0 far away."""
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    def g(er, ep, eq):
        total = 0.0
        for c in centers:
            total += norm * np.exp(-0.5 * ((er - c) / sigma) ** 2)
        return total

    return g


# (name, centers, sigma, res) -- Ep = Eq = 5 puts the scan range at
# [~0, 10]; the scan step is res / 2.
CASES = [
    # two sharp peaks whose padded ranges overlap: the historical
    # pop-the-first-range bug returned ~1 here instead of 2
    ("overlapping windows", [5.0, 5.5], 0.002, 0.1),
    # two peaks with well-separated windows (must remain two ranges)
    ("well separated",      [3.0, 7.0], 0.002, 0.1),
    # three sharp peaks, adjacent pairs overlapping: exercises overlap
    # handling beyond the first pair
    ("three peaks",         [4.0, 4.5, 5.0], 0.002, 0.1),
    # single broad peak (the common case; scan sees one group)
    ("single peak",         [5.0], 0.3, 0.1),
]

failures = 0
for name, centers, sigma, res in CASES:
    g = make_gaussian_sum(centers, sigma)
    (integral, err), _, peaks = integrate_g_safe_inspect(g, 5.0, 5.0, centers[0], res)
    expected = float(len(centers))
    ok = abs(integral - expected) < 1e-6
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {name:20s}: integral = {integral:.9f}  (expect {expected:.0f}, "
          f"{len(peaks)} range(s))")
    failures += 0 if ok else 1

if failures:
    print(f"\n{failures} case(s) FAILED")
    sys.exit(1)
print("\nAll cases pass.")
