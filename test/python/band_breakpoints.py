"""
Ridge breakpoints for integrating the band PDFs over histogram bins.

The problem
-----------
The expected count in a bin is the integral of the PDF over the bin
rectangle, computed in chisquare_harness.expected_counts_from_pdf with
nested adaptive quadrature (scipy quad over Ep, inner quad over Eq).
Adaptive quadrature starts from a fixed set of sample points across the
integration interval: if the PDF is a narrow ridge inside a wide bin,
every one of those initial points can land in the ~zero region on
either side of the ridge, and quad confidently returns ~0 without ever
subdividing toward the peak.  See quad_breakpoints_demo.py in this
directory for a self-contained demonstration of the failure and the fix.

The fix is quad's `points=` argument: telling the inner (Eq) quadrature
where the ridge is forces a subdivision there, so the peak cannot be
missed.  This module computes those breakpoints from the band physics.

The physics: where the ridge is
-------------------------------
A recoil of true energy Er produces (noiselessly)

    N  = Y(Er) * Er / eps          e/h pairs, so
    Eq = Y(Er) * Er                (ionization channel)
    Ep = Er + (V/1000) * N
       = Er * (1 + Y(Er) * V / (1000 * eps))   (phonon + Neganov-Luke)

with Y(Er) the Lindhard yield (k, Z) for nuclear recoils and Y = 1 for electron
recoils.  These are parametric equations for the band centroid: each Er
gives one point (Ep(Er), Eq(Er)) on the ridge.  We need the centroid in
the form "most probable Eq at a given Ep" — that direction because the
harness's *inner* quadrature runs over Eq at fixed Ep, so the
breakpoints must be Eq values.  Ep(Er) = ep cannot be solved for Er
algebraically (Y makes it transcendental), but both Ep(Er) and Eq(Er)
are monotone in Er, so the centroid is tabulated on an Er grid once and
np.interp(ep, ep_tab, eq_tab) performs the inversion: find where the
table's Ep equals ep, read off the Eq at the same Er.

The geometry: how wide the ridge is
-----------------------------------
The breakpoints must bracket the peak, so we need the band's local
width in the Eq direction at fixed measured Ep.  Picture the band as a
tilted ridge in the (Ep, Eq) plane and slice it vertically at ep.  Two
independent noise sources spread events along that vertical slice:

* Charge noise smears Eq directly: a vertical spread of sigq.
* Phonon noise smears the measured Ep, i.e. it displaces events
  *horizontally*.  An event that landed in the slice because its Ep
  fluctuated by delta really belongs to a ridge point whose Eq differs
  by (dEq_ridge/dEp) * delta — so horizontal noise, viewed through the
  tilt of the band, appears as a vertical spread of slope * sigp.  A
  flat band gets no vertical smearing from phonon noise; a steep band
  gets a lot.

The two contributions are independent Gaussians, so they add in
quadrature (np.hypot is sqrt(a^2 + b^2), written that way only for
numerical safety):

    width(Ep) = sqrt( sigq(Eq_ridge)^2 + (dEq_ridge/dEp * sigp(Ep))^2 )

with the slope taken by central finite difference of the interpolated
ridge.  This width ignores the Fano broadening (the spread in N), which
is harmless here: the breakpoints sit at ridge +/- 10 widths, so even a
substantially underestimated width still brackets the whole peak, and
the adaptive quadrature does the accurate work inside the bracket.

The breakpoints handed to the harness are the ridge center and the
window of +/- 10 local widths around it; the harness clips them to
each bin.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import pq_dist_v10 as ppq


def make_ridge_breakpoints(band, *, k, Z, eps, V,
                           p0, p10,
                           q0, q10,
                           er_max, n_window_widths):
    """
    Build an inner_points_func for chisquare_harness.expected_counts_from_pdf.

    Parameters
    ----------
    band : "NR" or "ER"
        Nuclear recoils use the Lindhard yield model Y(Er, k, Z); electron
        recoils use Y = 1 (the k and Z arguments are then ignored).
    k, Z, eps, V, p0, p10, q0, q10 : float
        Detector parameters, matching the PDF being integrated.
    er_max : float
        Upper end of the tabulated Er range (keV); must comfortably
        exceed the largest Ep of interest.
    n_window_widths : float
        Half-width of the breakpoint window in units of the local band
        width.

    Returns
    -------
    inner_points_func : callable
        inner_points_func(ep, ymin, ymax) -> [lo, ridge, hi], the Eq
        breakpoints for the inner quadrature at phonon energy ep.  The
        harness drops any that fall outside the bin's (ymin, ymax).
    """
    if band == "NR":
        y_tab_func = lambda er_tab: ppq.Y(er_tab, k=k, Z=Z)
    elif band == "ER":
        y_tab_func = lambda er_tab: np.ones_like(er_tab)
    else:
        raise ValueError(f"band must be 'NR' or 'ER', got {band!r}")

    # Tabulate the noiseless band centroid Er -> (Ep, Eq) and invert by
    # interpolation (both coordinates are monotone in Er).
    er_tab = np.geomspace(1e-3, er_max, 4000)
    y_tab = y_tab_func(er_tab)
    ep_tab = er_tab * (1.0 + y_tab * V / (1000.0 * eps))
    eq_tab = y_tab * er_tab

    def eq_ridge(ep):
        return np.interp(ep, ep_tab, eq_tab)

    def inner_points_func(ep, ymin, ymax):
        eqr = float(eq_ridge(ep))
        # local band width in Eq: charge resolution plus the phonon
        # resolution mapped through the ridge slope
        dep = max(1e-3, 0.01 * ep)
        slope = (float(eq_ridge(ep + dep)) - float(eq_ridge(ep - dep))) / (2 * dep)
        sigp_val = float(ppq.sigp(ep, eps=eps, V=V, p0=p0, p10=p10))
        sigq_val = float(ppq.sigq(eqr, q0=q0, q10=q10))
        width = np.hypot(sigq_val, slope * sigp_val)
        return [eqr - n_window_widths * width, eqr,
                eqr + n_window_widths * width]

    return inner_points_func
