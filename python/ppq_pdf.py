"""
PpqPDF: the fixed context for one fit/MCMC run -- an (Ep, Eq) region, the
observed dataset, and the normalization-integral's convergence tolerance
-- exposing, per parameter point, the un-normalized PDF at the bound
dataset, the region's normalization integral, and their ratio (normalized
PDF values), for both the NR (PpqN) and ER (PpqG) bands.

All physics parameters (k, Z, F0, eps, V, p0, p10, q0, q10) are passed
fresh to every method call rather than bound at construction: those are
what an MCMC step actually varies, while the region, dataset, and
tolerance don't change across a run.

Built entirely on the already-validated primitives in ppqfort_pdf.py --
ppqn_region/ppqg_region (-> Fortran PpqN_region/PpqG_region, a doubling-
verified nested Gauss-Legendre quadrature) and make_ppqn_pdf/make_ppqg_pdf
(-> Fortran PpqN_vector/PpqG_vector) -- no new Fortran code and no new
numerics.

For MCMC, pass precomputed normalization tables (python/normgrid.py,
built once on a batch system) as ppqn_table/ppqg_table: the integral then
costs tens of microseconds instead of ~2 s, and raises rather than ever
extrapolating outside the table's parameter box.  A table is a file path or
a registered name (python/normtables.py fetches and caches it).

At MCMC setup, pass the prior's hard bounds as prior_bounds so a prior that
reaches outside a table fails at construction, with the box to rebuild the
table with, rather than hours into the run; check the starting walker
positions with check_points.  table_bounds() gives a prior box the tables
cover.  In log_prob, reject points outside the prior before calling the
likelihood:

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf          # never reaches the table
        return lp + log_likelihood(theta)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppqfort_pdf import make_ppqg_pdf, make_ppqn_pdf, ppqg_region, ppqn_region


class PpqPDF:
    """
    ep_min, ep_max, eq_min, eq_max : float
        The (Ep, Eq) region the PDF is normalized over.
    ep_data, eq_data : array_like
        The fixed observed dataset this fit is evaluated against.
    norm_epsrel, norm_epsabs : float or None
        Convergence tolerance for the normalization integral's doubling-
        verified quadrature (see PpqN_region's doc comment in
        src/PpqFort_m.f90) -- unrelated to the size of ep_data/eq_data,
        hence the norm_ prefix.  Required for any band without a table.
    ppqn_table, ppqg_table : path, normgrid.NormInterpolator, or None
        Precomputed normalization table for that band (built by
        python/normgrid.py for exactly this region), or a name registered in
        python/table_registry.json.  If given, ppqn_integral/ppqg_integral
        interpolate it instead of running the quadrature.
    prior_bounds : dict or None
        The MCMC prior's hard bounds, {param: (lo, hi)} for k (NR), F0, V,
        p0, p10, q0, q10 (Z/eps may be given as their fixed values).  Keys
        shared by both bands go at the top level; a band's own parameters
        (e.g. its Fano factor) go under "NR"/"ER" and override them:
            {"V": (2.7, 3.3), ..., "NR": {"k": (0.13, 0.22), "F0": (1e-5, 1)},
                                   "ER": {"F0": (0.1, 0.35)}}
        Checked against every table given, assuming the prior also enforces
        p10 >= p0 and q10 >= q0; raises normgrid.PriorNotCoveredError if it
        reaches outside one.
    n_workers : int or None
        Threads used for large batched PpqN_vector/PpqG_vector calls;
        see make_ppqn_pdf.
    """

    def __init__(self, ep_min, ep_max, eq_min, eq_max, ep_data, eq_data, *,
                 norm_epsrel=None, norm_epsabs=None,
                 ppqn_table=None, ppqg_table=None, prior_bounds=None,
                 n_workers=None):
        self.ep_min, self.ep_max = ep_min, ep_max
        self.eq_min, self.eq_max = eq_min, eq_max
        self.ep_data = np.ascontiguousarray(np.asarray(ep_data, dtype=np.float64))
        self.eq_data = np.ascontiguousarray(np.asarray(eq_data, dtype=np.float64))
        self.norm_epsrel = norm_epsrel
        self.norm_epsabs = norm_epsabs
        self.n_workers = n_workers
        self.ppqn_table = self._load_table(ppqn_table, "NR")
        self.ppqg_table = self._load_table(ppqg_table, "ER")
        if prior_bounds is not None:
            if not self._tables():
                raise ValueError("prior_bounds is checked against the tables; no table was given")
            for t in self._tables():
                t.check_prior(self._band_view(prior_bounds, t.band))

    def _tables(self):
        return [t for t in (self.ppqn_table, self.ppqg_table) if t is not None]

    @staticmethod
    def _band_view(d, band):
        """The shared keys of d, overridden by d[band]; k only for NR."""
        v = {p: x for p, x in d.items() if p not in ("NR", "ER")}
        v.update(d.get(band, {}))
        if band == "ER":
            v.pop("k", None)
        return v

    def check_points(self, points):
        """Raise normgrid.PriorNotCoveredError unless every point is inside
        every table: e.g. the starting walker positions, {param: array} laid
        out like prior_bounds (band-specific arrays under "NR"/"ER")."""
        for t in self._tables():
            t.check_points(self._band_view(points, t.band))

    def table_bounds(self):
        """{band: {param: (lo, hi)}}: a prior box each given table covers,
        for a prior that enforces p10 >= p0 and q10 >= q0.  A parameter the
        bands share (V, p0, p10, q0, q10) must lie in both."""
        if not self._tables():
            raise ValueError("no table was given")
        return {t.band: t.table_bounds() for t in self._tables()}

    def _load_table(self, table, band):
        if table is None:
            return None
        import normgrid
        if not isinstance(table, normgrid.NormInterpolator):
            if not os.path.exists(str(table)):
                import normtables
                table = normtables.fetch(table)
            table = normgrid.NormInterpolator.from_hdf5(table)
        if table.band != band:
            raise ValueError(f"table is for the {table.band} band, expected {band}")
        region = (self.ep_min, self.ep_max, self.eq_min, self.eq_max)
        if not np.allclose(table.region, region, rtol=0, atol=1e-12):
            raise ValueError(f"table was built for region {table.region}, not {region}")
        return table

    def _quadrature_tolerances(self):
        if self.norm_epsrel is None or self.norm_epsabs is None:
            raise ValueError("norm_epsrel and norm_epsabs are required to compute a normalization "
                             "integral without a precomputed table")
        return dict(epsrel=self.norm_epsrel, epsabs=self.norm_epsabs)

    # ---- NR (PpqN) band ----

    def ppqn_integral(self, *, k, Z, F0, eps, V, p0, p10, q0, q10):
        """Normalization: integral of PpqN over this region."""
        if self.ppqn_table is not None:
            return self.ppqn_table(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
        return ppqn_region(self.ep_min, self.ep_max, self.eq_min, self.eq_max,
                            **self._quadrature_tolerances(),
                            k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    def ppqn_values(self, *, k, Z, F0, eps, V, p0, p10, q0, q10):
        """Un-normalized PpqN at the bound dataset."""
        pdf_func = make_ppqn_pdf(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10,
                                  q0=q0, q10=q10, n_workers=self.n_workers)
        return pdf_func(self.ep_data, self.eq_data)

    def ppqn_normalized_values(self, *, k, Z, F0, eps, V, p0, p10, q0, q10):
        """ppqn_values(...) / ppqn_integral(...), elementwise."""
        return (self.ppqn_values(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
                / self.ppqn_integral(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10))

    # ---- ER (PpqG) band ----

    def ppqg_integral(self, *, F0, eps, V, p0, p10, q0, q10):
        """Normalization: integral of PpqG over this region."""
        if self.ppqg_table is not None:
            return self.ppqg_table(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
        return ppqg_region(self.ep_min, self.ep_max, self.eq_min, self.eq_max,
                            **self._quadrature_tolerances(),
                            F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    def ppqg_values(self, *, F0, eps, V, p0, p10, q0, q10):
        """Un-normalized PpqG at the bound dataset."""
        pdf_func = make_ppqg_pdf(F0=F0, eps=eps, V=V, p0=p0, p10=p10,
                                  q0=q0, q10=q10, n_workers=self.n_workers)
        return pdf_func(self.ep_data, self.eq_data)

    def ppqg_normalized_values(self, *, F0, eps, V, p0, p10, q0, q10):
        """ppqg_values(...) / ppqg_integral(...), elementwise."""
        return (self.ppqg_values(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
                / self.ppqg_integral(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10))
