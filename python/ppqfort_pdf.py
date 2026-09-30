"""
Python wrapper exposing the Fortran PpqN / PpqG PDFs as vectorized
pdf_func(Ep_flat, Eq_flat) -> values_flat callables.

All physics parameters are required keyword arguments — there are no
defaults, so the caller must specify the model fully.

The shared library evaluates points sequentially (~3-25 ms per point,
scaling with max(Ep, Eq) because the internal Er integration range grows
with energy).  Since ctypes releases the GIL during foreign calls, this
wrapper parallelizes large requests by splitting them into chunks and
dispatching PpqN_vector calls across a thread pool.  Points are
interleaved across chunks after sorting by max(Ep, Eq) so each chunk
carries a similar share of expensive (high-energy) points.
"""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_DIR = os.path.join(_REPO_ROOT, "lib")

_DOUBLE_ARR = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")

_api = None


def _load_library():
    """Load libband_distribution.so.

    The library has no SONAME and its RUNPATH points at nix store paths,
    so its dependencies (libassert.so, libjulienne.so) can only be found
    via LD_LIBRARY_PATH, which must be set before the Python process
    starts:  LD_LIBRARY_PATH=lib python your_script.py
    """
    global _api
    if _api is not None:
        return _api

    try:
        _api = ctypes.CDLL(os.path.join(_LIB_DIR, "libband_distribution.so"))
    except OSError as e:
        raise RuntimeError(
            f"Could not load libband_distribution.so ({e}). "
            f"Run with LD_LIBRARY_PATH={_LIB_DIR} set before Python starts, "
            f"e.g.: LD_LIBRARY_PATH=lib python your_script.py"
        ) from e

    _api.PpqN_vector.argtypes = [
        _DOUBLE_ARR, _DOUBLE_ARR, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        _DOUBLE_ARR,
    ]
    _api.PpqN_vector.restype = None

    _api.PpqG_vector.argtypes = [
        _DOUBLE_ARR, _DOUBLE_ARR, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        _DOUBLE_ARR,
    ]
    _api.PpqG_vector.restype = None

    _api.PpqFort_version.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    _api.PpqFort_version.restype = None

    _api.PpqN_region.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # ep_min, ep_max, eq_min, eq_max
        ctypes.c_double, ctypes.c_double,                                    # epsrel, epsabs
        ctypes.c_double, ctypes.c_double, ctypes.c_double,                   # k, Z, F0
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # eps, V, p0, p10
        ctypes.c_double, ctypes.c_double,                                    # q0, q10
    ]
    _api.PpqN_region.restype = ctypes.c_double

    _api.PpqG_region.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # ep_min, ep_max, eq_min, eq_max
        ctypes.c_double, ctypes.c_double,                                    # epsrel, epsabs
        ctypes.c_double,                                                     # F0
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # eps, V, p0, p10
        ctypes.c_double, ctypes.c_double,                                    # q0, q10
    ]
    _api.PpqG_region.restype = ctypes.c_double

    return _api


def version():
    """Return (major, minor, patch) reported by the loaded shared library."""
    api = _load_library()
    major, minor, patch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    api.PpqFort_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
    return (major.value, minor.value, patch.value)


def make_ppqn_pdf(*, k, Z, F0, eps, V, p0, p10, q0, q10, n_workers=None):
    """
    Build pdf_func(Ep_flat, Eq_flat) -> values_flat backed by the Fortran
    PpqN (nuclear recoil band PDF).

    All physics parameters are required:
      k, Z           : Lindhard ionization yield calibration constant and
                        target atomic number
      F0             : Fano factor (constant)
      eps            : energy per e/h pair [keV]
      V              : bias voltage [V]
      p0, p10        : phonon resolution parameters
      q0, q10        : charge resolution parameters

    n_workers : int or None
        Threads used for large requests.  Defaults to os.cpu_count().
        Requests of <= 16 points bypass the pool (e.g. scalar calls from
        scipy quadrature).
    """
    api = _load_library()
    if n_workers is None:
        n_workers = os.cpu_count()
    scalars = (k, Z, F0, eps, V, p0, p10, q0, q10)

    def _eval_chunk(ep, eq, out):
        api.PpqN_vector(ep, eq, ep.size, *scalars, out)

    return _make_threaded_pdf(_eval_chunk, n_workers)


def make_ppqg_pdf(*, F0, eps, V, p0, p10, q0, q10, n_workers=None):
    """Same as make_ppqn_pdf but for the Fortran PpqG (gamma / ER band).
    The yield is fixed at Y = 1 internally, so k and Z are not taken."""
    api = _load_library()
    if n_workers is None:
        n_workers = os.cpu_count()
    scalars = (F0, eps, V, p0, p10, q0, q10)

    def _eval_chunk(ep, eq, out):
        api.PpqG_vector(ep, eq, ep.size, *scalars, out)

    return _make_threaded_pdf(_eval_chunk, n_workers)


def ppqn_region(ep_min, ep_max, eq_min, eq_max, *,
                 epsrel, epsabs,
                 k, Z, F0, eps, V, p0, p10, q0, q10):
    """
    Integral of PpqN over [ep_min,ep_max] x [eq_min,eq_max], computed
    entirely in Fortran (a single call, no per-point ctypes round trips)
    with a doubling-verified nested Gauss-Legendre quadrature (see
    PpqFort_s.f90's region_integral_gl) -- the ridge location/width is
    already known exactly (ridge_eq_and_width), so this needs far fewer
    points than a naive grid or generic adaptive cubature for the same
    accuracy, and is much faster than nested scipy.integrate.quad.

    epsrel/epsabs are required (no defaults): they set how tightly two
    successive doubled quadrature orders must agree before the result is
    trusted, and directly control the accuracy of a number that feeds a
    likelihood normalization. The Fortran side error-stops rather than
    returning an unverified number if that isn't reached by its highest
    order.
    """
    api = _load_library()
    return api.PpqN_region(ep_min, ep_max, eq_min, eq_max,
                            epsrel, epsabs,
                            k, Z, F0, eps, V, p0, p10, q0, q10)


def ppqg_region(ep_min, ep_max, eq_min, eq_max, *,
                 epsrel, epsabs,
                 F0, eps, V, p0, p10, q0, q10):
    """Same as ppqn_region but for PpqG (electron-recoil band, Y=1)."""
    api = _load_library()
    return api.PpqG_region(ep_min, ep_max, eq_min, eq_max,
                            epsrel, epsabs,
                            F0, eps, V, p0, p10, q0, q10)


def _make_threaded_pdf(eval_chunk, n_workers):
    def pdf(ep_flat, eq_flat):
        ep = np.ascontiguousarray(np.asarray(ep_flat, dtype=np.float64).ravel())
        eq = np.ascontiguousarray(np.asarray(eq_flat, dtype=np.float64).ravel())
        n = ep.size
        out = np.empty(n, dtype=np.float64)

        if n <= 16 or n_workers <= 1:
            eval_chunk(ep, eq, out)
        else:
            # Cost per point scales with max(Ep, Eq): sort by cost and deal
            # points round-robin so every chunk gets a balanced mix.
            order = np.argsort(-np.maximum(ep, eq), kind="stable")
            n_chunks = min(n_workers * 4, n)
            chunks = [order[i::n_chunks] for i in range(n_chunks)]

            def run(idx):
                ep_c = np.ascontiguousarray(ep[idx])
                eq_c = np.ascontiguousarray(eq[idx])
                out_c = np.empty(idx.size, dtype=np.float64)
                eval_chunk(ep_c, eq_c, out_c)
                out[idx] = out_c

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(run, chunks))

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return pdf
