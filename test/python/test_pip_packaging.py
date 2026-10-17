"""
Validates the pip-installable package's graceful-degradation promise:
pure-Python functionality (event_generator.py) works with no compiled
Fortran library anywhere on the system, while PpqPDF -- which always
needs it, even with a precomputed normalization table -- fails clearly
and immediately at construction rather than deep into a fit.

Must run against a `pip install -e .[test]` of this repository (imports
band_distribution as an installed package, not via sys.path hacks like
the rest of this test suite), and before the Fortran library is ever
built/installed to lib/ -- otherwise the "no Fortran available" half of
this test exercises nothing.

Run from the repository root:  python test/python/test_pip_packaging.py
"""

import sys

failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


import band_distribution as bd  # noqa: E402 -- the import itself is part of what's tested

check("import band_distribution with no compiled library present", True)
check("PpqPDF is exported", hasattr(bd, "PpqPDF"))
check("generate_events is exported", hasattr(bd, "generate_events"))
check("generate_NR_events is exported", hasattr(bd, "generate_NR_events"))
check("generate_ER_events is exported", hasattr(bd, "generate_ER_events"))

PARAMS = dict(
    k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
)
SPECTRUM = dict(PNa=0.53693208, PNb=6.41515782, PNd=23.71789286)

try:
    Ep, Eq, N0, seeds = bd.generate_NR_events(50, seed=1, **PARAMS, **SPECTRUM)
    check("generate_NR_events runs with no Fortran library",
          Ep.shape == Eq.shape == N0.shape == seeds.shape == (50,))
except Exception as e:
    check("generate_NR_events runs with no Fortran library", False, repr(e))

try:
    bd.PpqPDF(0, 1, 0, 1, [], [], norm_epsrel=0.1, norm_epsabs=0.1)
    check("PpqPDF() raises at construction without the Fortran library", False,
          "construction succeeded -- it should have failed fast instead")
except RuntimeError as e:
    check("PpqPDF() raises at construction without the Fortran library", True, str(e)[:80])
except Exception as e:
    check("PpqPDF() raises at construction without the Fortran library", False,
          f"wrong exception type: {e!r}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
