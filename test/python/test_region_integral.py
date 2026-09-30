"""
Validates the rectangular-region normalization integral
(ppqfort_pdf.ppqn_region/ppqg_region -> Fortran PpqN_region/PpqG_region,
a doubling-verified nested Gauss-Legendre quadrature) against the
existing, algorithmically independent nested-quad reference
(chisquare_harness.debug_bin, treating the whole region as one "bin",
via band_breakpoints.make_ridge_breakpoints for the required ridge
breakpoints) -- ground truth -- plus timing, across a few representative
regions including the user's actual MCMC target region, for both the NR
and ER bands.

Also checks that the doubling loop's error-stop safety net actually
fires on a synthetically unconvergeable request rather than silently
returning a wrong answer.

Run from the repository root with:
  LD_LIBRARY_PATH=lib python test/python/test_region_integral.py
"""

import io
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppqfort_pdf import make_ppqg_pdf, make_ppqn_pdf, ppqg_region, ppqn_region
from band_breakpoints import make_ridge_breakpoints
from chisquare_harness import debug_bin

PARAMS_NR = dict(k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
                  p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151)
PARAMS_ER = dict(F0=0.122, eps=3.0e-3, V=3.0,
                  p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151)

TOL = dict(epsrel=1e-4, epsabs=1e-10)

# (ep_min, ep_max, eq_min, eq_max, label) -- the user's actual MCMC target
# region (the case that surfaced a since-removed Cuba/Cuhre implementation's
# unreliable convergence flag, so it stays in this sweep), plus a few other
# representative regions: a wider fit window, a narrow low-energy slice,
# and one intentionally off-band to confirm it correctly integrates to ~0.
REGIONS = [
    (2.0, 200.0, 4.0, 100.0, "MCMC target region"),
    (3.0, 250.0, 0.5, 130.0, "typical fit region"),
    (3.0, 20.0, 0.5, 10.0, "low-energy slice"),
    (200.0, 250.0, 0.5, 5.0, "off-band (high Ep, tiny Eq)"),
]


def quad_reference(band, ep_min, ep_max, eq_min, eq_max, params):
    """The user's original method: nested scipy.integrate.quad with ridge
    breakpoints (chisquare_harness.debug_bin's "fixed" value -- the same
    computation expected_counts_from_pdf does, but unnormalized so it's
    the raw integral).  Also cross-checks against debug_bin's independent
    brute-force Simpson grid_ref."""
    if band == "NR":
        pdf_func = make_ppqn_pdf(**params, n_workers=os.cpu_count())
        ridge = make_ridge_breakpoints("NR", **{k: params[k] for k in
                                                 ("k", "Z", "eps", "V", "p0", "p10", "q0", "q10")},
                                        er_max=700.0, n_window_widths=10.0)
    else:
        pdf_func = make_ppqg_pdf(**params, n_workers=os.cpu_count())
        ridge = make_ridge_breakpoints("ER", k=None, Z=None,
                                        **{k: params[k] for k in
                                           ("eps", "V", "p0", "p10", "q0", "q10")},
                                        er_max=700.0, n_window_widths=10.0)
    with redirect_stdout(io.StringIO()):
        result = debug_bin(pdf_func, (ep_min, ep_max, eq_min, eq_max), ridge)
    rel_fixed_grid = abs(result["fixed"] - result["grid_ref"]) / max(abs(result["grid_ref"]), 1e-300)
    if rel_fixed_grid > 1e-2:
        print(f"    NOTE: debug_bin's own fixed vs grid_ref disagree by {rel_fixed_grid:.3e}")
    return float(result["fixed"])


def run(band, params, region_func):
    print(f"\n{'='*70}")
    print(f"Band: {band}")
    print(f"{'='*70}")
    all_ok = True
    for ep_min, ep_max, eq_min, eq_max, label in REGIONS:
        t0 = time.time()
        ref = quad_reference(band, ep_min, ep_max, eq_min, eq_max, params)
        t_quad = time.time() - t0

        t0 = time.time()
        val = region_func(ep_min, ep_max, eq_min, eq_max, **TOL, **params)
        t_region = time.time() - t0

        rel = abs(val - ref) / max(abs(ref), 1e-300)

        print(f"\n  {label}: Ep in [{ep_min},{ep_max}], Eq in [{eq_min},{eq_max}]")
        print(f"    quad reference : {ref:.10e}   ({t_quad*1000:.1f} ms)")
        print(f"    region (GL)    : {val:.10e}   ({t_region*1000:.1f} ms)  "
              f"rel vs quad: {rel:.3e}  speedup vs quad: "
              f"{t_quad/t_region if t_region > 0 else float('inf'):.1f}x")

        ok = rel < 5e-4
        print(f"    {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def run_nonconvergence_check():
    """Confirms the doubling loop's error stop fires (fails loudly)
    rather than silently returning an unverified number when order 256
    still can't meet an impossibly tight request -- the safety net this
    whole design exists for must itself be proven to work, not assumed
    to.  error stop terminates the process, so this runs in a subprocess
    and checks the exit code and message rather than catching an
    exception in-process."""
    print(f"\n{'='*70}")
    print("Safety net: error stop fires on a non-convergent request")
    print(f"{'='*70}")
    script = (
        "import sys; sys.path.insert(0, 'python'); "
        "from ppqfort_pdf import ppqn_region; "
        "ppqn_region(2.0, 200.0, 4.0, 100.0, epsrel=1e-300, epsabs=0.0, "
        "k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0, "
        "p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151)"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                           capture_output=True, text=True,
                           env={**os.environ, "LD_LIBRARY_PATH": os.path.join(REPO_ROOT, "lib")})
    fired = proc.returncode != 0 and "did not converge" in proc.stderr
    print(f"    exit code: {proc.returncode}")
    print(f"    stderr tail: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else '(empty)'}")
    print(f"    {'PASS' if fired else 'FAIL'} -- error stop {'fired' if fired else 'did NOT fire'} as expected")
    return fired


def run_invalid_input_check():
    """Confirms invalid parameter sets error out immediately (seconds, not
    minutes) with a clear message.  Before the guard existed, p10 < p0
    (negative sigp^2 at high energy -> NaNs) ground for 15+ minutes per
    call -- fatal inside an MCMC, where a bad proposal must just be
    rejected quickly."""
    print(f"\n{'='*70}")
    print("Guard: invalid parameters error out immediately")
    print(f"{'='*70}")
    base = dict(k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
                p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151)
    cases = [
        ("p10 < p0",   dict(p0=0.35, p10=0.30), "p10 < p0"),
        ("q10 < q0",   dict(q0=0.30, q10=0.25), "q10 < q0"),
        ("F0 = 0",     dict(F0=0.0), "F0 must be > 0"),
        ("k <= 0",     dict(k=-0.1), "k must be > 0"),
        ("NaN in p0",  dict(p0=float("nan")), "p0 and q0 must be > 0"),
    ]
    all_ok = True
    for label, override, expect in cases:
        params = dict(base, **override)
        script = ("import sys; sys.path.insert(0, 'python'); nan = float('nan'); "
                  "from ppqfort_pdf import ppqn_region; "
                  f"ppqn_region(2.0, 200.0, 4.0, 100.0, epsrel=1e-4, epsabs=1e-10, **{params!r})")
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=60,
                                  env={**os.environ, "LD_LIBRARY_PATH": os.path.join(REPO_ROOT, "lib")})
            dt = time.time() - t0
            ok = proc.returncode != 0 and expect in proc.stderr and dt < 30
            print(f"    {label:10s}: exit {proc.returncode} in {dt:.2f} s  {'PASS' if ok else 'FAIL'}"
                  f"{'' if ok else '  stderr: ' + proc.stderr.strip()[-200:]}")
        except subprocess.TimeoutExpired:
            ok = False
            print(f"    {label:10s}: TIMED OUT (>60 s) FAIL -- guard did not fire")
        all_ok = all_ok and ok
    return all_ok


if __name__ == "__main__":
    ok_nr = run("NR", PARAMS_NR, ppqn_region)
    ok_er = run("ER", PARAMS_ER, ppqg_region)
    ok_safety_net = run_nonconvergence_check()
    ok_guard = run_invalid_input_check()

    print(f"\n{'='*70}")
    if ok_nr and ok_er and ok_safety_net and ok_guard:
        print("ALL PASS")
    else:
        print("FAILURES ABOVE")
        sys.exit(1)
