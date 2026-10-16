"""
event_generator is the public entrypoint for generating simulated (Ep, Eq)
detector events -- an independent physics simulator, not a second way to
evaluate the PDF (see "Physical picture" below for why that independence
matters). If you want normalized PDF values or a likelihood instead, use
PpqPDF (python/ppq_pdf.py).

Unlike PpqPDF, this module has no Fortran dependency at all -- it is pure
Python/numpy built on python/internal/ppq_dist.py's Y/Nbar/sigp/sigq, so it
works in any environment that can `pip install` this package, compiled
library or not.

Quickstart
----------
    from event_generator import generate_events
    events = generate_events(
        target_size=20_000, ER_frac=0.6, seed_value=8,
        ep_min=2.5, ep_max=350.0, eq_min=0.75, eq_max=200.0,
        k=0.18, Z=32.0, F0_ER=0.2, F0_NR=0.122, eps=3.0e-3, V=3.0,
        p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
        PNa=0.53693208, PNb=6.41515782, PNd=23.71789286,
        PGa=0.573211975, PGb=0.169520023, PGd=279.552394,
    )
    Ep, Eq = events["EP"].to_numpy(), events["EQ"].to_numpy()

All physics parameters are required keyword arguments -- there are no
defaults, matching the rest of this library (PpqPDF, _ppqfort_bindings.py,
ppq_dist.py). ER_frac=1.0 generates ER-only, ER_frac=0.0 generates NR-only,
and anything in between generates a mix.

Physical picture
-----------------
Each event starts from a true recoil energy Er, drawn from the species'
biexponential spectrum (PErG for electron recoils, PErN for nuclear
recoils). Er then produces a Fano-broadened number of e/h pairs N, which
in turn sets the phonon energy Ep (recoil + Neganov-Luke) and charge
energy Eq (N * eps), each further smeared by its own detector resolution:

    Er ~ PErN or PErG        (biexponential true-recoil-energy spectrum)
    N  | Er ~ TruncNormal     (Fano-factor fluctuation, truncated at N >= 0)
    Ep | Er, N ~ Normal       (phonon detector: Neganov-Luke + thermal)
    Eq | N ~ Normal           (charge detector)

The analytic PDFs PpqN/PpqG are obtained by marginalizing this same chain
analytically over N and numerically over Er -- so agreement between events
generated here and PpqN/PpqG (see test/python/test_chisquare_{nr,er}_simulator.py)
is a real physics check, not a tautology: this module never touches PpqN/PpqG.

Returned DataFrame
-------------------
generate_events() returns columns EP, EQ, N0, is_ER, seed_value, is_bkg
(is_bkg is always False here -- it exists only so these rows line up with
a separate zero-charge/surface-background generator, e.g. the KDE-based
one in nrFanoII_paper2022's ba_bknd_lines.py, which can be concatenated
directly as long as it produces the same columns).

Reproducibility
----------------
Every retry batch (see _species_in_roi below) needs its own independent
random stream, but "independent" from a single seed_value has to be done
carefully. np.random.SeedSequence.spawn() deterministically derives any
number of child streams from one seed via a proper hash-based mix,
guaranteeing (for practical purposes) no correlation between them or with
streams spawned from a different seed_value.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

import os
import sys

_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
_INTERNAL_DIR = os.path.join(_PYTHON_DIR, "internal")
sys.path.insert(0, _INTERNAL_DIR)
import ppq_dist as ppq


def _sample_biexponential(n, rng, *, mix_frac, scale_fast, scale_slow):
    """n draws from a two-component exponential mixture -- the shared shape
    of both PErN (nuclear recoils) and PErG (electron recoils)."""
    is_fast = rng.random(n) < mix_frac
    return np.where(is_fast, rng.exponential(scale_fast, n), rng.exponential(scale_slow, n))


def _sample_N(Nbar_arr, F_arr, rng):
    """
    Sample N (number of e/h pairs) for each event from a half-normal
    truncated at 0. Mean = Nbar, Var = Nbar * F.

    Uses scipy.stats.truncnorm vectorised over events.
    """
    sigma_N = np.sqrt(np.maximum(Nbar_arr * F_arr, 0.0))

    # scalar path: avoid divide-by-zero when sigma is zero
    N = np.empty_like(Nbar_arr)
    zero_mask = sigma_N == 0.0
    N[zero_mask] = Nbar_arr[zero_mask]

    nonzero = ~zero_mask
    if nonzero.any():
        a_trunc = -Nbar_arr[nonzero] / sigma_N[nonzero]   # lower clip in std units
        b_trunc = np.inf
        N[nonzero] = truncnorm.rvs(a_trunc, b_trunc,
                                   loc=Nbar_arr[nonzero],
                                   scale=sigma_N[nonzero],
                                   random_state=rng)
    return N


def _generate_species_events(n_events, *, Nbar_func,
                              F0, eps, V, p0, p10, q0, q10,
                              mix_frac, scale_fast, scale_slow,
                              seed):
    """
    Shared generator: draws Er from the biexponential mix_frac/scale_fast/
    scale_slow spectrum and mean e/h pair count Nbar_func(Er), then
    simulates the rest of the detector chain. NR/ER differ only in
    Nbar_func and which spectrum parameters they pass (see
    generate_NR_events/generate_ER_events).
    """
    rng = np.random.default_rng(seed)

    Er = _sample_biexponential(n_events, rng, mix_frac=mix_frac,
                                scale_fast=scale_fast, scale_slow=scale_slow)

    Nbar = Nbar_func(Er)                              # mean e/h pairs
    F_val = np.full_like(Er, F0)                      # Fano factor (constant)

    N = _sample_N(Nbar, F_val, rng)

    Ep_true = Er + N * (V / 1e3)                      # Neganov-Luke + recoil
    Eq_true = N * eps                                 # ionization

    # Detector smearing: evaluate resolution at the true (noise-free) signal.
    # sigp and sigq vary slowly, so this closely approximates the heteroscedastic
    # PDF where resolutions are evaluated at the measured values.
    sigp_val = ppq.sigp(Ep_true, eps=eps, V=V, p0=p0, p10=p10)
    sigq_val = ppq.sigq(Eq_true, q0=q0, q10=q10)

    Ep = rng.normal(Ep_true, sigp_val)
    Eq = rng.normal(Eq_true, sigq_val)

    return Ep, Eq, Er, N


def generate_NR_events(n_events, *,
                        k, Z,
                        F0,
                        eps, V,
                        p0, p10,
                        q0, q10,
                        PNa, PNb, PNd,
                        seed):
    """
    Generate (Ep, Eq) pairs distributed according to PpqN -- no ROI
    cropping, no species mixing; see generate_events() for that.

    Returns
    -------
    Ep, Eq : ndarray, shape (n_events,)
    Er : ndarray, shape (n_events,)   -- latent true recoil energies
    N  : ndarray, shape (n_events,)   -- latent e/h pair counts
    """
    return _generate_species_events(
        n_events, Nbar_func=lambda Er: ppq.Y(Er, k=k, Z=Z) * Er / eps,
        F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10,
        mix_frac=PNa, scale_fast=PNb, scale_slow=PNd, seed=seed)


def generate_ER_events(n_events, *,
                        F0,
                        eps, V,
                        p0, p10,
                        q0, q10,
                        PGa, PGb, PGd,
                        seed):
    """
    Generate (Ep, Eq) pairs distributed according to PpqG -- no ROI
    cropping, no species mixing; see generate_events() for that.

    Gamma / electron-recoil events have ionization yield Y = 1.

    Returns
    -------
    Ep, Eq : ndarray, shape (n_events,)
    Er : ndarray, shape (n_events,)
    N  : ndarray, shape (n_events,)
    """
    return _generate_species_events(
        n_events, Nbar_func=lambda Er: Er / eps,
        F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10,
        mix_frac=PGa, scale_fast=PGb, scale_slow=PGd, seed=seed)


# Initial guess for each species' ROI acceptance rate, refined from batch 1
# onward using the actually-observed rate. Only affects how generously the
# FIRST retry batch is sized -- a bad guess costs one extra retry, not
# correctness. ER's band is intrinsically narrower in a typical ROI than
# NR's, hence the lower starting guess.
_DEFAULT_PASS_RATE = {"ER": 0.08, "NR": 0.20}


def _species_in_roi(n_target, species, seed_sequence, *,
                     ep_min, ep_max, eq_min, eq_max, max_retries,
                     sampler, sampler_kwargs):
    """
    Generate exactly n_target ROI-passing (Ep, Eq, Er, N) events of one
    species, by calling sampler(n, seed=..., **sampler_kwargs) in retry
    batches (generate_NR_events or generate_ER_events).

    Each retry batch draws a fresh set of events (sized from a running
    estimate of the ROI acceptance rate) using its own independent seed
    spawned from seed_sequence, so retries never reuse a random stream.
    """
    if n_target == 0:
        empty = np.array([])
        return empty, empty, empty, empty

    child_seeds = iter(seed_sequence.spawn(max_retries))

    ep_parts, eq_parts, er_parts, n_parts = [], [], [], []
    n_have = 0
    pass_rate = _DEFAULT_PASS_RATE[species]

    for _ in range(max_retries):
        if n_have >= n_target:
            break
        seed = int(next(child_seeds).generate_state(1)[0])

        remaining = n_target - n_have
        want = int(remaining / pass_rate * 1.3) + 2000

        Ep, Eq, Er, N = sampler(want, seed=seed, **sampler_kwargs)

        mask = (Ep >= ep_min) & (Ep <= ep_max) & (Eq >= eq_min) & (Eq <= eq_max)
        passed = int(mask.sum())
        pass_rate = max(passed / want, 0.01)

        keep = min(passed, remaining)
        idx = np.flatnonzero(mask)[:keep]
        ep_parts.append(Ep[idx])
        eq_parts.append(Eq[idx])
        er_parts.append(Er[idx])
        n_parts.append(N[idx])
        n_have += keep

    if n_have < n_target:
        warnings.warn(f"{species}: only generated {n_have}/{n_target} events "
                      f"within {max_retries} retry batches -- raise max_retries.")

    return (np.concatenate(ep_parts), np.concatenate(eq_parts),
            np.concatenate(er_parts), np.concatenate(n_parts))


def generate_events(target_size, ER_frac, seed_value, *,
                     ep_min, ep_max, eq_min, eq_max,
                     k, Z, F0_ER, F0_NR, eps, V, p0, p10, q0, q10,
                     PNa, PNb, PNd, PGa, PGb, PGd,
                     max_retries=40):
    """
    Generate target_size (Ep, Eq) events, ER_frac of them (in expectation)
    electron recoils and the rest nuclear recoils, cropped to [ep_min,
    ep_max] x [eq_min, eq_max].

    ER_frac=1.0 / 0.0 generate pure ER / pure NR samples.

    Parameters
    ----------
    target_size : int
        Total number of ROI-passing events to generate.
    ER_frac : float
        Probability that a given event is an electron recoil; the actual
        ER count is Binomial(target_size, ER_frac), not a fixed split.
        Var(Binomial(n, p)) = n*p*(1-p) is exactly zero at p=0 or p=1, so
        ER_frac=1.0 / 0.0 are still fully deterministic (ER count = exactly
        target_size / 0, every seed) -- no separate ER_only/NR_only flag
        is needed to get a pure sample with zero chance of a stray event
        from the other species.
    seed_value : int
        Seed for the whole call. Reproducible: the same seed_value always
        produces the same events, independent of what other seed_values
        are used elsewhere in the same run.
    ep_min, ep_max, eq_min, eq_max : float
        ROI bounds, keV -- required, no default: this must be your real
        analysis region, not an arbitrary wide range.
    k, Z : float
        Lindhard ionization yield calibration constant and target atomic
        number (nuclear recoils only; unused but still required when
        ER_frac=1.0, for the same reason PpqPDF always requires them).
    F0_ER, F0_NR : float
        Fano factor (constant), one per species -- ER and NR generally
        have different Fano factors even at the same operating point.
    eps, V, p0, p10, q0, q10 : float
        Shared detector parameters (energy per e/h pair, bias voltage,
        phonon/charge resolution parameters) -- see ppq_dist.py's
        sigp/sigq.
    PNa, PNb, PNd : float
        NR true-recoil-energy (PErN) biexponential spectrum parameters.
    PGa, PGb, PGd : float
        ER true-recoil-energy (PErG) biexponential spectrum parameters.
    max_retries : int
        Cap on retry batches per species before giving up with a warning.

    Returns
    -------
    pandas.DataFrame with columns EP, EQ, N0, is_ER, seed_value, is_bkg --
    shuffled, so downstream code doesn't need to worry about ER/NR
    ordering. is_bkg is always False; see the module docstring.
    """
    if not 0.0 <= ER_frac <= 1.0:
        raise ValueError(f"ER_frac must be in [0, 1], got {ER_frac}")

    ss = np.random.SeedSequence(seed_value)
    split_seed, er_seed, nr_seed, shuffle_seed = ss.spawn(4)

    # deterministic (no fluctuation) at ER_frac=0.0/1.0 -- see docstring
    n_ER = int(np.random.default_rng(split_seed).binomial(target_size, ER_frac))
    n_NR = target_size - n_ER

    roi = dict(ep_min=ep_min, ep_max=ep_max, eq_min=eq_min, eq_max=eq_max)

    Ep_er, Eq_er, _, N_er = _species_in_roi(
        n_ER, "ER", er_seed, max_retries=max_retries, **roi,
        sampler=generate_ER_events,
        sampler_kwargs=dict(F0=F0_ER, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10,
                            PGa=PGa, PGb=PGb, PGd=PGd))
    Ep_nr, Eq_nr, _, N_nr = _species_in_roi(
        n_NR, "NR", nr_seed, max_retries=max_retries, **roi,
        sampler=generate_NR_events,
        sampler_kwargs=dict(k=k, Z=Z, F0=F0_NR, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10,
                            PNa=PNa, PNb=PNb, PNd=PNd))

    Ep = np.concatenate([Ep_er, Ep_nr])
    Eq = np.concatenate([Eq_er, Eq_nr])
    N0 = np.concatenate([N_er, N_nr])
    is_ER = np.concatenate([np.ones(len(Ep_er), dtype=bool), np.zeros(len(Ep_nr), dtype=bool)])

    order = np.random.default_rng(shuffle_seed).permutation(len(Ep))
    df = pd.DataFrame({
        "EP": Ep[order], "EQ": Eq[order], "N0": N0[order], "is_ER": is_ER[order],
    })
    df["seed_value"] = seed_value
    df["is_bkg"] = False

    n_ok = len(df)
    print(f"Generated {n_ok}/{target_size} events "
          f"(ER {int(df['is_ER'].sum())}, NR {n_ok - int(df['is_ER'].sum())}; "
          f"target ER count was {n_ER} of {target_size} via Binomial(ER_frac={ER_frac})); "
          f"EP:[{ep_min},{ep_max}], EQ:[{eq_min},{eq_max}] keV")

    return df
