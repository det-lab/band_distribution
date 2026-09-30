"""
Generate (Ep, Eq) samples from the PpqN / PpqG probability densities using
the hierarchical generative model that underlies the analytic PDF.

The analytic PDF PpqN(Ep, Eq) is obtained by:
  1. Marginalizing N analytically (→ the erf factor in PpqFullN)
  2. Integrating over Er numerically

Generating samples instead follows the latent-variable chain:
  Er ~ PErN               (biexponential neutron recoil spectrum)
  N  | Er ~ TruncNormal   (Fano-factor fluctuation, truncated at N >= 0)
  Ep | Er,N ~ Normal      (phonon detector: Neganov-Luke + thermal)
  Eq | N    ~ Normal      (charge detector)

For gamma (ER) events, substitute PErG and Y=1.
"""

import numpy as np
from scipy.stats import truncnorm
import sys, os

# Y, sigp, sigq come from the validated python reference implementation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import pq_dist_v10 as ppq


def _sample_PErN(n, rng, PNa, PNb, PNd):
    """Sample Er from the biexponential neutron spectrum PErN."""
    component = rng.uniform(size=n) < PNa
    return np.where(component,
                    rng.exponential(PNb, size=n),
                    rng.exponential(PNd, size=n))


def _sample_PErG(n, rng, PGa, PGb, PGd):
    """Sample Er from the biexponential gamma spectrum PErG."""
    component = rng.uniform(size=n) < PGa
    return np.where(component,
                    rng.exponential(PGb, size=n),
                    rng.exponential(PGd, size=n))


def _sample_N(Nbar_arr, F_arr, rng):
    """
    Sample N (number of e/h pairs) for each event from a half-normal truncated
    at 0.  Mean = Nbar, Var = Nbar * F.

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


def _generate_events(n_events, *, Nbar_func,
                     F0,
                     eps,
                     V,
                     p0, p10,
                     q0, q10,
                     PNa, PNb, PNd,
                     seed):
    """
    Shared generator: draws Er from the biexponential PNa/PNb/PNd spectrum
    and mean e/h pair count Nbar_func(Er), then simulates the rest of the
    detector chain. NR/ER differ only in Nbar_func and which spectrum
    parameters they pass (see generate_NR_events/generate_ER_events).
    """
    rng = np.random.default_rng(seed)

    Er = _sample_PErN(n_events, rng, PNa=PNa, PNb=PNb, PNd=PNd)

    Nbar = Nbar_func(Er)                             # mean e/h pairs
    F_val = np.full_like(Er, F0)                      # Fano factor (constant)

    N = _sample_N(Nbar, F_val, rng)

    Ep_true = Er + N * (V / 1e3)                    # Neganov-Luke + recoil
    Eq_true = N * eps                                # ionization

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
                       eps,
                       V,
                       p0, p10,
                       q0, q10,
                       PNa, PNb, PNd,
                       seed):
    """
    Generate (Ep, Eq) pairs distributed according to PpqN.

    Returns
    -------
    Ep : ndarray, shape (n_events,)
    Eq : ndarray, shape (n_events,)
    Er : ndarray, shape (n_events,)   – latent true recoil energies
    N  : ndarray, shape (n_events,)   – latent e/h pair counts
    """
    return _generate_events(n_events, Nbar_func=lambda Er: ppq.Y(Er, k=k, Z=Z) * Er / eps,
                            F0=F0, eps=eps, V=V,
                            p0=p0, p10=p10, q0=q0, q10=q10,
                            PNa=PNa, PNb=PNb, PNd=PNd, seed=seed)


def generate_ER_events(n_events, *,
                       F0,
                       eps,
                       V,
                       p0, p10,
                       q0, q10,
                       PGa, PGb, PGd,
                       seed):
    """
    Generate (Ep, Eq) pairs distributed according to PpqG.

    Gamma / electron-recoil events have ionization yield Y = 1.

    Returns
    -------
    Ep : ndarray, shape (n_events,)
    Eq : ndarray, shape (n_events,)
    Er : ndarray, shape (n_events,)
    N  : ndarray, shape (n_events,)
    """
    return _generate_events(n_events, Nbar_func=lambda Er: Er / eps,
                            F0=F0, eps=eps, V=V,
                            p0=p0, p10=p10, q0=q0, q10=q10,
                            PNa=PGa, PNb=PGb, PNd=PGd, seed=seed)
