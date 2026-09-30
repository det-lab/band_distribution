submodule(PpqFort_m) PpqFort_s
  implicit none

  ! Number of composite Simpson 1/3 points for the N integral (must be
  ! odd).  If this changes, update the simps_w weights below to match:
  ! the pattern is 1, 4, 2, 4, ..., 2, 4, 1.  (Generating the weights
  ! from n_quad_N with a typed implied-do is F2018 that gfortran 15
  ! does not yet parse.)  21 points reproduce adaptive quadrature to
  ! ~1e-12; 15 points are ~25% faster at ~4e-5 relative error, too
  ! coarse for likelihood work.
  integer, parameter :: n_quad_N = 21

  ! Simpson 1/3 weights, matching n_quad_N
  real(c_double), parameter :: simps_w(n_quad_N) = &
      [1.0d0, 4.0d0, 2.0d0, 4.0d0, 2.0d0, 4.0d0, 2.0d0, 4.0d0, 2.0d0, 4.0d0, 2.0d0, &
       4.0d0, 2.0d0, 4.0d0, 2.0d0, 4.0d0, 2.0d0, 4.0d0, 2.0d0, 4.0d0, 1.0d0]

  ! Cap on the number of Er sample points in any one integration piece.
  ! Protects against point-count blow-up when a window's step comes from
  ! a mode much narrower than the window it ends up in (e.g. after
  ! merging): 20,000 uniform points are denser than the old full-range
  ! 0.01 keV grid for any piece narrower than 200 keV.
  integer, parameter :: max_pts = 20000

  ! Lower edge of the Er integration domain (keV)
  real(c_double), parameter :: er_min = 1.0d-6

  ! Shortest decay length in the gamma recoil spectrum PErG (keV); must
  ! stay in sync with PGb inside the PErG module procedure.  The Er
  ! sampling step is capped at a fraction of this scale when the
  ! integration window reaches down to the low-Er spectrum spike.
  real(c_double), parameter :: PGb_scale = 0.169520023d0

  ! The Er integrand can be multimodal (typically bimodal, usually
  ! dominated by the first, sharp peak).  Rather than assume dominance,
  ! every local maximum of the closed-form log-integrand gets its own
  ! integration window; overlapping windows are merged.
  integer, parameter :: max_modes = 4

  ! Bundles the scalar model/detector constants that are fixed for one
  ! PpqN/PpqG/PpqFullN call, so the internal engine below threads one
  ! argument instead of nine.  Internal-only: never crosses the bind(c)
  ! boundary, so the public interface in PpqFort_m.f90 is unaffected.
  ! Zfac = Z**(-7/3) (not raw Z) since that's the form the hot path needs,
  ! computed once here rather than once per Er sample.
  type :: detector_params_t
      real(c_double) :: k, Zfac, F0, eps, V, p0, p10, q0, q10
  end type detector_params_t

  ! ---------------------------------------------------------------------
  ! Gauss-Legendre doubling-verified quadrature: PpqN_region / PpqG_region.
  ! Replaces an earlier fixed-grid trapezoid implementation (superseded
  ! entirely -- no reason to keep both once this is faster and more
  ! accurate) and, before that, a Cuba/Cuhre-based implementation
  ! (generic adaptive cubature) dropped after real testing showed
  ! Cuhre's own convergence flag is not trustworthy for this integrand at
  ! loose tolerances (it reported success on the ER-band target region at
  ! ~180x worse than the requested accuracy).  Unlike Cuhre, we already
  ! know exactly where the band ridge is (ridge_eq_and_width/
  ! build_ridge_table) so no generic subdivision is needed to find it;
  ! what actually made the old fixed grid slow was trapezoid's O(h^2)
  ! convergence, not "not knowing where to look". Gauss-Legendre converges
  ! spectrally for the smooth integrand integrate_band is once inside the
  ! ridge window, so a modest order reaches the same accuracy as the fixed
  ! grid's ~97,000 points in ~2,500-8,000 -- confirmed empirically in
  ! Python before writing this (see gl_experiment.py in session history).
  ! Because we own the quadrature rule completely, convergence is verified
  ! by computing at order N and 2N and requiring agreement (doubling,
  ! escalating up to a cap, erroring out rather than guessing if it's
  ! never reached) instead of trusting an external library's heuristic.
  !
  ! Nodes/weights on [-1,1] from numpy.polynomial.legendre.leggauss(N)
  ! (float64); regenerate with, e.g.:
  !   python3 -c "import numpy as np; x,w = np.polynomial.legendre.leggauss(32); print(x); print(w)"
  include "gl_tables.f90.inc"

  ! Table resolution used by build_ridge_table below.
  integer(c_int), parameter :: region_ridge_n_tab = 4000

contains

  ! Zfac is 0 (and unused downstream) when is_gamma is .true.: PpqG passes
  ! a placeholder Z that may be 0, and Z**(-7/3) would be invalid there.
  pure function make_params(k, Z, F0, eps, V, p0, p10, q0, q10, is_gamma) result(pars)
      real(c_double), intent(in) :: k, Z, F0, eps, V, p0, p10, q0, q10
      logical, intent(in) :: is_gamma
      type(detector_params_t) :: pars

      pars%k   = k
      pars%F0  = F0
      pars%eps = eps
      pars%V   = V
      pars%p0  = p0
      pars%p10 = p10
      pars%q0  = q0
      pars%q10 = q10
      if (is_gamma) then
          pars%Zfac = 0.0d0
      else
          pars%Zfac = Z**(-7.0d0/3.0d0)
      end if
  end function make_params

  module procedure PpqN_vector
      integer :: i

      do concurrent(i = 1:n) default (none) shared(res_arr, Ep_arr, Eq_arr, k, Z, F0, eps, V, p0, p10, q0, q10)
          res_arr(i) = PpqN(Ep_arr(i), Eq_arr(i), k, Z, F0, eps, V, p0, p10, q0, q10)
      end do
  end procedure PpqN_vector

  module procedure PpqG_vector
      integer :: i

      do concurrent(i = 1:n) default (none) shared(res_arr, Ep_arr, Eq_arr, F0, eps, V, p0, p10, q0, q10)
          res_arr(i) = PpqG(Ep_arr(i), Eq_arr(i), F0, eps, V, p0, p10, q0, q10)
      end do
  end procedure PpqG_vector

  ! PpqN, PpqG: integrate P(Ep,Eq|Er)*P(Er) dEr.
  !
  ! The integrand is a narrow bump in Er: the phonon and ionization
  ! measurements each constrain Er to within a few sigma, so sampling the
  ! full physical range wastes almost every evaluation.  Instead, locate
  ! the peak of the closed-form log-integrand (Laplace approximation) and
  ! integrate only over peak +/- 8 sigma with a uniform trapezoid rule.
  ! For a smooth integrand that vanishes at both window edges the uniform
  ! trapezoid rule converges spectrally, so ~100 points reproduce the old
  ! full-range fine grid (~35,000 points) to near machine precision.

  module procedure PpqN
      type(detector_params_t) :: pars

      if (F0 == 0.0d0) &
          error stop "Fano factor F(Er) = F0 is zero for all Er"
      pars = make_params(k, Z, F0, eps, V, p0, p10, q0, q10, .false.)
      res = integrate_band(Ep, Eq, pars, .false.)
  end procedure PpqN

  module procedure PpqG
      type(detector_params_t) :: pars

      if (F0 == 0.0d0) &
          error stop "Fano factor F(Er) = F0 is zero for all Er"
      ! k, Z are unused when is_gamma is .true. (electron-recoil yield is
      ! always 1); the placeholders below are never read.
      pars = make_params(0.0d0, 0.0d0, F0, eps, V, p0, p10, q0, q10, .true.)
      res = integrate_band(Ep, Eq, pars, .true.)
  end procedure PpqG

  module procedure Y
      real(c_double) :: eps_L, g

      eps_L = 11.5d0 * abs(Er) * Z**(-7.0d0/3.0d0)
      g     = 3.0d0*eps_L**0.15d0 + 0.7d0*eps_L**0.29d0 + eps_L
      res   = k*g / (1.0d0 + k*g)
  end procedure Y

  module procedure Nbar
      res = Y(Er, k, Z) * Er / eps
  end procedure Nbar

  module procedure sigp
      real(c_double) :: e2pre, e2e

      e2pre = (p10**2 - p0**2)
      e2e = (Ep / (10.0d0 * (1.0d0 + (V / eps / 1000.0d0))))**2
      res = sqrt(p0**2 + e2pre * e2e)
  end procedure sigp

  module procedure sigq
      real(c_double) :: e2pre, e2e

      e2pre = (q10**2 - q0**2)
      e2e = (Eq / 10.0d0)**2
      res = sqrt(q0**2 + e2pre * e2e)
  end procedure sigq

  ! The bind(c) module procedures cannot themselves be elemental, so the
  ! spectrum formulas live in elemental implementations that array-valued
  ! internal code (the mode scan) can evaluate in vectorizable form.
  module procedure PErN
      res = PErN_impl(Er)
  end procedure PErN

  module procedure PErG
      res = PErG_impl(Er)
  end procedure PErG

  elemental function PErN_impl(Er) result(res)
      real(c_double), intent(in) :: Er
      real(c_double) :: res
      real(c_double), parameter :: PNa = 0.53693208d0
      real(c_double), parameter :: PNb = 6.41515782d0
      real(c_double), parameter :: PNd = 23.71789286d0
      real(c_double), parameter :: PNa_over_PNb   = PNa / PNb
      real(c_double), parameter :: one_minus_PNa_over_PNd = (1.0d0 - PNa) / PNd

      res = PNa_over_PNb * exp(-Er / PNb) + one_minus_PNa_over_PNd * exp(-Er / PNd)
  end function PErN_impl

  elemental function PErG_impl(Er) result(res)
      real(c_double), intent(in) :: Er
      real(c_double) :: res
      real(c_double), parameter :: PGa = 0.573211975d0
      real(c_double), parameter :: PGb = 0.169520023d0
      real(c_double), parameter :: PGd = 279.552394d0
      real(c_double), parameter :: PGa_over_PGb   = PGa / PGb
      real(c_double), parameter :: one_minus_PGa_over_PGd = (1.0d0 - PGa) / PGd

      res = PGa_over_PGb * exp(-Er / PGb) + one_minus_PGa_over_PGd * exp(-Er / PGd)
  end function PErG_impl

  ! Lindhard yield, hot-path form: takes Zfac = Z**(-7/3) precomputed once
  ! per PpqN/PpqG call (by integrate_band) instead of recomputing the Z
  ! power at every one of the thousands of Er samples in the mode scan and
  ! window integration.  Matches the public Y module procedure's formula.
  elemental function lindhard_yield_zfac(Er, k, Zfac) result(res)
      real(c_double), intent(in) :: Er, k, Zfac
      real(c_double) :: res
      real(c_double) :: eps_L, g

      eps_L = 11.5d0 * abs(Er) * Zfac
      g     = 3.0d0*eps_L**0.15d0 + 0.7d0*eps_L**0.29d0 + eps_L
      res   = k*g / (1.0d0 + k*g)
  end function lindhard_yield_zfac

  module procedure PpqFort_version
      major = version_major
      minor = version_minor
      patch = version_patch
  end procedure PpqFort_version

  ! PpqFullN: integrand for the Er integral, P(Ep,Eq|Er)*P(Er), NR case.
  ! Evaluates the N integral numerically via 21-point Simpson's rule.
  ! sigp and sigq are evaluated at exact noiseless energies (Er+V*N/1000,
  ! eps*N) per sample point, matching the data simulator exactly.
  module procedure PpqFullN
      type(detector_params_t) :: pars

      pars = make_params(k, Z, F0, eps, V, p0, p10, q0, q10, .false.)
      res = PpqCondN(Er, Ep, Eq, pars, .false.) * PErN(Er)
  end procedure PpqFullN

  ! PpqN_region / PpqG_region: integral of PpqN/PpqG over a rectangular
  ! (Ep,Eq) region, e.g. to normalize a likelihood to its fit region.
  ! See region_integral_gl for the algorithm.
  module procedure PpqN_region
      type(detector_params_t) :: pars

      pars = make_params(k, Z, F0, eps, V, p0, p10, q0, q10, .false.)
      res = region_integral_gl(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
                               pars, .false.)
  end procedure PpqN_region

  module procedure PpqG_region
      type(detector_params_t) :: pars

      ! k, Z are unused when is_gamma is .true.; placeholders as elsewhere.
      pars = make_params(0.0d0, 0.0d0, F0, eps, V, p0, p10, q0, q10, .true.)
      res = region_integral_gl(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
                               pars, .true.)
  end procedure PpqG_region

  ! ---------------------------------------------------------------------
  ! Private helpers
  ! ---------------------------------------------------------------------

  ! P(Ep,Eq|Er): the 21-point Simpson N integral, without the recoil
  ! spectrum factor.  is_gamma selects the electron-recoil yield (Y=1)
  ! instead of the Lindhard NR yield; pars%Zfac is Z**(-7/3), precomputed
  ! once per PpqN/PpqG call (unused when is_gamma is .true.).
  pure function PpqCondN(Er, Ep, Eq, pars, is_gamma) result(res)
      real(c_double), intent(in) :: Er, Ep, Eq
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma
      real(c_double) :: res
      real(c_double) :: F_val, Nbar_val, sigma_N, norm_const
      real(c_double) :: sp0, sp1, sq0, sq1
      real(c_double) :: EP_mean, EQ_mean, sigp_mean_sq, sigq_mean_sq
      real(c_double) :: a_coeff, b_coeff, N_star, sigma_Neff
      real(c_double) :: N_lo, N_hi, h_N
      real(c_double) :: N_j, EP_nl, EQ_nl, sig_p2, sig_q2
      real(c_double) :: integrand_arr(n_quad_N), exp_args(n_quad_N), inv_sig(n_quad_N)
      integer :: j

      F_val = pars%F0
      if (is_gamma) then
          Nbar_val = abs(Er) / pars%eps
      else
          Nbar_val = lindhard_yield_zfac(Er, pars%k, pars%Zfac) * Er / pars%eps
      end if
      if (Nbar_val <= 0.0d0 .or. F_val <= 0.0d0) then
          res = 0.0d0
          return
      end if
      sigma_N    = sqrt(Nbar_val * F_val)
      norm_const = 1.0d0 / ((2.0d0 * pi)**1.5d0)
      sp0 = pars%p0**2
      sp1 = (pars%p10**2 - pars%p0**2) / (10.0d0 * (1.0d0 + pars%V / (pars%eps * 1.0d3)))**2
      sq0 = pars%q0**2
      sq1 = (pars%q10**2 - pars%q0**2) / 100.0d0

      EP_mean      = Er + (pars%V * 1.0d-3) * Nbar_val
      EQ_mean      = pars%eps * Nbar_val
      sigp_mean_sq = sp0 + sp1 * EP_mean**2
      sigq_mean_sq = sq0 + sq1 * EQ_mean**2
      a_coeff = (pars%V * 1.0d-3) * (Ep - Er) / sigp_mean_sq &
              + pars%eps * Eq / sigq_mean_sq &
              + 1.0d0 / F_val
      b_coeff = 1.0d0 / (2.0d0 * Nbar_val * F_val) &
              + pars%eps**2 / (2.0d0 * sigq_mean_sq) &
              + pars%V**2 / (2.0d0 * 1.0d6 * sigp_mean_sq)
      N_star     = a_coeff / (2.0d0 * b_coeff)
      sigma_Neff = 1.0d0 / sqrt(2.0d0 * b_coeff)
      if (N_star - 5.0d0*sigma_Neff < 0.0d0) then
          N_lo = 0.0d0
          N_hi = 10.0d0 * sigma_Neff
      else
          N_lo = N_star - 5.0d0 * sigma_Neff
          N_hi = N_star + 5.0d0 * sigma_Neff
      end if
      h_N = (N_hi - N_lo) / real(n_quad_N - 1, c_double)
      ! Two branch-free loops so both SIMD-vectorize (a single loop with
      ! a conditional exp defeats the auto-vectorizer).  No underflow
      ! guard is needed: exp_arg <= 0 by construction and exp underflows
      ! smoothly to zero for very negative arguments, which is exactly
      ! the behaviour the old skip-below--700 branch hand-coded.  With
      ! the integration window centred on the peak, nearly every point
      ! is within 8 sigma anyway.
      do j = 1, n_quad_N
          N_j   = N_lo + (j - 1) * h_N
          EP_nl = Er + (pars%V * 1.0d-3) * N_j
          EQ_nl = pars%eps * N_j
          sig_p2 = sp0 + sp1 * EP_nl**2
          sig_q2 = sq0 + sq1 * EQ_nl**2
          inv_sig(j) = 1.0d0 / sqrt(sig_p2 * sig_q2)
          exp_args(j) = -0.5d0*(Ep - EP_nl)**2/sig_p2 &
                        - 0.5d0*(Eq - EQ_nl)**2/sig_q2 &
                        - 0.5d0*((N_j - Nbar_val)/sigma_N)**2
      end do
      do j = 1, n_quad_N
          integrand_arr(j) = exp(exp_args(j)) * inv_sig(j)
      end do
      res = (norm_const / sigma_N) * (h_N / 3.0d0) * dot_product(simps_w, integrand_arr)
  end function PpqCondN

  ! The full integrand P(Ep,Eq|Er)*P(Er) with the spectrum and yield
  ! selected by is_gamma (electron recoils: Y=1, PErG; nuclear recoils:
  ! Lindhard(k,Zfac), PErN).
  pure function band_integrand(Er, Ep, Eq, pars, is_gamma) result(res)
      real(c_double), intent(in) :: Er, Ep, Eq
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma
      real(c_double) :: res

      res = PpqCondN(Er, Ep, Eq, pars, is_gamma)
      if (is_gamma) then
          res = res * PErG(Er)
      else
          res = res * PErN(Er)
      end if
  end function band_integrand

  ! Closed-form approximation to log of the Er integrand WITHOUT the
  ! recoil spectrum: complete the square in N analytically, with
  ! sigp/sigq frozen at the measured energies.  Not accurate enough for
  ! the PDF value itself -- the real integrand evaluates sigp/sigq at
  ! noiseless energies -- but the peak location and curvature agree to
  ! well within a fraction of the peak width, which is all the window
  ! placement needs.
  !
  ! Branch-free on purpose: the validity guards are computed with
  ! clamped-safe values and applied with merge (a blend, not a branch)
  ! so the mode scan's array evaluation vectorizes cleanly.
  elemental function log_band_exponent(Er, Ep, Eq, pars, sp2, sq2, is_gamma) result(H)
      real(c_double), intent(in) :: Er, Ep, Eq
      type(detector_params_t), intent(in) :: pars
      real(c_double), intent(in) :: sp2, sq2
      logical, intent(in) :: is_gamma
      real(c_double) :: H
      real(c_double) :: Er_s, F_s, Nbar_s, aN, bN, cN
      logical :: valid

      Er_s = max(Er, 1.0d-30)
      F_s  = max(pars%F0, 1.0d-30)
      if (is_gamma) then
          Nbar_s = max(Er_s / pars%eps, 1.0d-30)
      else
          Nbar_s = max(lindhard_yield_zfac(Er_s, pars%k, pars%Zfac) * Er_s / pars%eps, 1.0d-30)
      end if
      valid = Er > 0.0d0 .and. pars%F0 > 0.0d0 .and. (is_gamma .or. pars%k > 0.0d0)

      aN = (pars%V * 1.0d-3) * (Ep - Er_s) / sp2 + pars%eps * Eq / sq2 + 1.0d0 / F_s
      bN = 1.0d0 / (2.0d0 * Nbar_s * F_s) + pars%eps**2 / (2.0d0 * sq2) &
         + pars%V**2 / (2.0d0 * 1.0d6 * sp2)
      cN = -(Ep - Er_s)**2 / (2.0d0 * sp2) - Eq**2 / (2.0d0 * sq2) &
         - Nbar_s / (2.0d0 * F_s)
      H = merge(cN + aN**2 / (4.0d0 * bN), -1.0d30, valid)
  end function log_band_exponent

  ! Scalar convenience for the Newton refinement: band exponent plus the
  ! log recoil spectrum.
  pure function log_integrand_approx(Er, Ep, Eq, pars, sp2, sq2, is_gamma) result(H)
      real(c_double), intent(in) :: Er, Ep, Eq
      type(detector_params_t), intent(in) :: pars
      real(c_double), intent(in) :: sp2, sq2
      logical, intent(in) :: is_gamma
      real(c_double) :: H

      H = log_band_exponent(Er, Ep, Eq, pars, sp2, sq2, is_gamma)
      if (is_gamma) then
          H = H + log(PErG_impl(Er))
      else
          H = H + log(PErN_impl(Er))
      end if
  end function log_integrand_approx

  ! Locate every mode of the Er integrand and estimate their widths.
  !
  ! An unconditional log-spaced scan of the closed-form log-integrand
  ! (cheap: no N integral) finds all candidate local maxima -- log
  ! spacing so the sharp low-Er peak cannot fall between scan points.
  ! Each candidate is refined with Newton iterations; sigma = 1/sqrt(-H'')
  ! from the curvature at the refined peak.  A candidate at the low-Er
  ! boundary with the integrand monotonically decaying away from it is
  ! kept as a boundary mode whose sigma is the exponential decay length.
  ! Modes more than ~100 log units below the best one cannot contribute
  ! and are dropped; near-duplicate refinements are merged.
  !
  ! n_modes = 0 signals failure; the caller falls back to the original
  ! full-range grid.
  pure subroutine locate_modes(Ep, Eq, pars, sp2, sq2, is_gamma, &
                               er_m, sig_m, H_m, bnd_m, n_modes)
      real(c_double), intent(in) :: Ep, Eq
      type(detector_params_t), intent(in) :: pars
      real(c_double), intent(in) :: sp2, sq2
      logical, intent(in) :: is_gamma
      real(c_double), intent(out) :: er_m(max_modes), sig_m(max_modes), H_m(max_modes)
      logical, intent(out) :: bnd_m(max_modes)
      integer, intent(out) :: n_modes

      ! n_scan = 512 is a deliberately conservative choice, not a tuned
      ! minimum.  A log-spaced scan finds every mode of a well-behaved
      ! landscape regardless of point count -- the only failure mode is
      ! two peaks separated by a valley narrower than the local scan
      ! spacing, which would show up as a large (order-unity) relative
      ! error, not a small one.  Empirically, dropping to n_scan = 64
      ! changed PpqN/PpqG values by at most ~1e-8 relative (vs a 4096-
      ! point reference) across the physical parameters and 10 random
      ! draws from the fit-parameter space, including the draws that
      ! produced genuine multimodality -- no mode was ever lost, and
      ! the achievable speedup from shrinking the scan is only ~10-15%
      ! (the scan is a small fraction of the total cost).  512 is kept
      ! as headroom against a genuinely narrow valley outside this
      ! tested ensemble, since the cost of being wrong (silently
      ! dropping a mode) is far higher than the ~1 us/eval this buys.
      integer, parameter :: n_scan = 512
      real(c_double), parameter :: x_scan_min = 1.0d-4
      real(c_double) :: xs(n_scan), Hs(n_scan)
      real(c_double) :: maxx_scan, dlx, x, d, H1, H2, sig, Hv
      integer :: ik, m, j, ncand, cand(n_scan), best
      logical :: ok, duplicate

      n_modes = 0
      maxx_scan = max(1.1d0 * max(Ep, Eq), max(Ep, Eq) + 10.0d0)

      dlx = (log(maxx_scan) - log(x_scan_min)) / (n_scan - 1)
      do ik = 1, n_scan
          xs(ik) = exp(log(x_scan_min) + (ik - 1) * dlx)
      end do
      ! Branch-free band exponent over the whole grid, then the log
      ! spectrum as a second array pass with the band choice hoisted out
      ! of the loop: everything the scan does is vectorizable.
      Hs = log_band_exponent(xs, Ep, Eq, pars, sp2, sq2, is_gamma)
      if (is_gamma) then
          Hs = Hs + log(PErG_impl(xs))
      else
          Hs = Hs + log(PErN_impl(xs))
      end if

      ! Candidate local maxima (including the low-Er boundary)
      ncand = 0
      do ik = 1, n_scan
          if (Hs(ik) < -1.0d29) cycle
          if ((ik == 1      .or. Hs(ik) >= Hs(ik - 1)) .and. &
              (ik == n_scan .or. Hs(ik) >  Hs(ik + 1))) then
              ncand = ncand + 1
              cand(ncand) = ik
          end if
      end do
      if (ncand == 0) return

      ! Sort candidates by scan height, tallest first (ncand is small)
      do m = 1, ncand - 1
          best = m
          do j = m + 1, ncand
              if (Hs(cand(j)) > Hs(cand(best))) best = j
          end do
          ik = cand(m); cand(m) = cand(best); cand(best) = ik
      end do

      refine_candidates: do m = 1, ncand
          if (n_modes == max_modes) exit
          ! Anything this far below the tallest mode contributes nothing
          if (n_modes > 0) then
              if (Hs(cand(m)) < maxval(H_m(1:n_modes)) - 100.0d0) cycle
          end if

          ik = cand(m)
          if (ik == 1) then
              ! Boundary mode: integrand decays away from the low-Er edge
              x = xs(1)
              d = 1.0d-5
              H1 = (log_integrand_approx(x + d, Ep, Eq, pars, sp2, sq2, is_gamma) &
                  - Hs(1)) / d
              if (H1 >= 0.0d0) cycle
              n_modes = n_modes + 1
              er_m(n_modes)  = x
              sig_m(n_modes) = min(-1.0d0 / H1, maxx_scan)
              H_m(n_modes)   = Hs(1)
              bnd_m(n_modes) = .true.
              cycle
          end if

          x = xs(ik)
          call newton_refine(x, ok, H1, H2)
          if (.not. ok) then
              ! Accept the scan point with finite-difference curvature if usable
              x = xs(ik)
              d = max(1.0d-5, 1.0d-5 * x)
              H2 = (log_integrand_approx(x + d, Ep, Eq, pars, sp2, sq2, is_gamma) &
                  - 2.0d0 * Hs(ik) &
                  + log_integrand_approx(x - d, Ep, Eq, pars, sp2, sq2, is_gamma)) / d**2
              if (H2 >= 0.0d0) cycle
          end if
          sig = 1.0d0 / sqrt(-H2)
          Hv  = log_integrand_approx(x, Ep, Eq, pars, sp2, sq2, is_gamma)
          if (sig /= sig .or. sig <= 1.0d-6 .or. sig > maxx_scan) cycle

          ! Drop refinements that landed on an already-recorded mode
          duplicate = .false.
          do j = 1, n_modes
              if (abs(x - er_m(j)) < 0.5d0 * (sig + sig_m(j))) then
                  duplicate = .true.
                  exit
              end if
          end do
          if (duplicate) cycle

          n_modes = n_modes + 1
          er_m(n_modes)  = x
          sig_m(n_modes) = sig
          H_m(n_modes)   = Hv
          bnd_m(n_modes) = .false.
      end do refine_candidates

      if (n_modes == 0) then
          ! Every candidate failed to refine (pathological closed-form
          ! shapes far off band).  Accept the tallest scan point with a
          ! conservative width -- the local scan spacing -- and let the
          ! edge guard grow the window as needed.  Far cheaper than the
          ! full-range-grid fallback this used to trigger.
          ik = cand(1)
          n_modes  = 1
          er_m(1)  = xs(ik)
          sig_m(1) = max(xs(min(ik + 1, n_scan)) - xs(max(ik - 1, 1)), 1.0d-3)
          H_m(1)   = Hs(ik)
          bnd_m(1) = .false.
      end if

  contains

      pure subroutine newton_refine(xx, converged, H1_out, H2_out)
          real(c_double), intent(inout) :: xx
          logical, intent(out) :: converged
          real(c_double), intent(out) :: H1_out, H2_out
          integer :: it
          real(c_double) :: dd, Hm, H0, Hp, step

          converged = .false.
          H1_out = 0.0d0
          H2_out = 0.0d0
          do it = 1, 60
              dd = max(1.0d-4, 1.0d-4 * xx)
              Hm = log_integrand_approx(xx - dd, Ep, Eq, pars, sp2, sq2, is_gamma)
              H0 = log_integrand_approx(xx,      Ep, Eq, pars, sp2, sq2, is_gamma)
              Hp = log_integrand_approx(xx + dd, Ep, Eq, pars, sp2, sq2, is_gamma)
              H1_out = (Hp - Hm) / (2.0d0 * dd)
              H2_out = (Hp - 2.0d0 * H0 + Hm) / dd**2
              if (H2_out >= 0.0d0) return
              step = -H1_out / H2_out
              step = sign(min(abs(step), 0.5d0 * xx + 1.0d0), step)
              if (xx + step <= 0.0d0) then
                  xx = 0.5d0 * xx
              else
                  xx = xx + step
              end if
              if (xx > 1.5d0 * maxx_scan) return
              if (abs(step) < 1.0d-4) then
                  converged = .true.
                  return
              end if
          end do
      end subroutine newton_refine

  end subroutine locate_modes

  ! Trapezoid rule over [lo, lo + (npts-1)*h].  For a smooth integrand
  ! vanishing at both edges this is spectrally accurate on a uniform grid.
  pure function integrate_window(lo, h, npts, Ep, Eq, pars, is_gamma) result(total)
      real(c_double), intent(in) :: lo, h, Ep, Eq
      type(detector_params_t), intent(in) :: pars
      integer, intent(in) :: npts
      logical, intent(in) :: is_gamma
      real(c_double) :: total
      integer :: i
      real(c_double) :: f_i, w_i

      total = 0.0d0
      do concurrent(i = 1:npts) default(none) reduce(+: total) &
          shared(lo, h, npts, Ep, Eq, pars, is_gamma) &
          local(f_i, w_i)
          f_i = band_integrand(lo + (i - 1) * h, Ep, Eq, pars, is_gamma)
          w_i = merge(0.5d0, 1.0d0, i == 1 .or. i == npts)
          total = total + w_i * f_i
      end do
      total = total * h
  end function integrate_window

  ! Integrate one window, refining the sampling step only over the part
  ! of the window that overlaps the sharp low-Er component of the gamma
  ! spectrum (decay length PGb).  The two pieces share their split point,
  ! whose two trapezoid half-weights sum to the exact union.
  pure function integrate_piecewise(lo, hi, h_target, Ep, Eq, pars, is_gamma) result(total)
      real(c_double), intent(in) :: lo, hi, h_target, Ep, Eq
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma
      real(c_double) :: total
      real(c_double), parameter :: spike_edge = 5.0d0 * PGb_scale
      real(c_double) :: h_fine

      if (is_gamma .and. lo < spike_edge) then
          h_fine = min(h_target, PGb_scale / 6.0d0)
          if (hi > spike_edge) then
              total = one_piece(lo, spike_edge, h_fine) + one_piece(spike_edge, hi, h_target)
          else
              total = one_piece(lo, hi, h_fine)
          end if
      else
          total = one_piece(lo, hi, h_target)
      end if

  contains

      pure function one_piece(p_lo, p_hi, p_ht) result(piece)
          real(c_double), intent(in) :: p_lo, p_hi, p_ht
          real(c_double) :: piece
          integer :: npts
          real(c_double) :: h

          npts  = min(max(int((p_hi - p_lo) / p_ht) + 2, 25), max_pts)
          h     = (p_hi - p_lo) / (npts - 1)
          piece = integrate_window(p_lo, h, npts, Ep, Eq, pars, is_gamma)
      end function one_piece

  end function integrate_piecewise

  ! Shared driver for PpqN / PpqG: locate every integrand mode, place an
  ! integration window around each, merge overlapping windows, and guard
  ! the window edges.
  pure function integrate_band(Ep, Eq, pars, is_gamma) result(res)
      real(c_double), intent(in) :: Ep, Eq
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma
      real(c_double) :: res
      real(c_double) :: sp2, sq2, maxx_scan, resolution
      real(c_double) :: er_m(max_modes), sig_m(max_modes), H_m(max_modes)
      logical :: bnd_m(max_modes)
      real(c_double) :: w_lo(max_modes), w_hi(max_modes), w_ht(max_modes), w_sig(max_modes)
      real(c_double) :: f_star, f_lo, f_hi, lo_limit, hi_limit, tmp
      integer :: n_modes, n_win, m, j, npts, guard
      logical :: need_lo, need_hi

      sp2 = sigp(Ep, pars%eps, pars%V, pars%p0, pars%p10)**2
      sq2 = sigq(Eq, pars%q0, pars%q10)**2

      call locate_modes(Ep, Eq, pars, sp2, sq2, is_gamma, &
                        er_m, sig_m, H_m, bnd_m, n_modes)

      if (n_modes == 0) then
          ! Last resort: the original fine grid over the full range.
          ! Guaranteed correct, rarely taken.
          maxx_scan  = max(1.1d0 * max(Ep, Eq), max(Ep, Eq) + 10.0d0)
          resolution = merge(0.002d0, 0.01d0, maxx_scan < 15.0d0)
          npts = int((maxx_scan - er_min) / resolution) + 1
          res  = integrate_window(er_min, resolution, npts, Ep, Eq, pars, is_gamma)
          return
      end if

      ! Predicted peak log-value so small the integral underflows to zero
      if (maxval(H_m(1:n_modes)) < -600.0d0) then
          res = 0.0d0
          return
      end if

      ! One window per mode: +/- 8 sigma around an interior peak; for a
      ! boundary decay, 35 decay lengths puts the truncated tail below
      ! 1e-15 relative.
      do m = 1, n_modes
          if (bnd_m(m)) then
              w_lo(m) = er_min
              w_hi(m) = er_m(m) + 35.0d0 * sig_m(m)
          else
              w_lo(m) = max(er_min, er_m(m) - 8.0d0 * sig_m(m))
              w_hi(m) = er_m(m) + 8.0d0 * sig_m(m)
          end if
          w_ht(m)  = sig_m(m) / 6.0d0
          w_sig(m) = sig_m(m)
      end do

      ! Sort windows by lower edge (n_modes <= 4)
      do m = 1, n_modes - 1
          do j = 1, n_modes - m
              if (w_lo(j) > w_lo(j + 1)) then
                  tmp = w_lo(j);  w_lo(j)  = w_lo(j + 1);  w_lo(j + 1)  = tmp
                  tmp = w_hi(j);  w_hi(j)  = w_hi(j + 1);  w_hi(j + 1)  = tmp
                  tmp = w_ht(j);  w_ht(j)  = w_ht(j + 1);  w_ht(j + 1)  = tmp
                  tmp = w_sig(j); w_sig(j) = w_sig(j + 1); w_sig(j + 1) = tmp
              end if
          end do
      end do

      ! Merge overlapping windows, keeping the finer step
      n_win = 1
      do m = 2, n_modes
          if (w_lo(m) <= w_hi(n_win)) then
              w_hi(n_win)  = max(w_hi(n_win), w_hi(m))
              w_ht(n_win)  = min(w_ht(n_win), w_ht(m))
              w_sig(n_win) = min(w_sig(n_win), w_sig(m))
          else
              n_win = n_win + 1
              w_lo(n_win)  = w_lo(m)
              w_hi(n_win)  = w_hi(m)
              w_ht(n_win)  = w_ht(m)
              w_sig(n_win) = w_sig(m)
          end if
      end do

      ! Edge-guard threshold: the tallest mode sets the scale
      f_star = 0.0d0
      do m = 1, n_modes
          f_star = max(f_star, band_integrand(er_m(m), Ep, Eq, pars, is_gamma))
      end do

      res = 0.0d0
      do m = 1, n_win
          ! Guard expansions must not cross into neighbouring windows:
          ! adjacent edges share a point, whose two half-weights sum to
          ! the full trapezoid weight of the union.
          lo_limit = er_min
          if (m > 1) lo_limit = w_hi(m - 1)
          hi_limit = huge(1.0d0)
          if (m < n_win) hi_limit = w_lo(m + 1)
          w_lo(m) = max(w_lo(m), lo_limit)

          do guard = 1, 4
              tmp = integrate_piecewise(w_lo(m), w_hi(m), w_ht(m), Ep, Eq, pars, is_gamma)
              ! Edge guard: expand if the integrand has not died off at the
              ! window edges (protects against an underestimated sigma)
              f_lo = band_integrand(w_lo(m), Ep, Eq, pars, is_gamma)
              f_hi = band_integrand(w_hi(m), Ep, Eq, pars, is_gamma)
              need_lo = f_lo > 1.0d-10 * f_star .and. w_lo(m) > lo_limit
              need_hi = f_hi > 1.0d-10 * f_star .and. w_hi(m) < hi_limit
              if (.not. (need_lo .or. need_hi)) exit
              if (need_lo) w_lo(m) = max(lo_limit, w_lo(m) - 4.0d0 * w_sig(m))
              if (need_hi) w_hi(m) = min(hi_limit, w_hi(m) + 4.0d0 * w_sig(m))
          end do
          res = res + tmp
      end do
  end function integrate_band

  ! Linear interpolation of ytab(xtab) at x, matching numpy.interp:
  ! clamps to the table's end values outside its range.  xtab must be
  ! sorted ascending.
  pure function interp1d(x, xtab, ytab, n) result(y)
      real(c_double), intent(in) :: x
      integer, intent(in) :: n
      real(c_double), intent(in) :: xtab(n), ytab(n)
      real(c_double) :: y
      integer :: lo, hi, mid
      real(c_double) :: t

      if (x <= xtab(1)) then
          y = ytab(1)
      else if (x >= xtab(n)) then
          y = ytab(n)
      else
          ! Binary search for xtab(lo) <= x < xtab(lo+1): xtab is sorted
          ! (build_ridge_table's geomspace), so O(log n) instead of the
          ! O(n) linear scan this replaced -- matters now that
          ! find_window_breakpoints's scan+bisect calls this thousands
          ! of times per region_integral_gl call.
          lo = 1
          hi = n
          do while (hi - lo > 1)
              mid = (lo + hi) / 2
              if (xtab(mid) <= x) then
                  lo = mid
              else
                  hi = mid
              end if
          end do
          t = (x - xtab(lo)) / (xtab(lo + 1) - xtab(lo))
          y = ytab(lo) + t * (ytab(lo + 1) - ytab(lo))
      end if
  end function interp1d

  ! Tabulate the noiseless band ridge Er -> (Ep(Er), Eq(Er)) on a
  ! log-spaced Er grid from 1e-3 keV to er_hi.  Same physics as
  ! band_breakpoints.py: Eq = Y(Er)*Er, Ep = Er*(1 + Y(Er)*V/(1000*eps)),
  ! with Y=1 for the ER band (is_gamma) and the Lindhard yield otherwise.
  ! Both are monotone increasing in Er, so this table is invertible by
  ! interpolation (interp1d, via ridge_eq_and_width below).
  pure subroutine build_ridge_table(pars, is_gamma, er_hi, n_tab, ep_tab, eq_tab)
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma
      real(c_double), intent(in) :: er_hi
      integer, intent(in) :: n_tab
      real(c_double), intent(out) :: ep_tab(n_tab), eq_tab(n_tab)
      real(c_double), parameter :: er_lo = 1.0d-3
      real(c_double) :: dlx, er, y_val
      integer :: i

      dlx = (log(er_hi) - log(er_lo)) / real(n_tab - 1, c_double)
      do i = 1, n_tab
          er = exp(log(er_lo) + real(i - 1, c_double) * dlx)
          if (is_gamma) then
              y_val = 1.0d0
          else
              y_val = lindhard_yield_zfac(er, pars%k, pars%Zfac)
          end if
          eq_tab(i) = y_val * er
          ep_tab(i) = er * (1.0d0 + y_val * pars%V / (1000.0d0 * pars%eps))
      end do
  end subroutine build_ridge_table

  ! The band's local ridge Eq value and width in the Eq direction at a
  ! given Ep, inverting the tabulated ridge by interpolation.  Mirrors
  ! band_breakpoints.py's eq_ridge()/inner_points_func width derivation
  ! exactly: width = hypot(sigq(ridge), slope * sigp(ep)), slope by
  ! central finite difference of the interpolated ridge.
  pure subroutine ridge_eq_and_width(ep, ep_tab, eq_tab, n_tab, pars, eqr, width)
      real(c_double), intent(in) :: ep
      integer, intent(in) :: n_tab
      real(c_double), intent(in) :: ep_tab(n_tab), eq_tab(n_tab)
      type(detector_params_t), intent(in) :: pars
      real(c_double), intent(out) :: eqr, width
      real(c_double) :: dep, eqr_plus, eqr_minus, slope, sigp_val, sigq_val

      eqr = interp1d(ep, ep_tab, eq_tab, n_tab)
      dep = max(1.0d-3, 0.01d0 * ep)
      eqr_plus  = interp1d(ep + dep, ep_tab, eq_tab, n_tab)
      eqr_minus = interp1d(ep - dep, ep_tab, eq_tab, n_tab)
      slope = (eqr_plus - eqr_minus) / (2.0d0 * dep)
      sigp_val = sigp(ep, pars%eps, pars%V, pars%p0, pars%p10)
      sigq_val = sigq(eqr, pars%q0, pars%q10)
      width = sqrt(sigq_val**2 + (slope * sigp_val)**2)
  end subroutine ridge_eq_and_width

  ! One nested Gauss-Legendre pass at order n_ord: outer do concurrent
  ! over Gauss-Legendre nodes on Ep, inner sequential loop over
  ! Gauss-Legendre nodes spanning the local ridge-centred Eq window from
  ! ridge_eq_and_width (nodes/weights mapped from [-1,1] onto each real
  ! interval). A degenerate window (ridge entirely outside
  ! [eq_min,eq_max]) simply contributes 0.
  ! nodes_ep/n_ord_ep and nodes_eq/n_ord_eq are independent parameters,
  ! but region_integral_gl below always calls this with them equal:
  ! pairing the inner Eq sweep with a lower order than the outer Ep sweep
  ! was tried (reasoning the ridge-centred window needs less resolution)
  ! and measured *slower* overall, since the coarser Eq resolution made
  ! coarse/fine agreement harder to reach, forcing escalation to a higher
  ! Ep order than the matched-order scheme needed -- see the
  ! doubling ladder's pairing below).
  pure function gl_region_pass(ep_min, ep_max, eq_min, eq_max, ep_tab, eq_tab, n_tab, &
      pars, is_gamma, n_window_widths, &
      nodes_ep, weights_ep, n_ord_ep, nodes_eq, weights_eq, n_ord_eq) result(res)
      real(c_double), intent(in) :: ep_min, ep_max, eq_min, eq_max
      integer, intent(in) :: n_tab
      real(c_double), intent(in) :: ep_tab(n_tab), eq_tab(n_tab)
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma
      real(c_double), intent(in) :: n_window_widths
      integer, intent(in) :: n_ord_ep, n_ord_eq
      real(c_double), intent(in) :: nodes_ep(n_ord_ep), weights_ep(n_ord_ep)
      real(c_double), intent(in) :: nodes_eq(n_ord_eq), weights_eq(n_ord_eq)
      real(c_double) :: res

      real(c_double) :: ep_half, ep_mid, ep_i, ep_w_i
      real(c_double) :: eqr, width, eq_lo, eq_hi, eq_half, eq_mid, eq_i, eq_w_i, inner
      integer :: i, j

      ep_half = 0.5d0 * (ep_max - ep_min)
      ep_mid  = 0.5d0 * (ep_max + ep_min)

      res = 0.0d0
      do concurrent (i = 1:n_ord_ep) default(none) reduce(+: res) &
          shared(nodes_ep, weights_ep, nodes_eq, weights_eq, ep_half, ep_mid, &
                 eq_min, eq_max, n_window_widths, &
                 ep_tab, eq_tab, n_tab, pars, is_gamma, n_ord_ep, n_ord_eq) &
          local(ep_i, ep_w_i, eqr, width, eq_lo, eq_hi, eq_half, eq_mid, j, eq_i, eq_w_i, inner)

          ep_i   = ep_mid + ep_half * nodes_ep(i)
          ep_w_i = ep_half * weights_ep(i)

          call ridge_eq_and_width(ep_i, ep_tab, eq_tab, n_tab, pars, eqr, width)
          eq_lo = max(eq_min, eqr - n_window_widths * width)
          eq_hi = min(eq_max, eqr + n_window_widths * width)

          inner = 0.0d0
          if (eq_hi > eq_lo) then
              eq_half = 0.5d0 * (eq_hi - eq_lo)
              eq_mid  = 0.5d0 * (eq_hi + eq_lo)
              do j = 1, n_ord_eq
                  eq_i   = eq_mid + eq_half * nodes_eq(j)
                  eq_w_i = eq_half * weights_eq(j)
                  inner  = inner + eq_w_i * integrate_band(ep_i, eq_i, pars, is_gamma)
              end do
          end if

          res = res + ep_w_i * inner
      end do
  end function gl_region_pass

  ! Whether two successive doubling levels agree tightly enough to trust
  ! the finer one -- the auditable check that replaces Cuhre's own
  ! (demonstrably unreliable, at loose tolerances) internal heuristic.
  pure function gl_converged(a, b, epsrel, epsabs) result(ok)
      real(c_double), intent(in) :: a, b, epsrel, epsabs
      logical :: ok
      ok = abs(b - a) <= max(epsabs, epsrel * abs(b))
  end function gl_converged

  ! Locate where the ridge-following Eq window's lower edge crosses
  ! eq_min, or its upper edge crosses eq_max, as Ep sweeps [ep_min,
  ! ep_max] -- i.e. where the max()/min() clipping in gl_region_pass
  ! switches on or off.  A single global Gauss-Legendre rule converges
  ! poorly across that kink (measured: the user's own MCMC target region,
  ! which clips at low Ep, needed order 256 -- ~17s -- to converge without
  ! splitting; ~1s once split at the crossing).  This mirrors why
  ! band_breakpoints.py hands scipy.integrate.quad explicit breakpoints
  ! for the same shape of problem.  A monotone scan-then-bisect: the
  ! window's edges are expected to cross each bound at most once over the
  ! range (they track the monotone ridge), so a modest scan reliably
  ! brackets each crossing before refining it.
  pure subroutine find_window_breakpoints(ep_min, ep_max, eq_min, eq_max, &
      ep_tab, eq_tab, n_tab, pars, n_window_widths, brk, n_brk)
      real(c_double), intent(in) :: ep_min, ep_max, eq_min, eq_max
      integer, intent(in) :: n_tab
      real(c_double), intent(in) :: ep_tab(n_tab), eq_tab(n_tab)
      type(detector_params_t), intent(in) :: pars
      real(c_double), intent(in) :: n_window_widths
      real(c_double), intent(out) :: brk(2)
      integer, intent(out) :: n_brk

      integer, parameter :: n_scan = 200, n_bisect = 60
      real(c_double) :: h, ep_a, ep_b, g_a, g_b, ep_m, g_m, eqr, width
      integer :: i, it

      n_brk = 0
      h = (ep_max - ep_min) / real(n_scan - 1, c_double)

      ! Lower-edge crossing of eq_min
      ep_a = ep_min
      call ridge_eq_and_width(ep_a, ep_tab, eq_tab, n_tab, pars, eqr, width)
      g_a = (eqr - n_window_widths * width) - eq_min
      do i = 2, n_scan
          ep_b = ep_min + real(i - 1, c_double) * h
          call ridge_eq_and_width(ep_b, ep_tab, eq_tab, n_tab, pars, eqr, width)
          g_b = (eqr - n_window_widths * width) - eq_min
          if ((g_a < 0.0d0) .neqv. (g_b < 0.0d0)) then
              do it = 1, n_bisect
                  ep_m = 0.5d0 * (ep_a + ep_b)
                  call ridge_eq_and_width(ep_m, ep_tab, eq_tab, n_tab, pars, eqr, width)
                  g_m = (eqr - n_window_widths * width) - eq_min
                  if ((g_m < 0.0d0) .eqv. (g_a < 0.0d0)) then
                      ep_a = ep_m; g_a = g_m
                  else
                      ep_b = ep_m; g_b = g_m
                  end if
              end do
              n_brk = n_brk + 1
              brk(n_brk) = 0.5d0 * (ep_a + ep_b)
              exit
          end if
          ep_a = ep_b; g_a = g_b
      end do

      ! Upper-edge crossing of eq_max
      ep_a = ep_min
      call ridge_eq_and_width(ep_a, ep_tab, eq_tab, n_tab, pars, eqr, width)
      g_a = (eqr + n_window_widths * width) - eq_max
      do i = 2, n_scan
          ep_b = ep_min + real(i - 1, c_double) * h
          call ridge_eq_and_width(ep_b, ep_tab, eq_tab, n_tab, pars, eqr, width)
          g_b = (eqr + n_window_widths * width) - eq_max
          if ((g_a < 0.0d0) .neqv. (g_b < 0.0d0)) then
              do it = 1, n_bisect
                  ep_m = 0.5d0 * (ep_a + ep_b)
                  call ridge_eq_and_width(ep_m, ep_tab, eq_tab, n_tab, pars, eqr, width)
                  g_m = (eqr + n_window_widths * width) - eq_max
                  if ((g_m < 0.0d0) .eqv. (g_a < 0.0d0)) then
                      ep_a = ep_m; g_a = g_m
                  else
                      ep_b = ep_m; g_b = g_m
                  end if
              end do
              n_brk = n_brk + 1
              brk(n_brk) = 0.5d0 * (ep_a + ep_b)
              exit
          end if
          ep_a = ep_b; g_a = g_b
      end do
  end subroutine find_window_breakpoints

  ! Sum of gl_region_pass over a set of adjacent Ep sub-intervals
  ! (n_seg segments, ep_bounds(1:n_seg+1) the sorted boundaries) -- lets
  ! region_integral_gl split at the breakpoints found above while reusing
  ! gl_region_pass unchanged on each piece.
  pure function gl_multi_segment_pass(ep_bounds, n_seg, eq_min, eq_max, ep_tab, eq_tab, n_tab, &
      pars, is_gamma, n_window_widths, &
      nodes_ep, weights_ep, n_ord_ep, nodes_eq, weights_eq, n_ord_eq) result(res)
      integer, intent(in) :: n_seg
      real(c_double), intent(in) :: ep_bounds(n_seg + 1)
      real(c_double), intent(in) :: eq_min, eq_max
      integer, intent(in) :: n_tab
      real(c_double), intent(in) :: ep_tab(n_tab), eq_tab(n_tab)
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma
      real(c_double), intent(in) :: n_window_widths
      integer, intent(in) :: n_ord_ep, n_ord_eq
      real(c_double), intent(in) :: nodes_ep(n_ord_ep), weights_ep(n_ord_ep)
      real(c_double), intent(in) :: nodes_eq(n_ord_eq), weights_eq(n_ord_eq)
      real(c_double) :: res
      integer :: s

      res = 0.0d0
      do s = 1, n_seg
          res = res + gl_region_pass(ep_bounds(s), ep_bounds(s + 1), eq_min, eq_max, &
                                      ep_tab, eq_tab, n_tab, pars, is_gamma, n_window_widths, &
                                      nodes_ep, weights_ep, n_ord_ep, nodes_eq, weights_eq, n_ord_eq)
      end do
  end function gl_multi_segment_pass

  ! Reject parameter sets the physics model cannot evaluate, up front and
  ! loudly.  The resolution model sigp^2 = p0^2 + (p10^2 - p0^2)(Ep/c)^2
  ! (likewise sigq) goes negative at high energy whenever p10 < p0 (q10 <
  ! q0), and the resulting NaNs previously sent the integrator into its
  ! slowest fallback paths -- minutes per call instead of a fast failure --
  ! which is intolerable inside an MCMC where a bad proposal must just be
  ! rejected.  All checks are written as .not. (x > ...) so NaN inputs
  ! fail them too.
  pure subroutine validate_region_inputs(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      pars, is_gamma)
      real(c_double), intent(in) :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma

      if (.not. (ep_max > ep_min .and. eq_max > eq_min)) &
          error stop "PpqN_region/PpqG_region: invalid region: need ep_max > ep_min and eq_max > eq_min"
      if (.not. (epsrel >= 0.0d0 .and. epsabs >= 0.0d0)) &
          error stop "PpqN_region/PpqG_region: invalid tolerance: epsrel and epsabs must be >= 0"
      if (.not. (pars%F0 > 0.0d0)) &
          error stop "PpqN_region/PpqG_region: invalid parameters: Fano factor F0 must be > 0"
      if (.not. (pars%eps > 0.0d0)) &
          error stop "PpqN_region/PpqG_region: invalid parameters: eps must be > 0"
      if (.not. (1.0d0 + pars%V / (pars%eps * 1.0d3) > 0.0d0)) &
          error stop "PpqN_region/PpqG_region: invalid parameters: need 1 + V/(1000*eps) > 0"
      if (.not. (pars%p0 > 0.0d0 .and. pars%q0 > 0.0d0)) &
          error stop "PpqN_region/PpqG_region: invalid parameters: p0 and q0 must be > 0"
      if (.not. (pars%p10 >= pars%p0)) &
          error stop "PpqN_region/PpqG_region: invalid parameters: p10 < p0 makes the phonon resolution sigp^2 negative at high energy"
      if (.not. (pars%q10 >= pars%q0)) &
          error stop "PpqN_region/PpqG_region: invalid parameters: q10 < q0 makes the charge resolution sigq^2 negative at high energy"
      if (.not. is_gamma) then
          if (.not. (pars%k > 0.0d0)) &
              error stop "PpqN_region: invalid parameters: k must be > 0"
          if (.not. (pars%Zfac > 0.0d0 .and. pars%Zfac < huge(1.0d0))) &
              error stop "PpqN_region: invalid parameters: Z must be > 0"
      end if
  end subroutine validate_region_inputs

  ! Doubling-verified driver for PpqN_region/PpqG_region:
  ! order 32 vs 64, then 64 vs 128, then 128 vs 256; accepts the finer
  ! estimate the first time two successive orders agree within
  ! max(epsabs, epsrel*|result|), or fails loudly (error stop) if order
  ! 256 still hasn't converged rather than silently returning an
  ! unverified number -- see the module header comment above for why this
  ! replaced an earlier Cuba/Cuhre implementation.  The outer Ep range is
  ! first split at any window-clipping breakpoints (find_window_breakpoints)
  ! so each Gauss-Legendre pass only ever integrates a smooth piece.
  pure function region_integral_gl(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      pars, is_gamma) result(res)
      real(c_double), intent(in) :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
      type(detector_params_t), intent(in) :: pars
      logical, intent(in) :: is_gamma
      real(c_double) :: res

      real(c_double), parameter :: n_window_widths = 8.0d0
      integer, parameter :: n_tab = region_ridge_n_tab
      real(c_double) :: ep_tab(n_tab), eq_tab(n_tab)
      real(c_double) :: er_hi, brk(2), tmp
      integer :: n_brk, n_seg
      real(c_double) :: ep_bounds(4)
      real(c_double) :: res32, res64, res128, res256

      call validate_region_inputs(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, pars, is_gamma)

      er_hi = max(ep_max, eq_max) * 1.5d0 + 10.0d0
      call build_ridge_table(pars, is_gamma, er_hi, n_tab, ep_tab, eq_tab)

      call find_window_breakpoints(ep_min, ep_max, eq_min, eq_max, ep_tab, eq_tab, n_tab, &
                                    pars, n_window_widths, brk, n_brk)
      if (n_brk == 2 .and. brk(1) > brk(2)) then
          tmp = brk(1); brk(1) = brk(2); brk(2) = tmp
      end if
      select case (n_brk)
      case (0)
          n_seg = 1
          ep_bounds(1:2) = [ep_min, ep_max]
      case (1)
          n_seg = 2
          ep_bounds(1:3) = [ep_min, brk(1), ep_max]
      case default
          n_seg = 3
          ep_bounds(1:4) = [ep_min, brk(1), brk(2), ep_max]
      end select

      ! Tried pairing the inner Eq sweep with the next-LOWER order than
      ! the outer Ep sweep (32/16, 64/32, ...), reasoning that the
      ! ridge-centred window needs less resolution -- measured slower
      ! overall despite each level costing less, because the coarser Eq
      ! resolution made coarse/fine agreement harder to reach, forcing
      ! escalation to a higher Ep order than the matched-order scheme
      ! needed.  Kept both orders equal per level instead.
      res32 = gl_multi_segment_pass(ep_bounds, n_seg, eq_min, eq_max, ep_tab, eq_tab, n_tab, &
                                     pars, is_gamma, n_window_widths, &
                                     gl_nodes_32, gl_weights_32, 32, gl_nodes_32, gl_weights_32, 32)
      res64 = gl_multi_segment_pass(ep_bounds, n_seg, eq_min, eq_max, ep_tab, eq_tab, n_tab, &
                                     pars, is_gamma, n_window_widths, &
                                     gl_nodes_64, gl_weights_64, 64, gl_nodes_64, gl_weights_64, 64)
      if (gl_converged(res32, res64, epsrel, epsabs)) then
          res = res64
          return
      end if

      res128 = gl_multi_segment_pass(ep_bounds, n_seg, eq_min, eq_max, ep_tab, eq_tab, n_tab, &
                                      pars, is_gamma, n_window_widths, &
                                      gl_nodes_128, gl_weights_128, 128, gl_nodes_128, gl_weights_128, 128)
      if (gl_converged(res64, res128, epsrel, epsabs)) then
          res = res128
          return
      end if

      res256 = gl_multi_segment_pass(ep_bounds, n_seg, eq_min, eq_max, ep_tab, eq_tab, n_tab, &
                                      pars, is_gamma, n_window_widths, &
                                      gl_nodes_256, gl_weights_256, 256, gl_nodes_256, gl_weights_256, 256)
      if (gl_converged(res128, res256, epsrel, epsabs)) then
          res = res256
          return
      end if

      error stop "PpqN_region/PpqG_region: Gauss-Legendre doubling " // &
                  "did not converge to the requested epsrel/epsabs by order 256"
  end function region_integral_gl

end submodule PpqFort_s
