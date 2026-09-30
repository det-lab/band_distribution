module PpqFort_m
  !! TODO: Add comment for FORD describing the module's purpose
  use iso_c_binding, only : c_double, c_int
  implicit none

  interface

    pure module subroutine PpqN_vector(Ep_arr, Eq_arr, n, k, Z, F0, eps, V, p0, p10, q0, q10, res_arr) bind(c, name="PpqN_vector")
      !! TODO: Add comment for FORD describing the function's purpose
      integer(c_int), value :: n
      real(c_double), intent(in) :: Ep_arr(n), Eq_arr(n)
      real(c_double), value :: k, Z, F0, eps, V, p0, p10, q0, q10
      real(c_double), intent(out) :: res_arr(n)
    end subroutine PpqN_vector

    pure module subroutine PpqG_vector(Ep_arr, Eq_arr, n, F0, eps, V, p0, p10, q0, q10, res_arr) bind(c, name="PpqG_vector")
      !! TODO: Add comment for FORD describing the function's purpose
      integer(c_int), value :: n
      real(c_double), intent(in) :: Ep_arr(n), Eq_arr(n)
      real(c_double), value :: F0, eps, V, p0, p10, q0, q10
      real(c_double), intent(out) :: res_arr(n)
    end subroutine PpqG_vector

    pure module function PpqN(Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="PpqN")
      !! TODO: Add comment for FORD describing the function's purpose
      real(c_double), value :: Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10
      real(c_double) :: res
    end function PpqN

    pure module function PpqG(Ep, Eq, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="PpqG")
      !! TODO: Add comment for FORD describing the function's purpose
      real(c_double), value :: Ep, Eq, F0, eps, V, p0, p10, q0, q10
      real(c_double) :: res
    end function PpqG

   pure module function Y(Er, k, Z) result(res) bind(c, name="Y")
       !! the ionization yield, using the Lindhard model
       !! Y(Er) = k*g(eps_L) / (1 + k*g(eps_L)), with
       !! eps_L = 11.5*Er*Z**(-7/3) and g(eps_L) = 3*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L
       real(c_double), value :: Er, k, Z
       real(c_double) :: res
    end function Y

   pure module function Nbar(Er, k, Z, eps) result(res) bind(c, name="Nbar")
     !! average number of electron-hole pairs for a given Er
     real(c_double), value :: Er, k, Z, eps
     real(c_double) :: res
   end function Nbar

   pure module function sigp(Ep, eps, V, p0, p10) result(res) bind(c, name="sigp")
      !! phonon sensor resolution
      real(c_double), value :: Ep, eps, V, p0, p10
      real(c_double) :: res
   end function sigp

   pure module function sigq(Eq, q0, q10) result(res) bind(c, name="sigq")
   !! charge sensor resolution
      real(c_double), value :: Eq, q0, q10
      real(c_double) :: res
   end function sigq

  pure module function PErN(Er) result(res) bind(c, name="PErN")
    !! the Er (energy) distribution of neutrons
    real(c_double), value :: Er
    real(c_double) :: res
  end function PErN

  pure module function PErG(Er) result(res) bind(c, name="PErG")
    !! the Er (energy) distribution of gamma events/electron recoils
    real(c_double), value :: Er
    real(c_double) :: res
  end function PErG

  pure module function PpqFullN(Er, Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="PpqFullN")
    ! Input arguments
    real(c_double), value :: Er, Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function PpqFullN

  pure module function PpqN_region(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      k, Z, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="PpqN_region")
    !! Integral of PpqN over the rectangle [ep_min,ep_max] x [eq_min,eq_max],
    !! e.g. for normalizing a likelihood to its fit region.  Computed with
    !! a doubling-verified nested Gauss-Legendre quadrature: the ridge
    !! location/width machinery (ridge_eq_and_width in PpqFort_s.f90)
    !! already tells this exactly where the band is, so it needs far
    !! fewer points than a naive fixed grid for the same accuracy.
    !! epsrel/epsabs set how tightly two successive doubled quadrature
    !! orders (32 vs 64, 64 vs 128, 128 vs 256) must agree before the
    !! result is trusted: this stops refining once
    !! |result_2N - result_N| <= max(epsabs, epsrel*|result_2N|), and
    !! error-stops rather than returning an unverified number if order 256
    !! still hasn't converged.  Parameter sets the model cannot evaluate
    !! (p10 < p0 or q10 < q0, which make the resolution variance negative
    !! at high energy; F0, eps, p0, q0, k, Z <= 0; empty region; NaN
    !! inputs) error-stop immediately instead of grinding.
    real(c_double), value :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
    real(c_double), value :: k, Z, F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function PpqN_region

  pure module function PpqG_region(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="PpqG_region")
    !! Same as PpqN_region but for PpqG (electron-recoil band, Y=1).
    real(c_double), value :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
    real(c_double), value :: F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function PpqG_region

  pure module subroutine PpqFort_version(major, minor, patch) bind(c, name="PpqFort_version")
    !! Report the package version (see fpm.toml's version field, which
    !! must be kept in sync by hand)
    integer(c_int), intent(out) :: major, minor, patch
  end subroutine PpqFort_version

  end interface

  ! Define pi
  real(c_double), parameter :: pi = 3.14159265358979323846_c_double
    ! TODO: move the pi definition to just above the submodule's "contains" statement

  ! Package version (semver).  Keep in sync with fpm.toml's version field.
  integer(c_int), parameter :: version_major = 1
  integer(c_int), parameter :: version_minor = 1
  integer(c_int), parameter :: version_patch = 0

end module PpqFort_m
