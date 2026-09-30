program profile_region
  use iso_c_binding, only : c_double
  use PpqFort_m, only : PpqN_region_adaptive, PpqG_region_adaptive
  implicit none

  ! Same physics parameters used throughout test/python/test_region_integral.py
  real(c_double), parameter :: k   = 0.18d0
  real(c_double), parameter :: Z   = 32.0d0
  real(c_double), parameter :: F0  = 0.122d0
  real(c_double), parameter :: eps = 3.0d-3
  real(c_double), parameter :: V   = 3.0d0
  real(c_double), parameter :: p0  = 0.06421907d0
  real(c_double), parameter :: p10 = 0.48998486d0
  real(c_double), parameter :: q0  = 0.23718488d0
  real(c_double), parameter :: q10 = 0.27093151d0

  ! The user's MCMC target region -- the case this profiling run cares about.
  real(c_double), parameter :: ep_min = 2.0d0, ep_max = 200.0d0
  real(c_double), parameter :: eq_min = 4.0d0, eq_max = 100.0d0
  real(c_double), parameter :: epsrel = 1.0d-4, epsabs = 1.0d-10

  integer, parameter :: n_iter = 20
  real(c_double) :: resN, resG, dummy
  integer :: iter
  integer(8) :: t_start, t_end, t_rate
  real(c_double) :: t_total

  call system_clock(t_start, t_rate)
  do iter = 1, n_iter
    resN = PpqN_region_adaptive(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
                                 k, Z, F0, eps, V, p0, p10, q0, q10)
  end do
  call system_clock(t_end)
  t_total = real(t_end - t_start, c_double) / real(t_rate, c_double)
  print '(a,f8.3,a,i0,a,f8.2,a)', &
      "PpqN_region_adaptive: ", t_total, " s for ", n_iter, " calls => ", &
      t_total / real(n_iter, c_double) * 1.0d3, " ms/call"
  print '(a,es14.6)', "  result: ", resN

  call system_clock(t_start, t_rate)
  do iter = 1, n_iter
    resG = PpqG_region_adaptive(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
                                 F0, eps, V, p0, p10, q0, q10)
  end do
  call system_clock(t_end)
  t_total = real(t_end - t_start, c_double) / real(t_rate, c_double)
  print '(a,f8.3,a,i0,a,f8.2,a)', &
      "PpqG_region_adaptive: ", t_total, " s for ", n_iter, " calls => ", &
      t_total / real(n_iter, c_double) * 1.0d3, " ms/call"
  print '(a,es14.6)', "  result: ", resG

  dummy = resN + resG
  if (dummy < 0.0d0) print *, "unreachable, keeps the compiler from dropping the calls"
end program profile_region
