program profile_driver
  use iso_c_binding, only : c_double, c_int
  use PpqFort_m, only : PpqN_vector, PpqG_vector
  implicit none

  ! Parameters from test suite
  real(c_double), parameter :: k   = 0.18d0
  real(c_double), parameter :: Z   = 32.0d0
  real(c_double), parameter :: F0  = 0.122d0
  real(c_double), parameter :: eps = 3.0d-3
  real(c_double), parameter :: V   = 3.0d0
  real(c_double), parameter :: p0  = 0.06421907d0
  real(c_double), parameter :: p10 = 0.48998486d0
  real(c_double), parameter :: q0  = 0.23718488d0
  real(c_double), parameter :: q10 = 0.27093151d0

  integer, parameter :: npts = 100
  integer(c_int), parameter :: n = int(npts, c_int)
  integer, parameter :: n_iter = 5

  real(c_double) :: Ep_arr(npts), Eq_arr(npts)
  real(c_double) :: resN(npts), resG(npts)
  real(c_double) :: dummy, t_total, t_per_ms
  integer :: i, iter
  integer(8) :: t_start, t_end, t_rate

  ! Build a grid spanning the physically relevant range
  do i = 1, npts
    Ep_arr(i) = 300.0d0 + (100.0d0 * (i - 1)) / (npts - 1)  ! 300 to 400 eV
    Eq_arr(i) =  50.0d0 + (100.0d0 * (i - 1)) / (npts - 1)  !  50 to 150 eV
  end do

  ! --- PpqN timing ---
  call system_clock(t_start, t_rate)
  do iter = 1, n_iter
    call PpqN_vector(Ep_arr, Eq_arr, n, k, Z, F0, eps, V, p0, p10, q0, q10, resN)
  end do
  call system_clock(t_end)
  dummy    = sum(resN)
  t_total  = real(t_end - t_start, c_double) / real(t_rate, c_double)
  t_per_ms = t_total / real(npts * n_iter, c_double) * 1.0d3
  print '(a,f8.3,a,i0,a,f8.4,a)', &
      "PpqN: ", t_total, " s for ", npts*n_iter, " evals => ", t_per_ms, " ms/eval"
  print '(a,es14.6)', "  checksum: ", dummy

  ! --- PpqG timing ---
  call system_clock(t_start, t_rate)
  do iter = 1, n_iter
    call PpqG_vector(Ep_arr, Eq_arr, n, F0, eps, V, p0, p10, q0, q10, resG)
  end do
  call system_clock(t_end)
  dummy    = sum(resG)
  t_total  = real(t_end - t_start, c_double) / real(t_rate, c_double)
  t_per_ms = t_total / real(npts * n_iter, c_double) * 1.0d3
  print '(a,f8.3,a,i0,a,f8.4,a)', &
      "PpqG: ", t_total, " s for ", npts*n_iter, " evals => ", t_per_ms, " ms/eval"
  print '(a,es14.6)', "  checksum: ", dummy

end program profile_driver
