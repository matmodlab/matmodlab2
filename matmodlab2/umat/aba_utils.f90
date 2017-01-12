subroutine rotsig(S, R, Sprime, lstr, ndi, nshr)
  implicit none
  integer, intent(in) :: lstr, ndi, nshr
  real(8), intent(in) :: S(6), R(3,3)
  real(8), intent(out) :: Sprime(6)
  real(8) :: work(2,3,3), c

  ! NOTE: this subroutine is set up to work only with 3D rotations
  if (ndi /= 3 .or. nshr /= 3) then
     print*, 'rotsig incompatible with tensors'
     call xit
  end if

  c = 1.
  if (lstr == 2) c = 2.

  ! convert to matrix
  call to_matrix_3d(S, work(1,:,:), 1./c)

  ! perform the rotation
  work(2,:,:) = matmul(R, matmul(work(1,:,:), transpose(R)))

  ! convert to tensor
  call to_tensor_3d(work(2,:,:), Sprime, c)

  return

end subroutine rotsig

subroutine to_tensor_3d(x, t, c)
  implicit none
  real(8), intent(in) :: x(3,3)
  real(8), intent(out) :: t(6)
  real(8), intent(in) :: c
  t(1) = x(1,1); t(4) = x(1,2)*c; t(5) = x(1,3)*c
                 t(2) = x(2,2);   t(6) = x(2,3)*c
                                  t(3) = x(3,3)
  return
end subroutine to_tensor_3d

subroutine to_matrix_3d(x, m, c)
  implicit none
  real(8), intent(in) :: x(6)
  real(8), intent(out) :: m(3,3)
  real(8), intent(in) :: c
  m(1,1) = x(1); m(1,2) = x(4)*c; m(1,3) = x(5)*c
                 m(2,2) = x(2);   m(2,3) = x(6)*c
                                  m(3,3) = x(3)
  m(2,1) = m(1,2)
  m(3,1) = m(1,3)
  m(3,2) = m(2,3)
  return
end subroutine to_matrix_3d

subroutine stdb_abqerr(ierr, msg, intv, realv, charv)
  implicit none
  integer, intent(in) :: ierr
  character(120), intent(in) :: msg
  integer, intent(in) :: intv(*)
  real(8), intent(in) :: realv(*)
  character(8), intent(in) :: charv(*)
  call mml_comm(ierr, msg, intv, realv, charv)
end subroutine stdb_abqerr

subroutine xit
  implicit none
  character*120 :: msg
  external log_error
  msg = 'STOPPING DUE TO FORTRAN PROCEDURE ERROR'
  call log_error(msg)
end subroutine xit

subroutine sprind(S, PS, AN, LSTR, NDI, NSHR)
  implicit none
  real(8), intent(in) :: S(6)
  real(8), intent(out) :: PS(3), AN(3,3)
  integer, intent(in) :: LSTR, NDI, NSHR
  integer, parameter :: N=3, LWORK=3*N-1
  real(8) :: WORK(LWORK), C
  integer :: INFO
  ! NOTE: this subroutine is set up to work only with 3D tensors
  if (NDI /= 3 .or. NSHR /= 3) then
     print *, 'wrong number of direct and/or shear components to sprind'
     call xit
  end if
  ! convert to matrix
  C = 1.
  if (LSTR == 2) C = 2.
  call to_matrix_3d(S, AN, 1./C)
  ! eigenvalues/vectors of S
  call DSYEV("V", "L", 3, AN, 3, PS, WORK, LWORK, INFO)
end subroutine sprind

subroutine sprinc(S, PS, LSTR, NDI, NSHR)
  implicit none
  real(8), intent(in) :: S(6)
  real(8), intent(out) :: PS(3)
  integer, intent(in) :: LSTR, NDI, NSHR
  real(8) :: AN(3,3)
  call sprind(S, PS, AN, LSTR, NDI, NSHR)
end subroutine sprinc
