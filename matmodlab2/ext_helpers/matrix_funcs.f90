module linalg
  contains
function expm(a)
  ! ----------------------------------------------------------------------- !
  real(kind=8) :: expm(3,3)
  real(kind=8), intent(in) :: a(3,3)
  integer, parameter :: m=3, ldh=3, ideg=6, lwsp=4*m*m+ideg+1
  integer, parameter :: n=3, lwork=3*n-1
  real(kind=8) :: t, wsp(lwsp), v(3,3)
  real(kind=8) :: w(n), work(lwork), l(3,3)
  integer :: ipiv(m), iexph, ns, iflag, info
  character*120 :: msg
  ! ----------------------------------------------------------------------- !
  expm = 0.e+00_8
  if (all(abs(a) <= epsilon(a))) then
     expm = eye(3)
     return
  else if (isdiag(a)) then
     expm(1,1) = exp(a(1,1))
     expm(2,2) = exp(a(2,2))
     expm(3,3) = exp(a(3,3))
     return
  end if

  ! try dgpadm (usually good)
  t = 1.e+00_8
  iflag = 0
  call DGPADM(ideg, m, t, a, ldh, wsp, lwsp, ipiv, iexph, ns, iflag)
  if (iflag >= 0) then
     expm = reshape(wsp(iexph:iexph+m*m-1), shape(expm))
     return
  end if

  ! problem with dgpadm, use other method
  if (iflag == -8) then
     msg = '*** ERROR: bad sizes (in input of DGPADM)'
  else if (iflag == -9) then
     msg = '*** ERROR: Error - null H in input of DGPADM.'
  else if (iflag == -7) then
     msg = '*** ERROR: Problem in DGESV (within DGPADM)'
  end if
  print*, msg
  stop

  v = a
  call DSYEV("V", "L", 3, v, 3, w, work, lwork, info)
  l = 0.e+00_8
  l(1,1) = exp(w(1))
  l(2,2) = exp(w(2))
  l(3,3) = exp(w(3))
  expm = matmul(matmul(v, l ), transpose(v))
  return
end function expm

! ------------------------------------------------------------------------- !
function powm(a, m)
  ! ----------------------------------------------------------------------- !
  ! Computes the matrix power
  ! ----------------------------------------------------------------------- !
  real(kind=8) :: powm(3,3)
  real(kind=8), intent(in) :: a(3,3), m
  integer, parameter :: n=3, lwork=3*n-1
  real(kind=8) :: w(n), work(lwork), v(3,3), l(3,3)
  integer :: info
  ! eigenvalues/vectors of a
  v = a
  powm = 0.e+00_8
  if (isdiag(a)) then
     powm(1,1) = a(1,1) ** m
     powm(2,2) = a(2,2) ** m
     powm(3,3) = a(3,3) ** m
  else
     call DSYEV("V", "L", 3, v, 3, w, work, lwork, info)
     l = 0.e+00_8
     l(1,1) = w(1) ** m
     l(2,2) = w(2) ** m
     l(3,3) = w(3) ** m
     powm = matmul(matmul(v, l ), transpose(v))
  end if
  return
end function powm

! ------------------------------------------------------------------------- !
function sqrtm(a)
  ! ----------------------------------------------------------------------- !
  ! Computes the matrix sqrt
  ! ----------------------------------------------------------------------- !
  real(kind=8) :: sqrtm(3,3)
  real(kind=8), intent(in) :: a(3,3)
  integer, parameter :: n=3, lwork=3*n-1
  real(kind=8) :: w(n), work(lwork), v(3,3), l(3,3)
  integer :: info
  sqrtm = 0.e+00_8
  if (isdiag(a)) then
     sqrtm(1,1) = sqrt(a(1,1))
     sqrtm(2,2) = sqrt(a(2,2))
     sqrtm(3,3) = sqrt(a(3,3))
  else
     ! eigenvalues/vectors of a
     v = a
     call DSYEV("V", "L", 3, v, 3, w, work, lwork, info)
     l = 0.e+00_8
     l(1,1) = sqrt(w(1))
     l(2,2) = sqrt(w(2))
     l(3,3) = sqrt(w(3))
     sqrtm = matmul(matmul(v, l ), transpose(v))
  end if
  return
end function sqrtm

! ------------------------------------------------------------------------- !
function logm(a)
  ! ----------------------------------------------------------------------- !
  ! Computes the matrix logarithm
  ! ----------------------------------------------------------------------- !
  real(kind=8) :: logm(3,3)
  real(kind=8), intent(in) :: a(3,3)
  integer, parameter :: n=3, lwork=3*n-1
  real(kind=8) :: w(n), work(lwork), v(3,3), l(3,3)
  integer :: info
  if (isdiag(a)) then
     logm = 0.e+00_8
     logm(1,1) = log(a(1,1))
     logm(2,2) = log(a(2,2))
     logm(3,3) = log(a(3,3))
  else
     ! eigenvalues/vectors of a
     v = a
     call DSYEV("V", "L", 3, v, 3, w, work, lwork, info)
     l = 0.e+00_8
     l(1,1) = log(w(1))
     l(2,2) = log(w(2))
     l(3,3) = log(w(3))
     logm = matmul(matmul(v, l), transpose(v))
  end if
  return
end function logm

! ------------------------------------------------------------------------- !
function isdiag(a)
  logical :: isdiag
  real(kind=8), intent(in) :: a(3,3)
  isdiag = all(abs((/a(1,2),a(1,3),a(2,1),a(2,3),a(3,1),a(3,2)/)) <= epsilon(a))
  return
end function isdiag

function eye(n)
  integer, intent(in) :: n
  real(kind=8) :: eye(n,n)
  integer :: i
  eye = 0.e+00_8
  forall(i=1:n) eye(i,i) = 1.e+00_8
  return
end function eye

! ------------------------------------------------------------------------- !

function det(a)
  ! ----------------------------------------------------------------------- !
  ! determinant of 3x3
  ! ----------------------------------------------------------------------- !
  implicit none
  real(kind=8) :: det
  real(kind=8), intent(in) :: a(3,3)
  det = a(1,1)*a(2,2)*a(3,3)  &
       - a(1,1)*a(2,3)*a(3,2)  &
       - a(1,2)*a(2,1)*a(3,3)  &
       + a(1,2)*a(2,3)*a(3,1)  &
       + a(1,3)*a(2,1)*a(3,2)  &
       - a(1,3)*a(2,2)*a(3,1)
  return
end function det

! ------------------------------------------------------------------------- !
function inv(a)
  ! ----------------------------------------------------------------------- !
  ! inverse of 3x3
  ! ----------------------------------------------------------------------- !
  implicit none
  real(kind=8) :: inv(3,3)
  real(kind=8), intent(in)  :: a(3,3)
  real(kind=8) :: deta
  real(kind=8) :: cof(3,3)
  deta = det(a)
  if (abs(deta) .le. epsilon(deta)) then
     inv = 0.e+00_8
     stop "non-invertible matrix sent to inv"
  end if
  cof(1,1) = +(a(2,2)*a(3,3)-a(2,3)*a(3,2))
  cof(1,2) = -(a(2,1)*a(3,3)-a(2,3)*a(3,1))
  cof(1,3) = +(a(2,1)*a(3,2)-a(2,2)*a(3,1))
  cof(2,1) = -(a(1,2)*a(3,3)-a(1,3)*a(3,2))
  cof(2,2) = +(a(1,1)*a(3,3)-a(1,3)*a(3,1))
  cof(2,3) = -(a(1,1)*a(3,2)-a(1,2)*a(3,1))
  cof(3,1) = +(a(1,2)*a(2,3)-a(1,3)*a(2,2))
  cof(3,2) = -(a(1,1)*a(2,3)-a(1,3)*a(2,1))
  cof(3,3) = +(a(1,1)*a(2,2)-a(1,2)*a(2,1))
  inv = transpose(cof) / deta
  return
end function inv

subroutine polar_decomp(F, R, U, ierr)
  implicit none
  real(kind=8), intent(in) :: F(3,3)
  real(kind=8), intent(out) :: R(3,3), U(3,3)
  integer, intent(out) :: ierr
  real(kind=8) :: I(3,3)
  integer :: j
  I = eye(3)
  R = F
  ierr = 0
  do j = 1, 20
     R = .5 * matmul(R, 3. * I - matmul(transpose(R), R))
     if (maxval(abs(matmul(transpose(R), R) - I)) < 1.e-6_8) then
        U = matmul(transpose(R), F)
        return
     end if
  end do
  ierr = 1
end subroutine polar_decomp
end module linalg
