subroutine sdvini(statev,coords,nstatv,ncrds,noel,npt,layer,kspt)
  integer, intent(in) :: nstatv, ncrds, noel, npt, layer, kspt
  real(kind=8), dimension(nstatv), intent(inout) :: statev
  real(kind=8), dimension(ncrds), intent(in) :: coords
  integer :: n
  real(kind=8) :: a
  n=nstatv; n=ncrds; n=noel; n=npt; n=layer; n=kspt
  a = coords(1)
  statev(:) = 0.e+00_8
end subroutine sdvini
