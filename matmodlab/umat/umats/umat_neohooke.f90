! --------------------------------------------------------------------------- !
!    UMAT FOR NEO HOOKEAN HYPERELASTICITY
!    CANNOT BE USED FOR PLANE STRESS
!
!    PROPS(1) : Young's modulus
!    PROPS(2) : Poisson's ratio
! --------------------------------------------------------------------------- !
subroutine umat(stress, statev, ddsdde, sse, spd, scd, rpl, &
     ddsddt, drplde, drpldt, stran, dstran, time, dtime, temp, dtemp, &
     predef, dpred, cmname, ndi, nshr, ntens, nstatv, props, nprops, &
     coords, drot, pnewdt, celent, dfgrd0, dfgrd1, noel, npt, layer, &
     kspt, kstep, kinc)

  implicit none
  integer, parameter :: dp=selected_real_kind(14)
  character*8, intent(in) :: cmname
  integer, intent(in) :: ndi, nshr, ntens, nstatv, nprops
  integer, intent(in) :: noel, npt, layer, kspt, kstep, kinc
  real(kind=dp), intent(in) :: sse, spd, scd, rpl, drpldt, time, dtime, temp, dtemp
  real(kind=dp), intent(in) :: pnewdt, celent
  real(kind=dp), intent(inout) :: stress(ntens), statev(nstatv), ddsdde(ntens, ntens)
  real(kind=dp), intent(inout) :: ddsddt(ntens), drplde(ntens)
  real(kind=dp), intent(in) :: stran(ntens), dstran(ntens)
  real(kind=dp), intent(in) :: predef(1), dpred(1), props(nprops), coords(3)
  real(kind=dp), intent(in) :: drot(3, 3), dfgrd0(3, 3), dfgrd1(3, 3)!

  integer :: i, j
  real(kind=dp) :: ee(6), eep(3), bbp(3), bbn(3,3)
  real(kind=dp) :: emod, enu, c10, d1, eg, ek, eg23, pr
  real(kind=dp) :: jac, scale, fb(3,3), bb(6), trbbar
  character*120 :: msg
  character*8 :: charv(1)
  integer :: intv(1)
  real(kind=dp) :: realv(1)
  real(kind=dp), parameter :: I6(6)=[1.,1.,1.,0.,0.,0.]
  ! ------------------------------------------------------------------------- !

  if (ndi /= 3) then
     msg = 'this umat may only be used for elements &
          &with three direct stress components'
     call stdb_abqerr(-3, msg, intv, realv, charv)
  end if

  ! elastic properties
  emod = props(1)
  enu = props(2)
  c10 = emod / (4. * (1. + enu))
  d1 = 6. * (1. - 2. * enu) / emod

  ! jacobian and distortion tensor
  jac = dfgrd1(1,1)*dfgrd1(2,2)*dfgrd1(3,3) - dfgrd1(1,2)*dfgrd1(2,1)*dfgrd1(3,3) &
      + dfgrd1(1,2)*dfgrd1(2,3)*dfgrd1(3,1) + dfgrd1(1,3)*dfgrd1(3,2)*dfgrd1(2,1) &
      - dfgrd1(1,3)*dfgrd1(3,1)*dfgrd1(2,2) - dfgrd1(2,3)*dfgrd1(3,2)*dfgrd1(1,1)
  scale = jac ** (-1. / 3.)
  fb = scale * dfgrd1

  ! deviatoric left cauchy-green deformation tensor
  bb(1) = fb(1,1) * fb(1,1) + fb(1,2) * fb(1,2) + fb(1,3) * fb(1,3)
  bb(2) = fb(2,1) * fb(2,1) + fb(2,2) * fb(2,2) + fb(2,3) * fb(2,3)
  bb(3) = fb(3,1) * fb(3,1) + fb(3,2) * fb(3,2) + fb(3,3) * fb(3,3)
  bb(4) = fb(2,1) * fb(1,1) + fb(2,2) * fb(1,2) + fb(2,3) * fb(1,3)
  bb(5) = fb(3,1) * fb(2,1) + fb(3,2) * fb(2,2) + fb(3,3) * fb(2,3)
  bb(6) = fb(3,1) * fb(1,1) + fb(3,2) * fb(1,2) + fb(3,3) * fb(1,3)
  trbbar = sum(bb(1:3)) / 3.
  eg = 2. * c10 / jac
  ek = 2. / d1 * (2. * jac - 1.)
  pr = 2. / d1 * (jac - 1.)

  ! cauchy stress
  stress = eg * (bb - trbbar * I6) + pr * I6

  ! spatial stiffness
  eg23 = eg * 2. / 3.
  ddsdde(1,1) =  eg23 * (bb(1) + trbbar) + ek
  ddsdde(1,2) = -eg23 * (bb(1) + bb(2)-trbbar) + ek
  ddsdde(1,3) = -eg23 * (bb(1) + bb(3)-trbbar) + ek
  ddsdde(1,4) =  eg23 * bb(4) / 2.
  ddsdde(1,5) = -eg23 * bb(5)
  ddsdde(1,6) =  eg23 * bb(6) / 2.

  ddsdde(2,2) =  eg23 * (bb(2) + trbbar) + ek
  ddsdde(2,3) = -eg23 * (bb(2) + bb(3)-trbbar) + ek
  ddsdde(2,4) =  eg23 * bb(4) / 2.
  ddsdde(2,5) =  eg23 * bb(5) / 2.
  ddsdde(2,6) = -eg23 * bb(6)

  ddsdde(3,3) =  eg23 * (bb(3) + trbbar) + ek
  ddsdde(3,4) = -eg23 * bb(4)
  ddsdde(3,5) =  eg23 * bb(5) / 2.
  ddsdde(3,6) =  eg23 * bb(6) / 2.

  ddsdde(4,4) =  eg * (bb(1) + bb(2)) / 2.
  ddsdde(4,5) =  eg * bb(6) / 2.
  ddsdde(4,6) =  eg * bb(5) / 2.

  ddsdde(5,5) =  eg * (bb(1) + bb(3)) / 2.
  ddsdde(5,6) =  eg * bb(4) / 2.

  ddsdde(6,6) =  eg * (bb(2) + bb(3)) / 2.
  forall(i=1:ntens,j=1:ntens,j<i) ddsdde(i,j) = ddsdde(j,i)

  ! ! logarithmic strains
  ! call sprind(bb, bbp, bbn, 1, 3, 3)
  ! eep(1) = log(sqrt(bbp(1)) / scale)
  ! eep(2) = log(sqrt(bbp(2)) / scale)
  ! eep(3) = log(sqrt(bbp(3)) / scale)
  ! ee(1) = eep(1) * bbn(1,1) ** 2 + eep(2) * bbn(2,1) ** 2 &
  !          + eep(3) * bbn(3,1) ** 2
  ! ee(2) = eep(1) * bbn(1,2) ** 2 + eep(2) * bbn(2,2) ** 2 &
  !          + eep(3) * bbn(3,2) ** 2
  ! ee(3) = eep(1) * bbn(1,3) ** 2 + eep(2) * bbn(2,3) ** 2 &
  !          + eep(3) * bbn(3,3) ** 2
  ! ee(4) = 2. * (eep(1) * bbn(1,1) * bbn(1,2) &
  !          + eep(2) * bbn(2,1) * bbn(2,2) &
  !          + eep(3) * bbn(3,1) * bbn(3,2))
  ! ee(5) = 2. * (eep(1) * bbn(1,1) * bbn(1,3) &
  !          + eep(2) * bbn(2,1) * bbn(2,3) &
  !          + eep(3) * bbn(3,1) * bbn(3,3))
  ! ee(6) = 2. * (eep(1) * bbn(1,2) * bbn(1,3) &
  !          + eep(2) * bbn(2,2) * bbn(2,3) &
  !          + eep(3) * bbn(3,2) * bbn(3,3))
  ! statev(1:ntens) = ee(k1)
  return
end subroutine umat
