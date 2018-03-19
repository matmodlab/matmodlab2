User Model to Define a Material's Anisotropic Hyperelastic Response
###################################################################

Overview
========

*uanisohyper_inv* is a material model for anisotropic hyperelastic materials.

Interface
=========

.. code:: fortran

   subroutine uanisohyper_inv(ainv, u, zeta, nfibers, ninv, &
        ui1, ui2, ui3, temp, noel, cmname, incmpflag, ihybflag, &
	nstatev, statev, nfieldv, fieldv, fieldvinc, &
        nprops, props)
     implicit none
     integer, parameter :: dp=selected_real_kind(14)
     real(kind=dp), parameter :: zero=0._dp, one=1._dp, two=2._dp, three=3._dp
     character*8, intent(in) :: cmname
     integer, intent(in) :: nprops, noel, nstatev, incmpflag, nfieldv
     real(kind=dp), intent(in) :: ainv(ninv), props(nprops), temp
     real(kind=dp), intent(inout) ui1(ninv)
     real(kind=dp), intent(inout) :: ui2(ninv*(ninv+1)/2), ui3(ninv*(ninv+1)/2),
     real(kind=dp), intent(inout) :: statev(nstatev)
     real(kind=dp), intent(inout) :: fieldv(nfieldv), fieldvinc(nfieldv)

     ! User coding

   end subroutine uanisohyper_inv
