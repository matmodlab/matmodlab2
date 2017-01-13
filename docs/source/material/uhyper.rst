User Model to Define a Material's Hyperelastic Response
#######################################################

Overview
========

*uhyper* is a material model for hyperelastic materials.

Interface
=========

.. code:: fortran

   subroutine uhyper(i1b, i2b, jac, u, du, d2u, d3u, temp, noel, cmname, &
        incmpflag, nstatev, statev, nfieldv, fieldv, fieldvinc, &
        nprops, props)
     implicit none
     integer, parameter :: dp=selected_real_kind(14)
     real(kind=dp), parameter :: zero=0._dp, one=1._dp, two=2._dp, three=3._dp
     character*8, intent(in) :: cmname
     integer, intent(in) :: nprops, noel, nstatev, incmpflag, nfieldv
     real(kind=dp), intent(in) :: i1b, i2b, jac, props(nprops), temp
     real(kind=dp), intent(inout) :: u(2), du(3), d2u(6), d3u(6), statev(nstatev)
     real(kind=dp), intent(inout) :: fieldv(nfieldv), fieldvinc(nfieldv)

     ! User coding

   end subroutine uhyper
