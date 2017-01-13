.. _umat:

User Model to Define a Material's Mechanical Response
#####################################################

Overview
========

``UMAT`` is a material model that completely describes the material behavior.

Interface
=========

.. code:: fortran

   subroutine umat(stress,statev,ddsdde,sse,spd,scd,rpl,ddsddt,drplde,drpldt,&
        stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname, ndi,nshr,ntens,&
        nstatv,props,nprops,fiber_dir,nfibers,drot,pnewdt,celent,f0,f1,noel,npt,&
        layer,kspt,kstep,kinc)
     implicit none

     character*8, intent(in) :: cmname
     integer, intent(in) :: ndi, nshr, ntens, nstatv, nprops, nfibers
     integer, intent(in) :: noel, npt, layer, kspt, kstep, kinc
     integer, parameter :: dp=selected_real_kind(14)
     real(kind=dp), intent(in) :: sse, spd, scd, rpl, drpldt, time(2), dtime
     real(kind=dp), intent(in) :: temp, dtemp, pnewdt, celent
     real(kind=dp), intent(inout) :: stress(ntens), statev(nstatv)
     real(kind=dp), intent(inout) :: ddsdde(ntens, ntens)
     real(kind=dp), intent(inout) :: ddsddt(ntens), drplde(ntens)
     real(kind=dp), intent(in) :: stran(ntens), dstran(ntens)
     real(kind=dp), intent(in) :: predef(1), dpred(1)
     real(kind=dp), intent(in) :: props(nprops),fiber_dir(nfibers,3)
     real(kind=dp), intent(in) :: drot(3, 3), f0(3, 3), f1(3, 3)

   ! User coding

   end subroutine umat

Output Only Variables
=====================

*ddsdde(6,6)*

   The material stiffness :math:`\partial\Delta\sigma / \partial\Delta\epsilon`

Input and Output Variables
==========================

*stress(6)*

   On input, the stress at the beginning of the increment.  On output, the stress at the end of the increment.

*statev(nstatv)*

   On ipnut, the values of the state dependent variables at the beginning of the increment.  On output, their values at the end of the increment.

Input Only Variables
====================

*stran(ntens)*

   The strain at the end of the increment.  The definition of strain depends on the value of the user input kappa. If thermal expansion is included, the strains passed are the mechanical strains only.

*dstran(ntens)*

   The strain increments. If thermal expansion is included, the strain increments passed are the mechanical strain increments only.

*time(1)*

   The step time at the beginning of the current increment

*time(2)*

   The total time at the beginning of the current increment

*dtime*

   Time increment

*temp*

   The temperature at the beginning of the increment.

*dtemp*

   The temperature increment.

*ndi*

   Number of direct stress components.  Always set to 3.

*nshr*

   Number of shear stress components.  Always set to 3.

*ntens*

   Size of the stress tensor.  Always set to 6.

*nstatv*

   Number of state dependent state variables.

*props(nprops)*

   The material property array.

*nprops*

   The number of material properties.

F0(3,3)

   The deformation gradient at the beginning of the increment.

F1(3,3)

   The deformation gradient at the end of the increment.

Other Variables
===============

The other variables in the ``umat`` definition are present to be consistent with popular commercial finite element codes but are not used by Matmodlab.

Example
=======

The following is an example of a linear elastic user defined material.

.. code:: fortran

   ! --------------------------------------------------------------------------- !
   subroutine umat(stress, statev, ddsdde, sse, spd, scd, rpl, &
        ddsddt, drplde, drpldt, stran, dstran, time, dtime, temp, dtemp, &
        predef, dpred, cmname, ndi, nshr, ntens, nstatv, props, nprops, &
        coords, drot, pnewdt, celent, dfgrd0, dfgrd1, noel, npt, layer, &
        kspt, kstep, kinc)

     implicit none
     character*8, intent(in) :: cmname
     integer, intent(in) :: ndi, nshr, ntens, nstatv, nprops
     integer, intent(in) :: noel, npt, layer, kspt, kstep, kinc
     real(8), intent(in) :: sse, spd, scd, rpl, drpldt, time, dtime, temp, dtemp
     real(8), intent(in) :: pnewdt, celent
     real(8), intent(inout) :: stress(ntens), statev(nstatv), ddsdde(ntens, ntens)
     real(8), intent(inout) :: ddsddt(ntens), drplde(ntens)
     real(8), intent(in) :: stran(ntens), dstran(ntens)
     real(8), intent(in) :: predef(1), dpred(1), props(nprops), coords(3)
     real(8), intent(in) :: drot(3, 3), dfgrd0(3, 3), dfgrd1(3, 3)!

     integer :: i, j
     real(8) :: K, K3, G, G2, Lam
     character*120 :: msg
     character*8 :: charv(1)
     integer :: intv(1)
     real(8) :: realv(1)
     ! ------------------------------------------------------------------------- !

     if (ndi /= 3) then
        msg = 'this umat may only be used for elements &
             &with three direct stress components'
        call stdb_abqerr(-3, msg, intv, realv, charv)
     end if

     ! elastic properties
     K = props(1)
     K3 = 3. * K
     G = props(2)
     G2 = 2. * G
     Lam = (K3 - G2) / 3.

     ! elastic stiffness
     ddsdde = 0.
     do i=1,ndi
        do j = 1,ndi
           ddsdde(j,i) = Lam
        end do
        ddsdde(i,i) = G2 + Lam
     end do
     do i=ndi+1,ntens
        ddsdde(i,i) = G
     end do

     ! stress update
     stress = stress + matmul(ddsdde, dstran)

     return
   end subroutine umat
