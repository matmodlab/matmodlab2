
.. _sdvini:

User Defined Initial Conditions
###############################

.. topic:: See Also

   * :ref:`mat_overview`

Overview
========

Subroutine ``SDVINI`` allows users to set initial values to state dependent variables.  If not provided by the user, Matmodlab will set state dependent to ``0``.

Interface
=========

.. code:: fortran

   subroutine sdvini(statev,coords,nstatv,ncrds,noel,npt,layer,kspt)
      integer, intent(in) :: nstatv, ncrds, noel, npt, layer, kspt
      real(kind=8), dimension(nstatv), intent(inout) :: statev
      real(kind=8), dimension(ncrds), intent(in) :: coords

      ! user code

   end subroutine sdvini
