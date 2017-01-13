.. _user_mats:

User Defined Materials
######################

.. topic:: See Also

   * :ref:`Role of Material Model`
   * :ref:`defining_a_material`
   * :ref:`intro_conventions`

Overview
========

Matmodlab calls user defined materials at each deformation increment through all analysis steps.  User defined materials are

* written in fortran or python
* called from python

Material Model API
==================

Two API's are provided for interfacing user developed materials with Matmodlab:

* :ref:`Python API <python_models>`. Material models are implemented as subclasses of MaterialModel,
* :ref:`Fortran API <fortran_models>`. Similar to commercial finite element codes, material models are implemented in Fortran subroutines that are compiled, linked, and called by Matmodlab.

Auxiliary Subroutines
=====================

Fortran subroutines can set initial conditions and interact with Matmodlab as described in the following sections:

* :ref:`sdvini`
* :ref:`comm_w_matmodlab`

.. toctree::
   :hidden:

   python
   fortran
   comm
   sdvini
