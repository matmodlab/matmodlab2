
.. _fortran_models:

Fortran User Material Interface
###############################

.. topic:: See Also

   * :ref:`Role of Material Model`
   * :ref:`defining_a_material`
   * :ref:`intro_conventions`
   * :ref:`comm_w_matmodlab`
   * :ref:`sdvini`

Overview
========

Procedures ``UMAT``, ``UHYPER``, and ``UANISOHYPER_INV`` are called for user defined materials defining the mechanical, hyperelastic, or anisotropic hyperelastic material responses, respectively.  Regardles of the interface procedure used, a fortran compiler must be available for Matmodlab to compile and link user procedures.

.. _invoke_user_f:

Invoking User Materials
=======================

User defined materials are invoked using the same
``MaterialPointSimulator.Material`` factory method as other materials, but
with additional required and optional arguments.

Required MaterialPointSimulator.Material Agruments
--------------------------------------------------

* The *model* argument must be set to one of ``USER``, ``UMAT``, ``UHYPER``, or ``UANISOHYPER_INV``.
* The *parameters* must be a ndarray of model constants (specified in the
  order expected by the model).
* *source_files*, a list of model source files. The source files must exist
  and be readable on the file system.

Optional MaterialPointSimulator.Material Arguments
--------------------------------------------------

* *param_names*, is a list of parameter names in the order expected by the model.
  If given, *parameters* must be given as dict of ``name:value`` pairs as for
  builtin models.
* *depvar*, is the number of state dependent variables required
  for the model. Can also be specified as a list of state dependent variable
  names, specified in the order expected by the model. If given as a list, the
  number of state variables allocated is inferred from its length. Matmodlab
  allocates storage for the *depvar* state dependent variables and
  initializes their values to 0.
* *response*, is a string specifying the type of model.  Must be one of ``MECHANICAL`` (default), ``HYPERELASTIC``, or ``ANISOHYPER``.
* *cmname*, is a string giving is the constitutive model name.
* *ordering*, is a list of symbolic constants specifying the ordering of
  second-order symmetric tensors. The default ordering of symmetric
  second-order tensor components is ``[XX, YY, ZZ, XY, YZ, XZ]``. The *ordering*
  argument can be used to change the ordering to be consistent with the
  assumptions of the material model.

Example
-------

::

   mps = MaterialPointSimulator('user_material')
   parameters = np.array([135e9, 53e9, 200e6])
   mps.Material(USER, parameters)


.. topic:: Abaqus Users:

   Setting the *model* name to one of ``UMAT``, ``UHYPER``, or
   ``UANISOHYPER_INV`` is equivalient to *model=*\ ``USER``, with
   *response=*\ ``MECHANICAL``, *response=*\ ``HYPERELASTIC``, or
   *response=*\ ``ANISOHYPER``, respectively, and *ordering=*\ ``[XX, YY,
   ZZ, XY, XZ, YZ]``.

Compiling Fortran Sources
=========================

Matmodlab compiles and links material model sources using ``f2py``.

User Subroutine Interfaces
==========================

.. toctree::
   :maxdepth: 1

   umat
   uhyper
   uanisohyper_inv
