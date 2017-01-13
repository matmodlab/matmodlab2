.. _mps:

MaterialPointSimulator
######################

.. topic:: See Also

   * :ref:`intro_conventions`
   * :ref:`mml_out_dbs`
   * :ref:`mat_builtin`
   * :ref:`user_mats`

Overview
========

A model in Matmodlab is defined by a ``MaterialPointSimulator`` object. The
``MaterialPointSimulator`` object manages and allocates memory for materials and
analysis steps. In this section, the ``MaterialPointSimulator`` object is
described.

The MaterialPointSimulator Constructor
======================================

.. class:: MaterialPointSimulator(job, verbosity=1, d=None, inital_temperature=DEFAULT_TEMP, output=DBX)

   Create a MaterialPointSimulator object and set up the simulation.

   The ``job`` string is the simulation ID.  Generated files are named ``job.ext``, where ext is the file extension.

   The following arguments are optional.

   The ``verbosity`` integer set the simulation verbosity. Generally, 0=quiet, 2=noisy.  The ``d`` string is the simulation directory, the default is the working directory.  ``initial_temperature`` is the initial temperature, the default is ``298`` K.  The ``output`` symbolic constant specifies the output type, it defaults to ``DBX`` if not specified.  See :ref:`mml_out_dbs` for a description of supported output formats.

   Examples::

     >>> mps = MaterialPointSimulator('model')

     >>> mps = MaterialPointSimulator('model', verbosity=2, d='/home/user/sim')


.. _defining_a_material:

Defining a Material Model
=========================

.. method:: MaterialPointSimulator.Material(model, parameters, name=None, switch=None, rebuild=False, source_files=None, depvar=None, fiber_dirs=None, user_ics=False, order=None, response=None, libname=None, param_names=None)

   Create and assign a material model

   The required arguments are a model name and material parameters.  The model name must be a recognized material model (see :ref:`mat_index`).  ``parameters`` is either a dictionary of ``key:value`` (``key`` being the parameter name, ``value`` its numeric value) or ndarray.

   The following arguments are optional and applicable to all materials.

   ``rebuild`` is a boolean that, when ``True``, forces the material model to be rebuilt before the simulation.  ``switch`` is a tuple containing the material name and the name of another material to be switched in to its place.

   **The following arguments are applicable to user materials:**

   A user material is invoked by setting ``model`` to one of ``USER``, ``UMAT``, ``UHYPER``, ``UANISOHYPER_INV`` (see :ref:`user_mats` for details).

   ``source_files`` is a list of model source files.  Each file must exist and be readable on the file system.  ``depvar`` is either the integer number of state dependent variables or a list of state dependent variable names. ``fiber_dirs`` is an array of fiber directions (applicable only to uanisohyper_inv models). ``param_names`` is a list of parameter names. If ``user_ics`` is ``True``, Matmodlab calls the user supplied ``SDVINI`` subroutine to initialize state dependent variables - otherwise they are set to ``0``.  ``order`` is a list of strings specifying the component ordering of second order tensors.  ``response`` is one of ``MECHANICAL``, ``HYPERELASTIC``, or ``ANISOHYPER`` and is used to determine which type of response the model will describe.

   Examples::

     >>> mps.Material('elastic', {'K': 123e9, 'G': 53e9})

     >>> mps.Material(UMAT, (10e6, .333, 42e3),
                      source_files=('umat.f', 'umat.pyf'),
		      param_names=('E', 'nu', 'Y'), user_ics=True,
		      depvar=('EQPS',))

Optional Material Addons
------------------------

.. _expan_model:

Thermal Expansion
.................

A :ref:`expan_model` object, enabling the computation of thermal strains associated with thermal expansion.

.. _visco_model:

Viscoelasticity
...............

A :ref:`visco_model` object defining the linear relaxation response of the material.  When given, the elastic moduli are treated as the instantaneous values.

.. _trs_model:

Time-Temperature Shift
......................

Used in conjuction with a :ref:`visco_model` to compute a reduced time.


Defining Simulation Steps
=========================

The recommended way to create simulation steps is to use the following convenience functions.


.. method:: MaterialPointSimulator.StrainStep(*)

   All step components are interpreted as components of the strain tensor.

   The arguments represented by the * are common to all other step methods and are described in :ref:`common_args`.

.. method:: MaterialPointSimulator.StrainRateStep(*)

   All step components are interpreted as components of the strain rate tensor.

   The arguments represented by the * are common to all other step methods and are described in :ref:`common_args`.

.. method:: MaterialPointSimulator.StressStep(*)

   All step components are interpreted as components of the stress tensor.

   The arguments represented by the * are common to all other step methods and are described in :ref:`common_args`.

   .. note:: ``kappa`` is set to ``0`` for stress steps

.. method:: MaterialPointSimulator.StressRateStep(*)

   All step components are interpreted as components of the stress rate tensor.

   The arguments represented by the * are common to all other step methods and are described in :ref:`common_args`.

   .. note:: ``kappa`` is set to ``0`` for stress rate steps

.. method:: MaterialPointSimulator.DisplacementStep(*)

   All step components are interpreted as components of the displacement vector, applied only to the "+" faces of a unit cube centered at the coordinate origin.

   The arguments represented by the * are common to all other step methods and are described in :ref:`common_args`.

.. method:: MaterialPointSimulator.DefGradStep(*)

   All step components are interpreted as components of the deformation gradient tensor.

.. method:: MaterialPointSimulator.DataSteps(filename, tc=0, columns=None, descriptors=None, skiprows=0, comments='#', sheet=None, *)

   Generate steps from a data file.

   ``filename`` is the name of a file containing the data.  ``tc`` is the integer index of the column containing time.  ``columns`` are the indices of the columns containing data.  If not given, ``columns`` is taken to be the first six columns of the file, that are not ``tc``.

   ``skiprows`` is the integer number of rows to skip before reading data, ``comments`` is the comment delimiter.  ``sheet`` is the sheet from which to read data, if ``filename`` is an excel file.

   The i\ :sup:`th` ``descriptor`` designates the physical interpretation of the i\ :sup:`th`.  ``descriptors`` must be one of ``'E'`` (strain), ``'D'`` (strain rate), ``'S'`` (stress), ``'R'`` (stress rate), ``'P'`` (electric field), ``'T'`` (temperature).

   The arguments represented by the * are common to all other step methods and are described in :ref:`common_args`.

.. _mixed_step:

.. method:: MaterialPointSimulator.MixedStep(descriptors=None, *)

   All step components are interpreted as components of stress and/or strain.

   The i\ :sup:`th` ``descriptor`` designates the physical interpretation of the i\ :sup:`th`.  ``descriptors`` must be one of ``'E'`` or ``'S'`` with ``'E'`` representing strain and ``'S'`` representing stress.

   The arguments represented by the * are common to all other step methods and are described in :ref:`common_args`.

.. _common_args:

Common Step Arguments
---------------------

The arguments common to all step functions are:

  ``components`` are the components of the tensor defining the step.  Tensor ordering is described in :ref:`intro_conventions`.  For all tensors, the components are assumed to be the "tensor values", as opposed to the "engineering values".  For symmetric tensors, specifying only the three diagonal components implicitly assigns the off-diagonal components a value of zero.  For strain type tensors, if only a single component is given, it is assumed to be a volumetric deformation.  For stress type tensors, if only a single component is given, it is assumed to be a pressure.

  ``scale`` is a multiplier applied to all components.  It can be a float or a numpy ndarray (so that a different scale could be applied to each component separately).

  ``frames`` is the integer number of increments that the step is subdivided in to.

  ``kappa`` the Seth-Hill strain parameter.  See :ref:`strain_tensor` for details.

  ``temperature`` is the temperature.  If not specified, the step is assigned the same temperature as the previous step.

  ``elec_field`` is the electric field vector.  If none is given, it is set to ``(0, 0, 0)``.

  ``num_dumps`` is the integer number of times to write the output database.  If not specified, all step increments are written.

Running the Simulation
======================

.. method:: MaterialPointSimulator.run(termination_time=None)

   Run the simulation

   ``termination_time`` is the termination time.  If not given, the final time from the last step is used.

Extracting Results from the Output Database
===========================================

.. method:: MaterialPointSimulator.get(*variables, model=None, disp=0)

   Get variables from output database.

   ``variables`` is a list of variables to extract.  If ``disp`` is ``1``, the variables are returned, in addition to a header describing the variables.


Viewing Simulation Results
==========================
.. method:: MaterialPointSimulator.view(model=None)

   Display simulation results in visualizer.

Exporting Results to Other Formats
==================================

.. method:: MaterialPointSimulator.dump(variables, model=None, format='ascii', ffmt='%.18f')

   Writes simulation results for the requested ``variables`` to the requested ``format``.

   ``variables`` is a list of variables to extract from the database and are written the ``MaterialPointSimulator.job.ext``, where ``ext`` is a file extension that depends on the requested output ``format``.
