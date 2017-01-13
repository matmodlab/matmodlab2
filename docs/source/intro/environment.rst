
.. _environment:

Environment Settings
####################

.. topic:: See Also

   * :ref:`intro_conventions`
   * :ref:`basic_cli`
   * :ref:`python_models`
   * :ref:`fortran_models`

Overview
========

Matmodlab sets up and performs execution of input scripts in a customized
environment (not to be confused with Python virtural environments). The
environment can be modifed by user environment files that are read at the
beginning of each job.

Environment File Locations
==========================

Matmodlab searches for the optional user environment file, ``mml_userenv.py``, in three locations, in the following order:

1) Your home directory.
2) The location specified by the environment variable ``MML_USERENV``
3) The current working directory.

The value of a parameter is the the last definition encountered, meaning that the order of precedence for user settings is the current working directory, ``MML_USERENV``, and the home directory.

Environment files use Python syntax, meaning that entries will have the following form::

  parameter = value

All usual Python conventions apply.

Recognized Environment Settings and Defaults
============================================

Below are the recognized environment settings and their defaults.  Any of these settings can be changed by specifying a different value in a user environment file.

.. note::

   When specifying environment settings in a user environment file, the
   setting must have the same type as the default. If the default is a list,
   the user setting is inserted in to the default list. If the default is a
   dictionary, it is updated with the user setting.

IO Settings
-----------

verbosity

  Matmodlab will print more, or less, information during job execution. Possible values are ``0``, ``1``, and ``2``. Set the value to ``0`` to suppress printing of information. Set the value to ``2``, or higher, to print increase the amount of information printed. The default value is ``1``.

warn

  Define how Matmodlab is to interpret warnings.  Possible values are ``WARN``, ``IGNORE``, ``ERROR``.  Set the value to ``IGNORE`` to suppress warnings and to ``ERROR`` to treat warnings as errors.  The default ``WARN`` prints warning messages.

Wall:

  Print *all* warning messages, including internal warnings.  Possible values are ``True`` and ``False``.  The default ``False`` does not print some internal warnings.

Debugging and SQA
-----------------

raise_e

  Define how Matmodlab treats errors that are raised during job execution.
  Possible values are ``True`` and ``False``. If ``False``, Matmodlab prints
  errors to the console and log file and exits. If set to ``True``, errors are
  raised instead. The default is ``False``, except for the Matmodlab.Notebook
  environment where it is ``True``.

sqa

  Run additional SQA checks.  The default is ``False``.

debug

  Run additional debug code.  The default is ``False``.

sqa_stiff

  Perform checks of stiffness sent back from constitutive models.  The default is ``False``.

Performance
-----------

nprocs

  The number of simultaneous jobs to run.  The option is used only by the Matmodlab.Permutator.  The default is ``1``.


Material Switching
------------------

switch

  A list of (old, new) tuples specifying the model switching behavior.  The default is ``[]``.


User Material Models
--------------------

materials

  A dictionary describing user material models.  The dictionary consists of ``model:information`` key:value pairs where ``information`` is a dictionary containing material model meta data needed by Matmodlab.  The default is ``{}``.  User values update and do not overwrite this variable.

std_materials

  A list containing directories and files to search for standard material models.  The default is ``[MAT_D]``.  User values are appended to and do not overwrite this variable.

Simulation Directory
--------------------

simulation_dir

  The directory to run the simulation.  The default is the current working directory.

A Note on Defining User Material Models
---------------------------------------

``std_materials`` and ``materials``  user settings are used to inform Matmodlab concerning user defined materials.  ``std_materials`` is a list of python interface files for standard models.  The ``materials`` dictionary is a dictionary of ``model_name: attribute_dict`` key:value pairs with the dictionary of model attributes containing the following information:

* *source_files*: [list, required] A list of model source files
* *model*: [symbolic constant, optional] the model type.  One of ``USER``, ``UMAT``, ``UHYPER``, ``UANISOHYPER``.  The default is ``USER``.
* *response*: [symbolic constant, optional] The model response, one of ``MECHANICAL``, ``HPERELASTIC``, ``ANISOHYPER``.  The default is ``MECHANICAL``.
* *ordering*: [list of symbolic constants, optional] Symmetric tensor ordering.  The default is ``[XX, YY, ZZ, XY, YZ, XZ]``.
* *user_ics*: [boolean, optional] If ``True``, the user model provides a SDVINI subroutine.
* *param_names*: [list of string, optional] A list of model parameter names.  If given, parameters should be sent to the ``MaterialPointSimulator.Material`` constructor as a dictionary of ``key:value`` pairs, where ``key`` is the parameter name.  Otherwise, parameters should be sent to the constructor as an array in the order expected by the model.

Example
.......

The following is a portion of the user environment file found in ``matmodlab/examples`` and is used by ``matmodlab/examples/users.py`` to define the material model's attributes::

  materials = {'neohooke_u': {'model': USER, 'response': HYPERELASTIC,
                              'libname': 'neohooke_u',
                              'source_files': [join(MAT_D, 'src/uhyper_neohooke.f90')],
                              'ordering': [XX, YY, ZZ, XY, XZ, YZ]}}
