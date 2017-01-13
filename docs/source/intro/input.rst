Input Syntax Guidelines
#######################

.. topic:: See Also

   * :ref:`model_create_and_execute`
   * :ref:`mml_out_dbs`

Overview
========

This section describes some conventions used in the Matmodlab API and adopted in user input scripts.

Matmodlab Namespace
===================

Input scripts to Matmodlab should include::

   from matmodlab import *

to populate the script's namespace with Matmodlab specific parameters,
classes, and symbolic constants.

Parameters
----------

The parameters exposed by importing ``matmodlab`` are

* ``ROOT_D``, the root ``matmodlab`` directory
* ``PKG_D``, the ``matmodlab/lib`` directory, the location shared objects are copied
* ``MAT_D``, the directory where builtin materials are contained

Classes
-------

The classes exposed by importing ``matmodlab`` are

* ``MaterialPointSimulator``, the Matmodlab material point simulator
* ``Permutator``, the Matmodlab permutator
* ``Optimizer``, the Matmodlab optimizer

Each of these classes is described in more detail in the following sections.

Symbolic Constants
------------------

The following symbolic constants are exposed by importing ``matmodlab``.

Symbolic Constants Relating to Matmodlab.Simulator
..................................................

* ``DBX, EXO, TXT, PKL, XLS, XLSX``, constants representing the output file formats.  See :ref:`mml_out_dbs`.

Symbolic Constants Relating to User Defined Materials
.....................................................

* ``XX, YY, ZZ, XY, YZ, XZ``, constants representing the *xx*, *yy*, *zz*, *xy*, *yz*, and *xz* components of second-order symmetric tensors.
* ``MECHANICAL``, ``HYPERELASTIC``, ``ANISOHYPER`` are constants representing user defined materials for mechanical, hyperelastic, and anisotropic-hyperelastic material responses, respectively (see :ref:`user_mats`).
* ``WLF`` specifies a WLF time-temperature shift (see :ref:`trs`)
* ``PRONY`` specifies a prony series input to the viscoelastic model (see :ref:`viscoelastic`)
* ``ISOTROPIC`` specifies isotropic thermal expansion (see :ref:`expansion`)
* ``USER`` specifies that a user developed model is of type "user"
* ``UMAT`` specifies that a user developed model is of type "umat"
* ``UHYPER`` specifies that a user developed model is of type "uhyper"
* ``UANISOHYPER_INV`` specifies that a user developed model is of type "uanisohyper_inv"

Symbolic Constants Relating to Matmodlab.Permutator
...................................................

* ``ZIP``, apply the zip method to a permutation job.
* ``COMBINATION``, apply the combination method to a permutation job.

* ``RANGE``, permutated variables defined as a range
* ``LIST``, permutated variables defined as a list
* ``WEIBULL``, permutated variables permutated using a Weibull distribution
* ``UNIFORM``, permutated variables permutated using a Uniform distribution
* ``NORMAL``, permutated variables permutated using a Normal distribution
* ``PERCENTAGE``, permutated variables permutated +/- b% from the nominal value
* ``UPERCENTAGE``, permutated variables permutated using a uniform distribution with bounds +/- b% from the nominal value
* ``NPERCENTAGE``, permutated variables permutate using a normal distribution with bounds +/- b% from the nominal value

Symbolic Constants Relating to Matmodlab.Optimizer
..................................................

* ``SIMPLEX``, Simplex optimization method
* ``POWELL``, Powell optimization method
* ``COBYLA``, COBYLA optimization method
* ``BRUTE``, brute force optimization method

Naming Conventions
==================

Throughout Matmodlab, the following naming conventions are adopted (see the `PEP8 <https://www.python.org/dev/peps/pep-0008>`_ guidelines for more guidance):

* Class names use the CapWords convention.
* Method names use lowercase with words separated by underscores as necessary to improve readability.
* Variable names adopt the same rule as method names.
* Symbolic constants are written in all capital letters with underscores separating words.
* Factory methods adopt the same rule as class names.
