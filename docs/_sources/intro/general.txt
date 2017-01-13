.. _intro_general:

Introduction: General
#####################

Overview of the Material Model Laboratory
=========================================

The Material Model Laboratory (Matmodlab) is a suite of tools designed for the development of material models intended for deployment in finite element codes.
Matmodlab includes:

* Matmodlab.Simulator, a specialized material model driver;
* Matmodlab.Permutator, a toolkit for determining model sensitivities to
  parameters;
* Matmodlab.Optimizer, an optimization toolkit for optimizing material model
  parameters;
* Matmodlab.Viewer, an interactive visualization tool;
* Matmodlab.Notebook, a customized `IPython Notebook <http://ipython.org/notebook.html>`_ environment.

The Matmodlab.Simulator can be thought to drive a single material point of a
finite element simulation through specific user defined paths. This permits
exercising material models in ways not possible in finite element
calculations, desgining verification and validation tests of the material
response, among others. Matmodlab is a small suite of tools at the developers
disposal to aid in the design and implementation of material models in larger
finite element host codes. It is also a useful tool for analysts to understand
and parameterize a material's response to deformation.

About Matmodlab
===============

Matmodlab is implemented in Python and leverages Python's object oriented
programming (OOP) design. OOP techniques are used throughout Matmodlab to
setup and manage simulation data. Computationally heavy portions of the code,
and many material models themselves, are written in Fortran for its speed and
ubiquity in scientific computing. Calling Fortran procedures from Python is
made possible by the `f2py
<http://docs.scipy.org/doc/numpy-dev/f2py/usage.html>`_ module, standard in
Numpy, that compiles and creates Python shared object libraries from Fortran
sources.

Using Matmodlab
===============

The Matmodlab.Simulator is run in the Matmodlab computational environment. The
environment is invoked by executing input scripts from the command line (see
:ref:`model_create_and_execute`, for details) or from the
:ref:`Matmodlab.Notebook <notebook>` environment. The main input the
Matmodlab.Simulator is a python script containing directives and data required
for the simulation. Directions on generating input scripts are provided in
this guide.

As you begin to use Matmodlab, it is recommended that you read and modify the many examples in the ``matmodlab/examples`` directory in the root directory of Matmodlab.

Viewing the Results of a Matmodlab Simulation
=============================================

Simulation data is written to an output database. Output database files can be
visualized in :ref:`Matmodlab.Viewer <viewer>` or interactively in the
:ref:`Matmodlab.Notebook <notebook>`.
