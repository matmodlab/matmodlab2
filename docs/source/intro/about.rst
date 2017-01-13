
Introduction: Background
########################

Why a Single Element Driver?
============================

Single element drivers allow the constituive model developer to concentrate on
model development and not the finite element response. Advantages of
Matmodlab (or, more generally, of any stand-alone constitutive model driver)
are

  * Matmodlab is a very small, special purpose, code. Thus, maintaining
    and adding new features to Matmodlab is very easy;

  * simulations are not affected by irrelevant artifacts such as artificial
    viscosity or uncertainty in the handling of boundary conditions;

  * it is straightforward to produce supplemental output for deep analysis of
    the results that would otherwise constitute an unnecessary overhead in a
    finite element code;

  * specific material benchmarks may be developed and automatically run
    quickly any time the model is changed; and

  * specific features of a material model may be exercised easily by the model
    developer by prescribing strains, strain rates, stresses, stress rates, and
    deformation gradients as functions of time.

Why Python?
===========

Python is an interpreted, high level object oriented language. Programs can be
written rapidly and, because it is an interpreted language, do not require a
compiling step. While this might make programs written in python slower than
those written in a compiled language, modern packages and computers make the
speed up difference between python and a compiled language for single element
problems almost insignificant.

For numeric computations, the `NumPy <http://www.numpy.org>`_ and `SciPy
<http://www.scipy.org>`_ modules allow programs written in Python to leverage
a large set of numerical routines provided by lapack, blas,
eigpack, etc. Python's APIs also allow for calling subroutines written in
C or Fortran (in addition to a number of other languages), a prerequisite for
model development as most legacy material models are written in Fortran. In
fact, most modern material models are still written in Fortran to this day.

Python's object oriented nature allows for rapid installation of new material
models.
