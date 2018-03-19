Input Syntax Guidelines
#######################

Overview
========

This section describes some conventions used in the *Matmodlab2* API and adopted
in user input scripts.

Matmodlab2 Namespace
====================

Input scripts to *Matmodlab2* should include::

   from matmodlab2 import *

to populate the script's namespace with *Matmodlab2* specific functions and
classes.

Primary Classes
---------------

The primary classes exposed by importing ``matmodlab2`` are

* ``MaterialPointSimulator``, the *Matmodlab2* material point simulator
* ``Optimizer``, the *Matmodlab2* optimizer

Each of these classes is described in more detail in the following sections.


Naming Conventions
==================

Throughout *Mamodlab2*, the following naming conventions are adopted (see the
`PEP8 <https://www.python.org/dev/peps/pep-0008>`_ guidelines for more
guidance):

* Class names use the ``CapWords`` convention.
* Method names use ``lowercase`` with words separated by underscores as
  necessary to improve readability.
* Variable names adopt the same rule as method names.
