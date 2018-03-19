The Material Model Laboratory
#############################

The Material Model Laboratory (*Matmodlab2*) is a single element material model
driver aimed at developers of constitutive routines targeted for deployment in
finite element codes.

This guide is separated into four main parts:

* Part 1: :ref:`intro_and_overview`
* Part 2: :ref:`model_create_and_execute`
* Part 3: :ref:`mat_index`
* Part 4: :ref:`test_index`

Additionally, a growing collection of `Jupyter Notebooks <https://github.com/matmodlab/matmodlab2/blob/master/notebooks/GettingStarted.ipynb>`_ demonstrate *Matmodlab2* usage.

About This Guide
================

This guide is both a User's Guide and Application Programming Interface (API)
guide to Matmodlab. The guide assumes a working knowledge of the computing
languages Matmodlab is developed in, namely `Python <https://www.python.org>`_
and `Fortran
<http://www.fortran.com/the-fortran-company-homepage/fortran-tutorials>`_. No
attempt to describe either is made. Online tutorials for each language are
readily available. Likewise, the target audience is assumed to have a basic
knowledge of continuum mechanics and familiarity with other finite element
codes. These concepts are also not described in detail.

Conventions Used in the Guide
-----------------------------

* Python objects are typeset in ``fixed point font``.
* ``$`` is used to denote a terminal prompt, e.g. ``$ cd`` is interpreted as executing the ``cd`` (change directory) command at a command prompt.

License
=======

*Matmodlab2* is an open source project licensed under the `MIT <http://opensource.org/licenses/MIT>`_ license.

Obtaining and Installing Matmodlab2
===================================

*Matmodlab2* is maintained with git and can be obtained from `<https://github.com/matmodlab/matmodlab2>`_.  To install *Matmodlab2*, obtain a copy (by cloning or downloading) of the source code and add the path to the ``matmodlab2`` root directory to your ``PYTHONPATH``.

Acknowledgments
===============

The inspiration for *Matmodlab2* came from Dr. `Rebecca Brannon's <http://www.mech.utah.edu/~brannon/>`_ *MED* material point driver and Tom Pucicks's *MMD* driver, developed at Sandia National Labs.

The syntax and documentation have been greatly influenced by the authors' exposure and use of research codes at Sandia National Labs and other commercial finite element codes, notably `Abaqus <http://www.3ds.com/products-services/simulia/products/abaqus/latest-release>`_.

Obtaining Additional Help
=========================

In addition to this guide, many examples can be found in
``matmodlab2/notebooks`` and ``matmodlab2/tests``

Indices and tables
==================

* :ref:`genindex`

* :ref:`search`

.. toctree::
   :maxdepth: 4
   :hidden:
   :numbered: 2

   intro/index
   execution/index
   material/index
   examples/index
   test/index
