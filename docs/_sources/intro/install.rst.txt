.. _intro_install:

Installing Matmodlab2
#####################

Overview
========

*Matmodlab2*'s code base is largely written in Python and requires no
additional compiling. However, several (optional) linear algebra packages and
material models are written in Fortran and require a separate compile step.

System Requirements
===================

*Matmodlab2* has been built and tested extensively on several versions of Linux
and the Apple Mac OSX operating systems.

Required Software
=================

The basic functionality of *Matmodlab2* requires the following software
installed for your platform:

#) `Python 2.7 or 3.5 <http://www.python.org/>`_

#) `NumPy <http://www.numpy.org/>`_

#) `SciPy <http://www.scipy.org/>`_

*Matmodlab2* has further functionality that can be utilized if the appropriate
packages are installed.

#) `pytest <http://pytest.org/latest>`_ for running tests

#) `pandas <http://pandas.pydata.org>`_ for the ``DataFrame`` object.

The required software may be obtained in several ways, though most development
has been made using the `Anaconda <http://www.continuum.io>`_ Python distribution.

.. _installation:

Installation
============

.. note::

   Ensure that all *Matmodlab2* prerequisites are installed and working properly before proceeding.

Obtain the source code from `github <https://github.com/matmodlab/matmodlab2>`_
and add the path to the ``matmodlab2`` directory to your ``PYTHONPATH`` environment
variable::

    $ git clone https://www.github.com/matmodlab/matmodlab2
    $ export MATMODLAB2_DIR=`pwd`/matmodlab
    $ export PYTHONPATH=${PYTHONPATH}:${MATMODLAB2_DIR}

Build (Optional)
----------------

A library of Fortran linear algebra routines can also be built. Navigate to
``matmodlab2/matmodlab2/ext_helpers`` and execute the ``build_ext.py`` script::

    $ cd ${MATMODLAB2_DIR}/matmodlab2/ext_helpers
    $ ./build_ext.py matfuncs

Testing the Installation
========================

Testing requires that the `pytest <http://pytest.org/latest>`_ module be installed.  Tests are run by executing::

  $ cd ${MATMODLAB2_DIR}/tests
  $ py.test .

Troubleshooting
===============

If you run in to problems, open an issue at
`<https://github.com/matmodlab/matmodlab2>`_.
