.. _intro_install:

Installing Matmodlab
####################

Overview
========

Matmodlab's code base is largely written in Python and requires no
additional compiling. However, several (optional) linear algebra packages and
material models are written in Fortran and require a seperate compile step.

System Requirements
===================

Matmodlab has been built and tested extensively on several versions of Linux
and the Apple Mac OS X 10.9 operating systems. It is unknown whether or not
Matmodlab will run on Windows.


Required Software
=================

The basic functionality of Matmodlab requires the following software installed
for your platform:

#) `Python 2.7 <http://www.python.org/>`_ (A, E)

#) `NumPy <http://www.numpy.org/>`_ (A, E)

#) `SciPy <http://www.scipy.org/>`_ (A, E)

Matmodlab has further functionality that can be utilized if the appropriate
packages are installed.

#) `traits <http://pypi.python.org/pypi/traits>`_, `traitsui <http://pypi.python.org/pypi/traitsui>`_, and `chaco <http://pypi.python.org/pypi/chaco>`_ for data visualization (E)

#) `pytest <http://pytest.org/latest>`_ for running tests (A, E)

#) `openpyxl <http://pypi.python.org/pypi/openpyxl>`_ for simulation output and visualization of .xlsx data (A, E)

#) `xlwt <http://pypi.python.org/pypi/xlwt>`_ for simulation output as .xls (A)

#) `xlrd <http://pypi.python.org/pypi/xlrd>`_ for visualization of .xls data (A)

#) `matplotlib <http://matplotlib.org>`_ for data visualization in :ref:`Matmodlab.Notebook <notebook>`  (A, E)

#) `bokeh <http://bokeh.pydata.org/en/latest>`_ for interactive data visualization in :ref:`Matmodlab.Notebook <notebook>`  (A)

The required software may be obtained in several ways, though most development
has been made using the `Anaconda <http://www.continuum.io>`_ and `Enthought Canopy <http://www.enthought.com>`_ Python Distributions (E=available in
base Enthought Canopy distribution, A=available in base Anaconda distribution).
It is also possible to get all of the required packages through a linux
distribution's package manager or, for all installations of python, by running

::

  easy_install xlrd

or

::

  pip install chaco

or, for Anaconda,

::

  conda install traitsui

and so on for each required package.

If using a linux distribution's version of python, you may need to install the
python development headers in order to build fortran models or use the faster
linear algebra package.

.. _installation:

Installation
============

.. note::

   Ensure that all Matmodlab prerequisites are installed and working properly before proceeding.

There are several options for installing Matmodlab, the two recommonded are

* Installing the :ref:`stable_v` from `PyPI <https://pypi.python.org/pypi>`_.
* Installing the :ref:`dev_v` from source.

Whichever method is chosen, it is recommended to install Matmodlab in a :ref:`venv`.

.. _stable_v:

Latest Stable Version
---------------------

The latest stable version of Matmodlab can be installed via pip::

  pip install matmodlab

or ``easy_install``::

  easy_install matmodlab

.. note::

   You may have to have administrative privileges to install Matmodlab through a package manager.

If you are unsure of the the difference between ``pip`` and ``easy_install``, try to
use ``pip`` first.

.. _dev_v:

Latest Development Branch
-------------------------

Installing the latest development version consists of retrieving the source code from the online code repository and installing it. The source code can be obtained from `<https://github.com/tjfulle/matmodlab>`_::

  git clone https://github.com/tjfulle/matmodlab.git

Change directories to ``matmodlab`` and run

::

  python setup.py install

or

::

  python setup.py develop

Both commands make Matmodlab visible to the Python interpreter,
in the same way as using ``pip`` or ``easy_install``. However, when you setup
using the ``develop`` argument, source files files are linked to the Python
interpreter's site-packages, rather than copied. This way, changes made to
source files are applied immediately and do not require you to re-install
Matmodlab.

Build (Optional)
----------------

Fortran models are built as-needed when Matmodlab attempts to run a
simulation and it cannot find the compiled model. However, if you want
to build a model without running a simulation or if you want to build an
extension pack you will need to use the ``mml build`` command.  See :ref:`cli_build` for details on the build command.

Example
.......

::

  $ mml build

This will build the Matmodlab Fortran utilities and material libraries. The
resultant shared object libraries are copied to ``matmodlab/lib``.

.. _venv:

Python Virtual Environment
--------------------------

It is recommended that you install Matmodlab in a `Virtual Environment <http://docs.python-guide.org/en/latest/dev/virtualenvs>`_.  As an example, consider installing Matmodlab using Anaconda::

  $ conda create -n matmodlab numpy scipy traitsui chaco xlrd pytest matplotlib
  $ source activate matmodlab
  $ cd ~/Developer/matmodlab
  $ python setup.py develop
  $ mml build

In the preceding commands, a virtual environment named *matmodlab* was created with the required packages and then activated.  We then navigated in to the ``matmodlab`` directory (in ``~/Developer/matmodlab`` in this example), executed the setup script and built the optional libraries.

Consult your Python distribution's documentations for instructions to create and use virtual environments.

Testing the Installation
========================

Testing requires that the `pytest <http://pytest.org/latest>`_ module be installed.  Tests are run by executing::

  $ mml test

See :ref:`cli_test` for details on the test command.

Troubleshooting
===============

If you experience problems when building/installing/testing Matmodlab, you can
ask help from `Tim Fuller <timothy.fuller@utah.edu>`_ or
`Scot Swan <scot.swan@gmail.com>`_.
