.. _examples:

Annotated Examples
##################

.. topic:: See Also

   * :ref:`intro_conventions`
   * :ref:`mps`

Overview
========

In this section, several example input scripts are described. A Matmodlab
input script consists of defining an instance of the
``MaterialPointSimulator`` class and defininig for it a ``material`` and
``steps``. The following examples provide illustration.

Job Execution
=============

Simulations are run by processing Matmodlab input scripts with the ``mml``
command line utility::

  $ mml run filename.py

where ``filename.py`` is the name of the input script.

Examples
========

.. toctree::
   :maxdepth: 1

   ../examples/ex1
   ../examples/ex2
   ../examples/ex3
