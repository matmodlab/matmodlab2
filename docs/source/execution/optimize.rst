.. _optimize:

Optimizer
#########

.. topic:: See Also

   * :ref:`mps`

Overview
========

Optimize specified parameters against user specified objective function. Ideal for finding optimal model parameters. A optimizer instance is created through the ``Optimizer`` constructor.

The Optimizer Constructor
=========================

.. class:: Optimzer(job, func, xinit, method=SIMPLEX, verbosity=1, d=None, maxiter=MAXITER, tolerance=TOL, descriptors=None, funcargs=[], Ns=10)

   Create a Optimzer object and set up the simulation.

   The *job* string is the simulation ID.  The Permutator creates a job.eval/ directory in the simulation directory. The ith individual job is then run in job.eval/eval_i/.

   The Optimzer writes relevant simulation information to job.eval/job.xml.  The Matmodlab.Visualizer can read the job.xml file and display the permutated job.

   *func* is a function that evaluates a Matmodlab simulation.  It is called as *func(x, xnames, d, job, *funcargs)*, where *x* are the current values of the permutated variables, *xnames* are their names, *d* is the simulation directory of the current job, *job* is the job ID, and *funcargs* are additional arguments to be sent to *func*.

   *xinit* is a list of initial values of the simulation parameters to be optimized.  Each member of the list must be a OptimizeVariable instance.

   The following arguments are optional

   *method* is the method for determining how to combine parameter values. One of ``SIMPLEX``, ``POWELL``, ``COBYLA``, or ``BRUTE``.

   *maxiter* is the integer maximum number of iterations and *tolerance* is the tolerance.  The default tolerance is 1e-8.

   *descriptors* is a list of descriptors for the values returned from *func*.

   *d* is the parent directory to run jobs.  If the directory does not exist, it will be created.  If the directory exists and *bu* is *False*, the directory will be first erased and then re-created.  If the directory exists but *bu* is *True*, the directory is archived.

   *Ns* is the number of evaluations per dimension for brute force optimization.

Running the Optimizer
=====================

.. method:: Optimizer.run()

   Run the simulation

OptimizeVariable Factory Method
===============================

.. function:: OptimizeVariable(name, initial_value, bounds=None)

   Create a OptimizeVariable object

   *name* is the name of variable and *initial_value* is its initial value.  *bounds* are the bounds on the variable given as (lower_bound, upper_bound).  Bounds are only applicable if the optimizer method is ``COBYLA``.

Example
=======

The following input demonstrates how to optimize the ``K`` and ``G``
parameters and can be found in ``matmodlab/examples/optimize.py``.  The objective function calls ``calculate_bounded_area`` to find the area between the calculated stress strain curve and the experimental.

The Example Script
------------------

.. code:: python

  import os
  import numpy as np

  from matmodlab import *
  import matmodlab.utils.fileio as ufio
  import matmodlab.utils.numerix.nonmonotonic as unnm

  filename = os.path.join(get_my_directory(), "optimize.xls")
  strain_exp, stress_exp = zip(*ufio.loadfile(filename, sheet="MML", disp=0,
                                              columns=["STRAIN_XX", "STRESS_XX"]))

  def func(x=[], xnames=[], evald="", job="", *args):
      mps = MaterialPointSimulator(job)

      xp = dict(zip(xnames, x))
      NU = 0.32  # poisson's ratio for aluminum
      parameters = {"K": xp["E"]/3.0/(1.0-2.0*NU), "G": xp["E"]/2.0/(1.0+NU),
                    "Y0": xp["Y0"], "H": xp["H"], "BETA": 0.0}
      mps.Material("vonmises", parameters)

      # create steps from data. note, len(columns) below is < len(descriptors).
      # The missing columns are filled with zeros -> giving uniaxial stress in
      # this case. Declaring the steps this way does require loading the excel
      # file anew for each run
      mps.DataSteps(filename, steps=30, sheet='MML',
                    columns=('STRAIN_XX',), descriptors='ESS')

      mps.run()
      if not mps.ran:
          return 1.0e9

      strain_sim, stress_sim = zip(*mps.get("STRAIN_XX", "STRESS_XX"))
      error = unnm.calculate_bounded_area(strain_exp, stress_exp,
                                        strain_sim, stress_sim)
      return error

  E = OptimizeVariable("E",  2.0e6, bounds=(1.0e5, 1.0e7))
  Y0= OptimizeVariable("Y0", 0.3e5, bounds=(1.0e4, 1.0e6))
  H = OptimizeVariable("H",  1.0e6, bounds=(1.0e4, 1.0e7))

  optimizer = Optimizer("optimize", func, [E, Y0, H], method=POWELL,
                        maxiter=200, tolerance=1.e-3)
  optimizer.run()

How Does the Script Work?
-------------------------

This section describes each part of the example script

.. code:: python

  from matmodlab import *
  import matmodlab.utils.fileio as ufio
  import matmodlab.utils.numerix.nonmonotonic as unnm

  filename = os.path.join(get_my_directory(), "optimize.xls")
  strain_exp, stress_exp = zip(*ufio.loadfile(filename, sheet="MML", disp=0,
                                              columns=["STRAIN_XX", "STRESS_XX"]))

This statement makes the Matmodlab objects accessible to the script and import other functions for reading Excel data files and comparing two curves.  The experimental data is read and stored.

.. code:: python

  E = OptimizeVariable("E",  2.0e6, bounds=(1.0e5, 1.0e7))
  Y0= OptimizeVariable("Y0", 0.3e5, bounds=(1.0e4, 1.0e6))
  H = OptimizeVariable("H",  1.0e6, bounds=(1.0e4, 1.0e7))

These statements define parameters ``E``, ``Y0``, and ``H`` to be the variable to be optimized.

.. code:: python

  optimizer = Optimizer("optimize", func, [E, Y0, H], method=POWELL,
                        maxiter=200, tolerance=1.e-3)

This statement instantiates the ``Optimzer`` object, using the ``POWELL`` method.

.. code:: python

  optimizer.run()

This statement runs the job.

.. code:: python

  def func(x, xnames, d, job, *args):

      mps = MaterialPointSimulator(job)
      xp = dict(zip(xnames, x))
      NU = 0.32  # poisson's ratio for aluminum
      parameters = {"K": xp["E"]/3.0/(1.0-2.0*NU), "G": xp["E"]/2.0/(1.0+NU),
                    "Y0": xp["Y0"], "H": xp["H"], "BETA": 0.0}
      mps.Material("vonmises", parameters)

These statements define the function exercised by the Optimzer.  The first lines are the instantiation of the MaterialPointSimulator, and its material.  The current parameters are passed in from Matmodlab.

.. code:: python

      mps.DataSteps(filename, steps=30, sheet='MML',
                    columns=('STRAIN_XX',), descriptors='ESS')

      mps.run()

These statements create the analysis steps from the experimental data file and run the simulation

.. code:: python

      strain_sim, stress_sim = zip(*mps.get("STRAIN_XX", "STRESS_XX"))
      error = unnm.calculate_bounded_area(strain_exp, stress_exp,
                                        strain_sim, stress_sim)
      return error

These statements read in the analysis results and compute the error between them and the experimental data.
