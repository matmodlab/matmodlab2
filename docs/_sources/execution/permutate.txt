.. _inpperm:

Permutator
##########

.. topic:: See Also

   * :ref:`mps`

Overview
========

It is useful to investiage sensitivities of models to model inputs. The ``Permutator`` runs jobs with different realization of parameters.  The ``Permutator`` is designed as a tool for discovering and investigating model sensitivities.

The Permutator Constructor
==========================

.. class:: Permutator(job, func, xinit, method=ZIP, correlations=False, verbosity=None, descriptors=None, nprocs=1, funcargs=[], d=None, shotgun=False, bu=False)

   Create a Permutator object and set up the simulation.

   The *job* string is the simulation ID.  The Permutator creates a job.eval/ directory in the simulation directory. The ith individual job is then run in job.eval/eval_i/.

   The Permutator writes relevant simulation information to job.eval/job.xml.  The Matmodlab.Visualizer can read the job.xml file and display the permutated job.

   *func* is a function that evaluates a Matmodlab simulation.  It is called as *func(x, xnames, d, job, *funcargs)*, where *x* are the current values of the permutated variables, *xnames* are their names, *d* is the simulation directory of the current job, *job* is the job ID, and *funcargs* are additional arguments to be sent to *func*.

   *xinit* is a list of initial values of the simulation parameters to be permutated.  Each member of the list must be a PermutateVariable instance.

   The following arguments are optional

   *method* is the method for determining how to combine parameter values. One of ``ZIP`` or ``COMBINATION``. The ``ZIP`` method runs one job for each set of parameters (and, thus, the number of realizations for each parameter must be identical), the ``COMBINATION`` method runs every combination of parameters.

   *correlations* is a boolean that, if True, instructs the Permutator to create a correlation table and plots relating permutated parameters and return value of *func*.

   *descriptors* is a list of descriptors for the values returned from *func*.

   *nprocs* is the integer number of simultaneous jobs to run.

   *d* is the parent directory to run jobs.  If the directory does not exist, it will be created.  If the directory exists and *bu* is *False*, the directory will be first erased and then re-created.  If the directory exists but *bu* is *True*, the directory is archived.

   If *shotgun* is *True*, input parameters are randomized.

Running the Permutator
======================

.. method:: Permutator.run()

   Run the simulation

PermutateVariable Factory Method
================================

.. function:: PermutateVariable(name, init, method=LIST, b=None, N=10)

   Create a PermutateVariable object

   *name* is the name of variable and *init* is its initial value (or values).  *method* is the method use to generate all values. If *method* is ``LIST``, then all values shall be given in *init*.  Otherwise, values will be generated. Valid methods are ``LIST, WEIBULL, UNIFORM, NORMAL, PERCENTAGE``.

   The interpretation of the following arguments depends on *method*

   For methods other than ``LIST``, values are generated from a function called as *fun(init, b, N)*.  The interpretation of *b* is dependent on which method fun represents.  *N* is the number of values to generate.

Example
=======

The following input demonstrates how to permutate the ``K`` and ``G``
parameters to the ``elastic`` model.  The input can be found in ``matmodlab/examples/permutate.py``.

The Example Script
------------------

.. code:: python

  from matmodlab import *

  def func(x, xnames, d, job, *args):

      mps = MaterialPointSimulator(job)
      mps.StrainStep(components=(1, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(2, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(1, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(0, 0, 0), increment=1., scale=-.5, frames=10)

      # set up the material
      parameters = dict(zip(xnames, x))
      mps.Material('elastic', parameters)

      # set up and run the model
      mps.run()

      s = mps.get('STRESS_XX')
      return np.amax(s)

  K = PermutateVariable('K', 125e9, method=WEIBULL, b=14, N=15)
  G = PermutateVariable('G', 45e9, method=PERCENTAGE, b=10, N=15)
  permutator = Permutator('permutation', func, [K, G], method=ZIP,
                          descriptors=['MAX_PRES'], correlations=True)
  permutator.run()


How Does the Script Work?
-------------------------

This section describes each part of the example script

.. code:: python

  from matmodlab import *

This statement makes the Matmodlab objects accessible to the script.

.. code:: python

  K = PermutateVariable('K', 125e9, method=WEIBULL, b=14, N=15)
  G = PermutateVariable('G', 45e9, method=PERCENTAGE, b=10, N=15)

These statements define parameters ``K`` and ``G`` to be permutated by the ``WEIBULL`` method with a Weibull modulus of 14 (``b``) and ``PERCENTAGE`` method with parameters chosen between +/- 10% of the initial (``b``), respectively.

.. code:: python

  permutator = Permutator('permutation', func, [K, G], method=ZIP,
                          descriptors=['MAX_PRES'], correlations=True)

This statement instantiates the ``Permutator`` object, using the ``ZIP`` method.  Correlations between ``K``, ``G`` and the output variable ``MAX_PRES`` are requested.  Note that ``MAX_PRES`` is returned by ``func`` and not Matmodlab.

.. code:: python

  permutator.run()

This statement runs the job.

.. code:: python

  def func(x, xnames, d, job, *args):

      mps = MaterialPointSimulator(job)
      mps.StrainStep(components=(1, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(2, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(1, 0, 0), increment=1., scale=-.5, frames=10)
      mps.StrainStep(components=(0, 0, 0), increment=1., scale=-.5, frames=10)

      # set up the material
      parameters = dict(zip(xnames, x))

These statements define the function exercised by the Permutator.  The first lines are the instantiation of the MaterialPointSimulator, and specification of the analysis steps.  In the last line, the *parameters* dictionary is assembled with the current values of the permutated variables in *x* and their names *xnames*.

.. code:: python

      mps.Material('elastic', parameters)

      # set up and run the model
      mps.run()

These statements set up the simulators material and run the job.

.. code:: python

      s = mps.get('STRESS_XX')
      return np.amax(s)

These statements get the maximum axial stress from the simulation and return it.
