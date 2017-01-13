.. _python_models:

Python User Material Interface
##############################

.. topic:: See Also

   * :ref:`Role of Material Model`
   * :ref:`defining_a_material`
   * :ref:`intro_conventions`
   * :ref:`comm_w_matmodlab`

Overview
========

Material models written in Python are implemented as subclasses
``matmodlab.mmd.material.MaterialModel`` and are treated as builtin materials.

Invoking User Materials
=======================

User materials that subclass ``MaterialModel`` are invoked by assigning the *model* argument of the ``MaterialPointSimulator.Material`` factory method to the name of the material model.

Required Attributes
===================

Material models that subclass ``MaterialModel`` must provide the following class attributes:

* *name*, as string defining the material's name.  Must be unique.

Required Methods
================

.. classmethod:: MaterialModel.param_names(n)

   Class method that returns a list of parameter names. *n* is the number of parameters and is used to set the names of parameters for user defined materials at run time.

.. method:: MaterialModel.setup(**kwargs)

   Sets up the material model and return a list of state dependent variable
   names and initial values. By the time that *setup* is called, the model
   parameters have been

   *kwargs* are optional keywords sent in to the model.

   *setup* must return *sdv_keys*, *sdv_vals*, *sdv_keys* being the list of
    state dependent variable names and *sdv_vals* being their initial values.
    Both should be consistent with the ordering expected by the material
    model.

.. method:: MaterialModel.update_state(time, dtime, temp, dtemp, energy, density, F0, F1, strain, dstrain, elec_field, stress, statev, **kwargs)

   Update the the material state

   The following parameters are sent in for information and should not be
   updated:

   * *time*, the time at the beginning of the time step
   * *dtime*, Step time step size
   * *temp*, the temperature at the beginning of the time step
   * *dtemp*, step temperature increment
   * *energy*, the energy at the beginning of the time step
   * *density*, the material density
   * *F0*, the deformation gradient at the beginning of the time step
   * *F1*, the deformation gradient at the beginning of the time step
   * *strain*, the strain at the beginning of the time step
   * *dstrain*, the strain increment over the step
   * *elec_field*, the electric field at the end of the step

   The following parameter are sent in for information and should be
   updated to the end of the step:

   * *stress*, the stress at the beginning of the step
   * *statev*, the state dependent variables at the beginning of the step

   The following variables are updated and returned

   *stiff*, the 6x6 material stiffness

   *update_state* must return *stress*, *statev*, *stiff*

Example
=======

.. code:: python

  from numpy import zeros, dot
  from mmd.material import MaterialModel
  from utils.errors import MatModLabError

  class MyElastic(MaterialModel):
      """Linear elastic material model"""
      name = 'my_elastic'

      @classmethod
      def param_names(cls, n):
          return ('K', 'G')

      def setup(self, **kwargs):
          K, Nu = self.parameters['E'], self.parameters['Nu']
	  if E <= 0.:
	      raise MatModLabError("negative Young's modulus")
	  if -1. >= Nu < .5:
	      raise MatModLabError("invalid Poisson's ratio")

      def update_state(self, time, dtime, temp, dtemp, energy, density,
                       F0, F1, strain, dstrain, elec_field,
		       stress, statev, **kwargs):
          # elastic properties
          E, Nu = self.parameters['E'], self.parameters['Nu']
	  K = E / 3 / (1 - 2 * Nu)
	  G = E / 2 / (1 + Nu)

          K3 = 3. * K
          G2 = 2. * G
          Lam = (K3 - G2) / 3.

          # elastic stiffness
          ddsdde = zeros((6,6))
          for i in range(3):
              for j in range(3):
                  ddsdde[j,i] = Lam
              ddsdde[i,i] = G2 + Lam
          for i in range(4, 6):
              ddsdde[i,i] = G

          # stress update
          stress += dot(ddsdde, dstrain)

	  return stress, statev, ddsdde
