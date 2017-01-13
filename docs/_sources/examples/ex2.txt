
Example 2: Uniaxial Stress
##########################

This example demonstrates exercising the elastic material model through a path
of uniaxial stress. The example input below is found in ``matmodlab/examples/uniaxial_stress.py``

The Example Script
..................

::

   from matmodlab import *

   # Create the material point simulator
   mps = MaterialPointSimulator('uniaxial_stress')

   # Define the material
   mps.Material('elastic', {'K': 1.35e11, 'G': 5.3e10})

   # Define the stress step
   mps.StressStep(components=(1, 0, 0), frames=25, scale=1e6)

   # Run the simulation
   mps.run()

How Does the Script Work?
.........................

This section describes each part of the example script

``from matmodlab import *``

This statement makes the Matmodlab objects accessible to the script.

``mps = MaterialPointSimulator('uniaxial_stress')``

This statement creates a new material point simlator object named ``uniaxial_stress``.  The variable ``mps`` is assigned to the simulator.

``mps.Material('elastic', {'K': 1.35e11, 'G': 5.3e10})``

This statement defines the material model to be the ``elastic`` material and
defines the bulk modulus ``K`` and shear modulus ``G`` to ``1.35e11`` and
``5.3e10``, respectively.

``mps.StressStep(components=(1, 0, 0), frames=25, scale=1e6)``

This statement defines an analysis step through which the material will be
exercised. The step is defined by the tensor ``components`` :math:`(1, 0, 0)`,
representing the ``xx``, ``yy``, and ``zz`` components of the stress tensor. A
``scale`` factor of ``1e6`` is applied to each component.

* The first 3 values of ``components`` represent the ``xx``, ``yy``, and
  ``zz`` components of the tensor describing the deformation path. The ``xy``,
  ``yz``, and ``xz`` components are implicitly 0.

``mps.run()``

This statement runs the material through the defined deformation path.
