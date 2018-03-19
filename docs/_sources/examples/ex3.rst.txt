
Example 3: Uniaxial Stress, Mixed Mode
######################################

This example demonstrates exercising the elastic material model through a path
of uniaxial stress, using a mixed mode step. The example input below is found in ``matmodlab/examples/uniaxial_stress.py``

The Example Script
..................

::

   from matmodlab import *

   # Create the material point simulator
   mps = MaterialPointSimulator('uniaxial_stress-1', output=EXO)

   # Define the material
   mps.Material('elastic', {'K': 1.35e11, 'G': 5.3e10})

   # Define the mixed mode step
   mps.MixedStep(components=(1, 0, 0), descriptors='ESS', frames=25, scale=.02)

   # Run the simulation
   mps.run()

   # Launch the viewer
   mps.view()

How Does the Script Work?
.........................

This section describes each part of the example script.

``from matmodlab import *``

This statement makes the Matmodlab objects accessible to the script.

``mps = MaterialPointSimulator('uniaxial_stress-1', output=EXO)``

This statement creates a new material point simlator object named ``uniaxial_stress-1``.  The variable ``mps`` is assigned to the simulator.  The ``output`` format is ``EXO`` (ExodusII) instead of the default ``DBX``.

``mps.Material('elastic', {'K': 1.35e11, 'G': 5.3e10})``

This statement defines the material model to be the ``elastic`` material and
defines the bulk modulus ``K`` and shear modulus ``G`` to ``1.35e11`` and
``5.3e10``, respectively.

``mps.MixedStep(components=(1, 0, 0), descriptors='ESS', frames=25, scale=.02)``

This statement defines an analysis step through which the material will be
exercised. The step is defined by the tensor ``components`` :math:`(1, 0, 0)`,
representing the ``xx`` component of the strain tensor and the ``yy``, and
``zz`` components of the stress tensor (as designated by the ``descriptors``
``"ESS"``). The step is run in ``25`` ``frames`` (increments) and a ``scale`` of
``.02`` is applied to each component. Note the following:

* The first 3 values of ``components`` represent the ``xx``, ``yy``, and
  ``zz`` components of the tensor describing the deformation path. The ``xy``,
  ``yz``, and ``xz`` components are implicitly ``0``.

* The i\ :superscript:`th` ``descriptor`` designates the physical
  interpretation of the i\ :superscript:`th` ``component`` with ``E``
  representing strain and ``S`` representing stress.

``mps.run()``

This statement runs the material through the defined deformation path.

``mps.view()``

This statement launches the Matmodlab viewer (the chaco and traitsui Python modules must be installed).
