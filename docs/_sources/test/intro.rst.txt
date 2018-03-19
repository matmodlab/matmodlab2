
Material Model Testing: Introduction
####################################

.. topic:: See Also

   * `http://pytest.org/latest/ <Pytest>`_

Overview
========

Regression testing is critical to the model development process.  Only through thoroughly testing a model can confidence in model results be obtained.  Testing of material models involves:

* tests of individual program units (unit testing),
* verfication of model response to controlled inputs,
* validation of model response against data, and
* regression testing when changes to the model are made.

Regression tests in Matmodlab are special purpose problems that have a twofold purpose:

* test the individual components and core capabilities of Matmodlab itself;
* verification and validation (V&V) of material models.

In the first role, problems are fast running and exercise specific features of
Matmodlab in a unit-test type fashion. In the second, material models are
exercised through specific paths with known, or expected outcomes.
