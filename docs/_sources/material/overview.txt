.. _mat_overview:

Materials: Introduction
#######################

.. topic:: See Also

   * :ref:`Role of Material Model`
   * :ref:`defining_a_material`
   * :ref:`intro_conventions`

Overview
========

Material constitutive models provide the material's complete mechanical response. In other words, given the current mechanical state and an increment in deformation, the material model returns

* the updated stress,
* the material stiffness, and
* the updated state dependent variable, if applicable.

Matmodlab Material Library
==========================

Matmodlab has a limited :ref:`library of material models <mat_builtin>`.

User Developed Materials
========================

Matmodlab provides an API for :ref:`user developed materials <user_mats>`.


Combining Material Responses
============================

Generally, constitutive models are treated as defining the entire mechanical
response of a material and cannot be combined with other models. However,
material models can be combined with the following behaviors:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   expansion
   trs
   viscoelastic
