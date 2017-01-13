.. _intro_conventions:

Conventions
###########

Overview
========

Conventions used through out Matmodlab are described.


Dimension
=========

Material models are always called with full 3D tensors.

Vector Storage
==============

Vector components are stored as

.. math::

   v_i = \{v_x, v_y, v_z\}

Tensor Storage
==============

Second-order symmetric tensors are stored as 6x1 arrays with the following
component ordering

.. math::
   :label: order-symtens

   A_{ij} = \{A_{xx}, A_{yy}, A_{zz}, A_{xy}, A_{yz}, A_{xz}\}

Nonsymmetric, second-order tensors are stored as 9x1 arrays in row major
ordering, i.e.,

.. math::

   B_{ij} = \{B_{xx}, B_{xy}, B_{xz},
              B_{yx}, B_{yy}, B_{yz},
              B_{zx}, B_{zy}, B_{zz}\}

Engineering Strains
-------------------

The shear components of strain-like tensors are sent to the material model as
engineering strains, i.e.

.. math::

   \epsilon_{ij} = \{\epsilon_{xx}, \epsilon_{yy}, \epsilon_{zz}, 2\epsilon_{xy}, 2\epsilon_{yz}, 2\epsilon_{xz}\}
           = \{\epsilon_{xx}, \epsilon_{yy}, \epsilon_{zz}, \gamma_{xy}, \gamma_{yz}, \gamma_{xz}\}

.. note::

   The tensor order is runtime configurable using *ordering* keyword to the ``MaterialModel`` constructor.  See :ref:`invoke_user_f` for details.
