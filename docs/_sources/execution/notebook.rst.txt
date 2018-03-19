.. _notebook:

Notebook
########

.. topic:: See Also

   * :ref:`basic_cli`
   * :ref:`intro_conventions`
   * :ref:`mps`
   * :ref:`viewer`

Overview
========

The Matmodlab.Notebook

* is an `IPython Notebook <http://ipython.org/notebook.html>`_ environment;
* requires the ``ipython`` and ``ipython-notebook`` modules;
* requires the `matplotlib <http://matplotlib.org>`_ and/or `Bokeh <http://bokeh.pydata.org/en/latest>`_ modules for data visualization; and
* allows interactive material model development.

The full power of the Matmodlab.Notebook is realized by interactive data visualization using `Bokeh <http://bokeh.pydata.org/en/latest>`_, which is a recommended install.

The Matmodlab.Notebook Environment
==================================

The Matmodlab.Notebook environment is invoked by launching the IPython Notebook through the ``mml`` procedure::

  $ mml notebook

After launching the notebook server and starting a new notebook, the following IPython magic should be executed in the notebook's first cell::

  %matmodlab

If Bokeh is installed, the :ref:`mps`\'s plotting methods use Bokeh for data visualization.  Otherwise, matplotlib is used as the plotting backend.  Matplotlib can be explicitly used as the plotting backend by specifying::

  %matmodlab matplotlib


.. todo::

   Finish this section
