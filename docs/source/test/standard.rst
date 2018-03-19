.. _rtest:

Matmodlab StandardMaterialTest
##############################

.. topic:: See Also

   .

Overview
========

This section describes the ``StandardMaterialTest`` class in ``matmodlab.tests.testconf``.

.. todo::

   This documentation is out of date.

.. _basic_test_ex:

Standard Material Test
======================

::

  from matmodlab import *

  path_table_id = "path_table"

  class TestPathTable(TestBase):
      def __init__(self):
          self.job = path_table_id
          self.keywords = ["fast", "feature", "path", "table", "builtin"]

  @matmodlab
  def run_path_table():

      job = path_table_id
      logger = Logger(job)
      path = """0E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00
                1E+00 1E-01 0E+00 0E+00 0E+00 0E+00 0E+00
                2E+00 0E+00 0E+00 0E+00 0E+00 0E+00 0E+00"""
      driver = Driver("Continuum", path, kappa=0.0, path_input="table",
                      step_multiplier=10.0, cfmt="222222", cols=range(7),
                      tfmt="time", logger=logger)
      parameters = {"K":1.350E+11, "G":5.300E+10}
      material = Material("elastic", parameters, logger=logger)
      mps = MaterialPointSimulator(job, driver, material, logger=logger, d=d)
      mps.run()
      return

  if __name__ == "__main__":
      run_path_table()

A test is created by subclassing ``TestBase``. Minimally, a test defines
``job`` and ``keywords`` attributes. The ``job`` attribute is used
internally for test identification and ``keywords`` for test filtering and
organization. The module containing the test must also define a
``run_<job>`` function (where ``<job>`` is replaced with the actual
``job`` of the test) to run the actual simulation. For each test so defined,
Matmodlab expects the existence of a base file ``<job>.base_exo`` containing
the expected, or baseline, results. Matmodlab also expects, on exercising
``run_<job>``, the creation of the results file ``<job>.exo``. At the
completion of the test, ``<job>.exo`` is compared to ``<job>.base_exo`` and
differences (if any) determined by ``exodiff``.

Command-Line Interface
======================

The ``test`` subcommand of ``mml`` gathers, runs, and analyzes tests. To run
tests with Matmodlab, be sure that ``matmodlab/bin`` is on your path and
execute::

  $ mml test

``mml test`` will create a results directory ``TestResults.<platform>``, where
``<platform>`` is is the machine platform (as determined by Python's
``sys.platform``) on which the tests are being run. The following files and
directories will be produced by ``mml test`` in the
``TestResults.<platform>``, directory ::

  $ ls
  TestsResults.darwin completed_tests.db mmd/ summary.html

``completed_tests.db`` is a database file containing information on all
completed tests and ``summary.html`` is an html summary file, viewable in any
web browser.

``mml test`` searches for test specification files in the ``matmodlab/tests``
directory and directories in the ``tests`` section of the configuration file.
Test files are python files whose names match ``(?:^|[\\b_\\.-])[Tt]est``.
Matmodlab supports

mml test Options
----------------

The full list of options to ``mml test`` is::

  usage: mml test [-h] [-k K] [-K K] [-X] [-j J] [--no-tear-down] [--html]
                  [--overlay] [-E] [-l] [--rebaseline]
                  [sources [sources ...]]

  mml test: run the matmodlab tests. By default, tests are found in
  MML_ROOT/tests and any other directories and/or files found in the tests group
  of the MATMODLABRC file (if any).

  positional arguments:
    sources         [Optional] directores and/or files to find matmodlab tests.
                    The default directories will not be searched.

  optional arguments:
    -h, --help      show this help message and exit
    -k K            Keywords to include [default: ]
    -K K            Keywords to exclude [default: ]
    -X              Do not stop on test initialization failure (tests that fail
                    to initialize will be skipped) [default: False]
    -j J            Number of simutaneous tests to run [default: ]
    --no-tear-down  Do not tear down passed tests on completion [default: ]
    --html          Write html summary of results (negates tear down) [default: ]
    --overlay       Create overlays of failed tests with baseline (negates tear
                    down) [default: ]
    -E              Do not use matmodlabrc configuration file [default: False]
    -l              List tests and exit [default: False]
    --rebaseline    Rebaseline test in PWD [default: False]

TestBase API
============

.. class:: TestBase

   Instances of the TestBase represent individual tests. The class is intended
   to be used as a base class, with specific tests being implemented by
   concrete subclasses. The class implements the interface needed by mml test
   to allow it to drive the test, and methods that the test code can use check
   for and report various kinds of failure. Each instance of TestBase will run
   a single test.

Required Attributes of TestBase
-------------------------------

.. attribute:: TestBase.keywords

   List of keywords identifying the test. Each test must define
   one of *long*, *medium*, *fast*.

.. attribute:: TestBase.job

   The test identifier.

Definable Attributes of TestBase
--------------------------------

.. attribute:: TestBase.base_res

   Base result file name  [default: ``job.base_exo``]

.. attribute:: TestBase.exodiff

   ``exodiff`` diff file [default: ``tests/base.exodiff``]

Useful Read-Only Attributes of TestBase
---------------------------------------

.. attribute:: TestBase.test_dir

   The directory in which the test will be run

Methods
=======

As described in :ref:`basic_test_ex`, minimally, a test subclasses ``TestBase`` and defines a ``job`` and ``keywords``, Matmodlab will set up the test, run, and perform post processing.  Optionally, a test may define the following methods.

.. method:: TestBase.setup(*args, **kwargs)

   Test setup. Minimally, setup should check for existence of needed files and
   create the test directory.

.. method:: TestBase.pre_hook(*args, **kwargs)

   Called before each test is run and after setup.  The base pre_hook performs a no-op.

.. method:: TestBase.run()

   Run the test.  Set test.status to one of FAILED_TO_RUN, FAILED, DIFFED, PASSED

.. method:: TestBase.tear_down(force=0)

   Tear down the test. The standard tears down the test by removing the test directory (if test passed).

   :parameter force: Force tear down even if test failed
   :type force: int

.. method:: TestBase.post_hook(*args, **kwargs)

   Run after test is run.  The standard post_hook performs a no-op.

.. method:: TestBase.make_test_dir()

   Make the test directory TestBase.test_dir
