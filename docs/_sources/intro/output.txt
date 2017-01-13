.. _mml_out_dbs:

Matmodlab File Formats
######################

.. topic:: See Also

   * :ref:`mps`

Overview
========

Matmodlab.Simulator writes to and Matmodlab.Viewer reads from a variety of output file formats.  The format of output is determined by the *output* keyword to the ``MaterialPointSimulator``.  This section describes the different file formats.

Output File Formats
===================

dbx
---

The default output database format.  It is written by Matmodlab.Simulator and read by Matmodlab.Viewer.  Output is written to dbx format if ``output=DBX``.

exo
---

`ExodusII <http://sourceforge.net/projects/exodusii>`_ output database format.  It is written by Matmodlab.Simulator and read by Matmodlab.Viewer.  Output is written to exo format if ``output=EXO``.

xls
---

Microsoft Excel spreadsheet.  It is written by Matmodlab.Simulator if the `xlwt <http://pypi.python.org/pypi/xlwt>`_ module is installed and read by Matmodlab.Viewer if the `xlrd <http://pypi.python.org/pypi/xlrd>`_ module is installed. Output is written to xls format if ``output=XLS``.

xlsx
----

Microsoft Excel spreadsheet.  It is written by Matmodlab.Simulator and read by Matmodlab.Viewer if the `openpyxl <http://pypi.python.org/pypi/openpyxl>`_ module is installed. Output is written to xlsx format if ``output=XLSX``.

Other File Formats
==================

log
---

Log file.  It is written by Matmodlab.Simulator.
