.. _mml_out_dbs:

Matmodlab File Formats
######################

Overview
========

*Matmodlab2* writes to one of two file formats.  The format of output is determined by the *db_fmt* keyword to the ``MaterialPointSimulator``.  The two formats are

npz
   Compressed ``numpy`` array format.  This is the default format and is chosen
   by letting ``db_fmt='npz'``

exo

   `ExodusII <http://sourceforge.net/projects/exodusii>`_ output database
   format.  Output is written to exo format if ``db_fmt=exo``.
