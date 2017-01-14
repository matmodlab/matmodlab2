
.. _environment:

Environment Settings
####################

Overview
========

*Matmodlab2* sets up and performs execution of input scripts in a customized
environment (not to be confused with Python virtural environments). The
environment can be modifed by user environment files that are read at the
beginning of each job.

Environment File Locations
==========================

*Matmodlab2* searches for the optional user environment file, ``mml_userenv.py``, in two locations, in the following order:

1) The location specified by the environment variable ``MML_USERENV``
2) The current working directory.

The value of a parameter is the the last definition encountered, meaning that
the order of precedence for user settings is the current working directory and
``MML_USERENV``.

Environment files use Python syntax, meaning that entries will have the
following form::

  parameter = value

All usual Python conventions apply.

Recognized Environment Settings and Defaults
============================================

Below are the recognized environment settings and their defaults. Any of these
settings can be changed by specifying a different value in a user environment
file.

.. note::

   When specifying environment settings in a user environment file, the
   setting must have the same type as the default. If the default is a list,
   the user setting is inserted in to the default list. If the default is a
   dictionary, it is updated with the user setting.

IO Settings
-----------

verbosity

  *Matmodlab2* will print more, or less, information during job execution. Possible values are ``0``, ``1``, and ``2``. Set the value to ``0`` to suppress printing of information. Set the value to ``2``, or higher, to print increase the amount of information printed. The default value is ``1``.

SQA

   *Matmodlab2* will run extra software quality checks.

loglevel

   Set the `logging` module logger level.  Value must be a valid `logging` level.

prepend_to_sys_path

    If `True`, prepend the environment file's directory to the python search path.  If a string (or list of strings), *Matmodlab2* will assume that the string (or list of strings) is a directory path and put it (or them) on the search path.
