
.. _comm_w_matmodlab:

Writing Messages to the Console and/or Log File
###############################################

Overview
========

Procedure *mml_comm* can be called from any user procedure to write messages
to the console and log file.

Interface
=========

::

   subroutine mml_comm(lop, string, intv, realv, charv)
      integer, intent(in) :: ierr
      character(120), intent(in) :: msg
      integer, intent(in) :: intv(*)
      real(8), intent(in) :: realv(*)
      character(8), intent(in) :: charv(*)

      ! coding

   end subroutine mml_comm

Parameters
==========

* *lop=1* writes an informational message to the log file

  *lop=-1* writes a warning message to the log file

  *lop=-2* writes an error message to the log file

  *lop=-3* writes an error message to the log file and stops the analysis

* *string* is the informational string
* *intv*
* *realv*
* *charv*


.. topic:: Abaqus Users:

   Procedure *stdb_abqerr*, which has the same interface as *mml_comm*, is also
   compiled and linked to user defined materials.
