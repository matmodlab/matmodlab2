! Include the following in each signature file
! python module gmd__user__routines
!     interface gmd_user_interface
!         subroutine log_message(message)
!             intent(callback) log_message
!             character*(*) :: message
!         end subroutine log_message
!         subroutine log_error(message)
!             intent(callback) log_error
!             character*(*) :: message
!             real intent(callback) :: log_message
!         end subroutine log_error
!     end interface gmd_user_interface
! end python module gmd__user__routines
!
! Then, in each fortran function
! use gmd__user__routines
! intent(callback) log_error
! external log_error
! intent(callback) log_message
! external log_message

subroutine mml_comm(ierr, msg, intv, realv, charv)
  implicit none
  integer, intent(in) :: ierr
  character(120), intent(in) :: msg
  integer, intent(in) :: intv(*)
  real(8), intent(in) :: realv(*)
  character(8), intent(in) :: charv(*)
  character(1) :: c(2)
  character(12) :: s
  character(220) :: string
  integer :: i, j, k, n, ls, ii, ir, ic
  external log_message
  external log_warning
  external log_error

  ii=1; ir=1; ic=1; j=1; k=1
  forall(i=1:len(string)) string(i:i) = " "
  do i=1,len_trim(msg)
     c(1) = msg(j:j); c(2) = msg(j+1:j+1)
     n = 1
     if (c(1) == "%") then
        n = 2
        if (c(2) == "i" .or. c(2) == "I") then
           ! replace integers
           write(s,"(i5)") intv(ii)
           ii = ii + 1
        else if (c(2) == "c" .or. c(2) == "C") then
           ! replace characters
           s = charv(ic)
           ic = ic + 1
        else if (c(2) == "r" .or. c(2) == "R") then
           ! replace reals
           write(s,"(e12.6)") realv(ir)
           ir = ir + 1
        else
           s = c(2)
           n = 1
        end if
        s = adjustl(s)
        ls = len(trim(s))
        string(k:k+ls) = trim(s)
        k = k + ls
     else
        string(k:k) = c(1)
        k = k + 1
     end if
     j = j + n
  end do

  if (ierr == -3) then
     string = "*** ERROR: " // adjustl(string)
     call log_error(string)
  else if (ierr == -1) then
     string = "*** WARNING: " // adjustl(string)
     call log_warning(string)
  else if (ierr == -2) then
     string = "*** ERROR: " // adjustl(string)
     call log_warning(trim(string))
  else
     call log_message(trim(string))
  end if
  return
end subroutine mml_comm

function upper(s1)  result (s2)
  character(*) :: s1
  character(len(s1)) :: s2
  character :: ch
  integer,parameter :: duc=ichar('a')-ichar('A')
  integer :: i
  do i=1,len(s1)
     ch = s1(i:i)
     if (ch >= 'A'.and.ch <= 'Z') ch = char(ichar(ch)+duc)
     s2(i:i) = ch
  end do
end function upper

subroutine bombed(msg)
  implicit none
  character(120), intent(in) :: msg
  integer :: intv(1)
  real(8) :: realv(1)
  character(8) :: charv(1)
  integer :: ierr
  ierr = -3
  call mml_comm(ierr, msg, intv, realv, charv)
end subroutine bombed

subroutine faterr(msg)
  implicit none
  character(120), intent(in) :: msg
  integer :: intv(1)
  real(8) :: realv(1)
  character(8) :: charv(1)
  integer :: ierr
  ierr = -3
  call mml_comm(ierr, msg, intv, realv, charv)
end subroutine faterr

subroutine logmes(msg)
  implicit none
  character(120), intent(in) :: msg
  integer :: intv(1)
  real(8) :: realv(1)
  character(8) :: charv(1)
  integer :: ierr
  ierr = 1
  call mml_comm(ierr, msg, intv, realv, charv)
end subroutine logmes

subroutine tokens(n, sa, ca)
  integer, intent(in) :: n
  character*(*), intent(in) :: sa(n)
  character(len=1), intent(out) :: ca(*)
  character(len=1), parameter :: pipe='|'
  character(len=1), parameter :: blank=' '
  integer :: i, knt, nchr, ichr
  knt = 0
  do 2 i = 1, n
      do 3 nchr = len(sa(i)), 1, -1
3   if(sa(i)(nchr:nchr).ne.blank) go to 7
7     do 1 ichr = 1, nchr
         knt = knt + 1
         ca(knt) = sa(i)(ichr:ichr)
1     continue
      knt = knt + 1
      ca(knt) = pipe
2  continue
  return
end subroutine tokens
