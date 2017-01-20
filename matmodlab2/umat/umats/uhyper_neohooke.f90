SUBROUTINE UHYPER(I1B, I2B, JAC, U, DU, D2U, D3U, TEMP, NOEL, CMNAME, &
     INCMPFLAG, NSTATEV, STATEV, NFIELDV, FIELDV, FIELDVINC, &
     NPROPS, PROPS)
  ! ----------------------------------------------------------------------- !
  ! HYPERELASTIC COMPRESSIBLE NEO-HOOKEAN MATERIAL
  ! ----------------------------------------------------------------------- !
  ! CANNOT BE USED FOR PLANE STRESS
  ! PROPS(1)  - C10
  ! PROPS(2)  - D1
  ! ----------------------------------------------------------------------- !
  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  REAL(KIND=DP), PARAMETER :: ZERO=0._DP, ONE=1._DP, TWO=2._DP, THREE=3._DP
  CHARACTER*8, INTENT(IN) :: CMNAME
  INTEGER, INTENT(IN) :: NPROPS, NOEL, NSTATEV, INCMPFLAG, NFIELDV
  REAL(KIND=DP), INTENT(IN) :: I1B, I2B, JAC, PROPS(NPROPS), TEMP
  REAL(KIND=DP), INTENT(INOUT) :: U(2), DU(3), D2U(6), D3U(6), STATEV(NSTATEV)
  REAL(KIND=DP), INTENT(INOUT) :: FIELDV(NFIELDV), FIELDVINC(NFIELDV)
  REAL(KIND=DP) :: C10, D1, U_VOL, U_DEV
  ! -------------------------------------------------------------- UHYPER --- !

  C10 = PROPS(1)
  D1 = PROPS(2)

  ! ENERGY: U
  U_VOL = ONE / D1 * (JAC - ONE) ** 2
  U_DEV = C10 * (I1B - THREE)
  U(1) = U_VOL + U_DEV
  U(2) = U_DEV

  ! FIRST DERIVATIVE OF ENERGY: DU
  DU(1) = C10                       ! DU/DI1B
  DU(2) = ZERO                      ! DU/DI2B
  DU(3) = TWO / D1 * (JAC - ONE)    ! DU/DJ

  ! SECOND DERIVATIVE OF ENERGY: D2U
  D2U(1) = ZERO      ! D(DU/DI1B)/DI1B
  D2U(2) = ZERO      ! D(DU/DI2B)/DI2B
  D2U(3) = TWO / D1  ! D(DU/DJ)/DJ
  D2U(4) = ZERO      ! D(DU/DI1B)/DI2B
  D2U(5) = ZERO      ! D(DU/DI1B)/DJ
  D2U(6) = ZERO      ! D(DU/DI2B)/DJ

  ! THIRD DERIVATIVE OF ENERGY: D3U
  D3U(1) = ZERO      ! D(D(DU/DI1B)/DI1B)/DJ
  D3U(2) = ZERO      ! D(D(DU/DI2B)/DI2B)/DJ
  D3U(3) = ZERO      ! D(D(DU/DI1B)/DI2B)/DJ
  D3U(4) = ZERO      ! D(D(DU/DI1B)/DJ)/DJ
  D3U(5) = ZERO      ! D(D(DU/DI1B)/DJ)/DJ
  D3U(6) = ZERO      ! D(D(DU/DJ)/DJ)/DJ
  RETURN
END SUBROUTINE UHYPER
