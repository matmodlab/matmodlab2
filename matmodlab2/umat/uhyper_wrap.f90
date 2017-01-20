MODULE HYPERELASTIC

  IMPLICIT NONE
  INTEGER, PARAMETER :: DP=SELECTED_REAL_KIND(14)
  REAL(KIND=DP), PARAMETER :: HALF=.5_DP, MONE=-1._DP
  REAL(KIND=DP), PARAMETER :: ZERO=0._DP, ONE=1._DP, TWO=2._DP, THREE=3._DP
  REAL(KIND=DP), PARAMETER :: FOUR=4._DP
  REAL(KIND=DP), PARAMETER :: THIRD=ONE/THREE, P2THIRD=TWO/THREE, P3HALF=THREE/TWO
  REAL(KIND=DP), PARAMETER :: FOURTH=ONE/FOUR, SIX=6._DP
  REAL(KIND=DP), PARAMETER :: IIMI4(6,6) = RESHAPE(&
                         [ZERO,  ONE,  ONE, ZERO, ZERO, ZERO, &
                           ONE, ZERO,  ONE, ZERO, ZERO, ZERO, &
                           ONE,  ONE, ZERO, ZERO, ZERO, ZERO, &
                          ZERO, ZERO, ZERO, MONE, ZERO, ZERO, &
                          ZERO, ZERO, ZERO, ZERO, MONE, ZERO, &
                          ZERO, ZERO, ZERO, ZERO, ZERO, MONE], SHAPE(IIMI4))

CONTAINS

  SUBROUTINE HYPEREL(NPROPS, PROPS, TEMP, F, NOEL, CMNAME, INCMPFLAG, &
       NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, PK2, CVOL, CISO)
    ! ----------------------------------------------------------------------- !
    ! HYPERELASTIC MATERIAL MODEL
    ! CALLS A UYHPER MODEL AND FORMULATES THE STRESS AND STIFFNESS ARRAYS
    ! ----------------------------------------------------------------------- !
    USE TENSALG, ONLY : DET,INVARS,INV,DYAD,SYMSHUFF,SYMSQ,I6,II1,II5
    ! --- PASSED ARGUMENTS
    CHARACTER*8, INTENT(IN) :: CMNAME
    INTEGER, INTENT(IN) :: NPROPS, NOEL, NSTATV, INCMPFLAG, NFLDV
    REAL(KIND=DP), INTENT(IN) :: PROPS(NPROPS), TEMP, F(3,3)
    REAL(KIND=DP), INTENT(INOUT) :: STATEV(NSTATV), FIELDV(NFLDV), DFIELDV(NFLDV)
    REAL(KIND=DP), INTENT(OUT) :: PK2(6), CVOL(6,6), CISO(6,6)
    ! --- LOCAL VARIABLES
    INTEGER :: I
    REAL(KIND=DP) :: JAC, C(6)
    REAL(KIND=DP) :: U(2), DU(3), D2U(6), D3U(6)
    REAL(KIND=DP) :: I1, I2, I3, I1B, I2B
    REAL(KIND=DP) :: SCALE, CI(6), A(3), DA(3,6), B(3,6), DB(3,6,6)
    REAL(KIND=DP) :: TERM1(6,6), TERM2(6,6)
    REAL(KIND=DP), DIMENSION(6,6) :: CII, ICI, CICI, LCICI, CCI, CIC
    ! ----------------------------------------------------------- HYPEREL --- !

    ! DEFORMATION TENSOR
    ! C = FT.F
    C = SYMSQ(F)

    ! JACOBIAN
    JAC = DET(F)

    ! INVARIANTS OF C
    CALL INVARS(C, I1, I2, I3)

    ! INVARIANTS OF CBAR
    SCALE = JAC ** (-ONE / THREE)
    I1B = I1 * (SCALE ** 2)
    I2B = I2 * (SCALE ** 4)

    CALL UHYPER(I1B, I2B, JAC, U, DU, D2U, D3U, TEMP, NOEL, CMNAME, &
         INCMPFLAG, NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, &
         NPROPS, PROPS)

    ! UPDATE THE STRESS AND MATERIAL STIFFNESS
    ! THE NOTATION BELOWS FOLLOWS THE APPENDIX OF ***TITLE*** WHERE
    ! THE SECOND PIOLA-KIRCHOFF STRESS IS GIVEN BY
    !
    !                           DU   DU  DIB
    !                   PK2 = 2 -- = --- --- = A.B
    !                           DC   DIB DC
    !
    ! WHERE A IS AN ARRAY CONTAINING THE DERIVATIVES OF U WRT ISOCHORIC
    ! INVARIANTS AND B IS AN ARRAY OF TENSORS CONTAINING THE DERIVATIVES OF
    ! THE ISOCHORIC INVARIANTS WRT C

    ! HELPER QUANTITIES
    CI = INV(C)

    A = DU
    B(1,1:6) = SCALE ** 2 * (I6 - THIRD * I1 * CI)
    B(2,1:6) = SCALE ** 4 * (I1 * I6 - C - P2THIRD * I2 * CI)
    B(3,1:6) = HALF * JAC * CI

    ! SECOND PIOLA-KIRCHHOFF STRESS: DERIVATIVE OF ENERGY WRT C
    FORALL(I=1:6) PK2(I) = TWO * SUM(A(1:3) * B(1:3, I))

    ! USING THE NOTATION FROM ABOVE, THE MATERIAL STIFFNESS L IS
    !
    !                      L  = 4(DA.B + A.DB)
    !
    !            DA   DA  DIB   D2U
    ! WHERE DA = -- = --- --- = ----.B
    !            DC   DIB DC    DIB2
    !
    !          DB
    ! AND DB = --
    !          DC
    !
    ! NOTE THAT DA AND B CONTAIN SECOND ORDER TENSORS SO THAT THE ARRAY DA.B
    ! IS AN ARRAY OF DYADIC PRODUCTS OF SECOND ORDER TENSORS (FOURTH ORDER
    ! TENSORS). DB IS AN ARRAY OF FOURTH ORDER TENSORS.

    ! HELPER QUANTITIES
    CICI = DYAD(CI, CI)
    CIC = DYAD(CI, C)
    CCI = DYAD(C, CI)
    LCICI = SYMSHUFF(CI)
    ICI = DYAD(I6, CI)
    CII = DYAD(CI, I6)

    ! DA/DC
    DA=ZERO
    DA(1,1:6) = D2U(1) * B(1,1:6) + D2U(4) * B(2,1:6) + D2U(5) * B(3,1:6)
    DA(2,1:6) = D2U(4) * B(1,1:6) + D2U(2) * B(2,1:6) + D2U(6) * B(3,1:6)
    DA(3,1:6) = D2U(5) * B(1,1:6) + D2U(6) * B(2,1:6) + D2U(3) * B(3,1:6)

    ! DB/DC
    DB = ZERO

    TERM1 = CII + ICI
    TERM2 = I1 * (LCICI + THIRD * CICI)
    DB(1,1:6,1:6) = THIRD * SCALE ** 2 * (-TERM1 + TERM2)

    TERM1 = P3HALF * (II1 - II5) + (CIC + CCI)
    TERM2 = -I1 * (CII + ICI) + I2 * (LCICI + P2THIRD * CICI)
    DB(2,1:6,1:6) = P2THIRD * SCALE ** 4 * (TERM1 + TERM2)

    DB(3,1:6,1:6) = FOURTH * JAC * (CICI - TWO * LCICI)

    ! ISOCHORIC PART
    CISO = ZERO
    DO I=1,2
       TERM1 = FOUR * (DYAD(DA(I,1:6), B(I,1:6)) + DU(I) * DB(I,1:6,1:6))
       CISO = CISO + TERM1
    END DO
    CISO = HALF * (CISO + TRANSPOSE(CISO))

    ! VOLUMETRIC PART
    CVOL = FOUR * (DYAD(DA(3,1:6), B(3,1:6)) + DU(3) * DB(3,1:6,1:6))
    CVOL = HALF * (CVOL + TRANSPOSE(CVOL))

  END SUBROUTINE HYPEREL

  ! ************************************************************************* !

  SUBROUTINE PUSHALL(F, PK2, CVOL, CISO, CFAC, STRESS, DDSDDE)
    ! ----------------------------------------------------------------------- !
    USE TENSALG, ONLY : PUSH, SYMSHUFF, I6
    ! --- PASSED ARGUMENTS
    REAL(KIND=DP), INTENT(IN) :: F(3,3), PK2(6), CVOL(6,6), CISO(6,6), CFAC(2)
    REAL(KIND=DP), INTENT(OUT) :: STRESS(6), DDSDDE(6,6)
    REAL(KIND=DP) :: TERM1(6,6), TERM2(6,6)
    ! ----------------------------------------------------------- PUSHALL --- !
    STRESS = PUSH(F, PK2)
    TERM1 = (ONE - CFAC(1)) * CISO + (ONE - CFAC(2)) * CVOL
    DDSDDE = PUSH(F, TERM1)

    ! THE CONSTITUTIVE EQUATIONS FOR THE HYPERELASTIC MATERIAL GIVES THE
    ! STRESS IN THE SPATIAL CONFIGURATION (CAUCHY STRESS). HOWEVER, THE
    ! CONSTITITUVE RESPONSE IS IN TERMS OF THE LIE DERIVATIVE, OR TRUESDELL
    ! RATE, OF THE KIRCHOFF STRESS. THE JAUMMANN RATE IS AN ALTERNATIVE LIE
    ! DERIVATIVE, GIVEN BY
    !                              _   .
    !                              S = S + S.W - W.S
    !       _
    ! WHERE S IS THE JAUMMANN RATE OF THE CAUCHY STRESS S. USING THE JAUMMANN
    ! RATE, THE CONSTITUTIVE RESPONSE IN THE COROTATED FRAME IS GIVEN BY
    !                                 _   _
    !                                 S = C:D
    !       _
    ! WHERE C IS THE JAUMMANN TANGENT STIFFNESS TENSOR AND IS DEFINED
    ! IMPLICITY FROM
    !                           _
    !                           S = C:D + D.S + S.D
    ! FROM WHICH
    !           _
    !           C = C + .5 (Iik Sjl + Iil Sjk + Sik Ijl + Sil Ijk)
    TERM1 = SYMSHUFF(I6, STRESS)
    TERM2 = SYMSHUFF(STRESS, I6)
    DDSDDE = DDSDDE + (TERM1 + TERM2)

  END SUBROUTINE PUSHALL

  ! ************************************************************************* !

  SUBROUTINE NUMERICAL_JACOBIAN(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
       NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, F, DDSDDE)
    ! THIS PROCEDURE NUMERICALLY COMPUTES AND RETURNS THE JACOBIAN MATRIX
    !
    !                      J = (JIJ) = (DSIGI/DEPSJ).
    !
    ! THE COMPONENTS OF JSUB ARE COMPUTED NUMERICALLY USING A CENTERED
    ! DIFFERENCING SCHEME WHICH REQUIRES TWO CALLS TO THE MATERIAL MODEL
    ! SUBROUTINE FOR EACH ELEMENT OF PK2. THE CENTERING IS ABOUT THE POINT
    !                       EPS = EPSOLD + D * DT,
    ! WHERE D IS THE RATE-OF-STRAIN ARRAY.
    USE TENSALG, ONLY: PUSH
    REAL(KIND=DP), PARAMETER :: TOL=1.E-6_DP
    CHARACTER*8, INTENT(IN) :: CMNAME
    INTEGER, INTENT(IN) :: NPROPS, NOEL, NSTATV, INCMPFLAG, NFLDV
    REAL(KIND=DP), INTENT(IN) :: PROPS(NPROPS), TEMP, STATEV(NSTATV)
    REAL(KIND=DP), INTENT(INOUT) :: FIELDV(NFLDV), DFIELDV(NFLDV)
    REAL(KIND=DP), INTENT(IN) :: F(3,3)
    REAL(KIND=DP), INTENT(OUT) :: DDSDDE(6,6)
    INTEGER :: N
    REAL(KIND=DP) :: EPS, DF(3,3), CISO(6,6), CVOL(6,6)
    REAL(KIND=DP) :: SVP(NSTATV), FP(3,3), TP(6), SP(6)
    REAL(KIND=DP) :: SVM(NSTATV), FM(3,3), TM(6), SM(6)
    ! ---------------------------------------------------------- JACOBIAN --- !
    DDSDDE = ZERO
    EPS = SQRT(EPSILON(ONE))
    DO N = 1, 6
       DF = DGEDDG(EPS/TWO, N, F)
       FP = F + DF
       SVP = STATEV
       CALL HYPEREL(NPROPS, PROPS, TEMP, FP, NOEL, CMNAME, INCMPFLAG, NSTATV,&
            SVP, NFLDV, FIELDV, DFIELDV, TP, CVOL, CISO)
       SP = PUSH(FP, TP)
       FM = F - DF
       SVM = STATEV
       CALL HYPEREL(NPROPS, PROPS, TEMP, FM, NOEL, CMNAME, INCMPFLAG, NSTATV,&
            SVM, NFLDV, FIELDV, DFIELDV, TM, CVOL, CISO)
       SM = PUSH(FM, TM)
       DDSDDE(:, N) = (SP - SM) / EPS
    END DO
    DDSDDE = HALF * (DDSDDE + TRANSPOSE(DDSDDE))
  END SUBROUTINE NUMERICAL_JACOBIAN

  ! ************************************************************************* !

  FUNCTION DGEDDG(EPS, N, F)
    ! -----------------------------------------------------------------------
    !  !
    ! PERTURB THE DEFORMATION GRADIENT
    ! -----------------------------------------------------------------------
    !  !
    USE TENSALG, ONLY: DYAD
    INTEGER, INTENT(IN) :: N
    REAL(KIND=DP), INTENT(IN) :: EPS, F(3,3)
    REAL(KIND=DP) :: DGEDDG(3,3)
    REAL(KIND=DP), PARAMETER :: O=1._DP,Z=0._DP
    REAL(KIND=DP), PARAMETER :: E(3,3)=RESHAPE((/O,Z,Z,Z,O,Z,Z,Z,O/),(/3,3/))
    INTEGER, PARAMETER :: IJ(2,6)=RESHAPE((/1,1,2,2,3,3,1,2,1,3,2,3/),(/2,6/))
    REAL(KIND=DP) :: EI(3), EJ(3), EIJ(3,3), EJI(3,3)
    ! -----------------------------------------------------------------------
    !  !
    EI = E(IJ(1,N),:3)
    EJ = E(IJ(2,N),:3)
    EIJ = DYAD(EI, EJ)
    EJI = DYAD(EJ, EI)
    DGEDDG = EPS / TWO * (MATMUL(EIJ, F) + MATMUL(EJI, F))
  END FUNCTION DGEDDG

  ! ************************************************************************* !

  SUBROUTINE NEOHOOKE(NPROPS, PROPS, F, SIG, C)
    ! --------------------------------------------------------------------- !
    ! COMPRESSIBLE NEO-HOOKEAN HYPERELASTIC MATERIAL
    !
    ! NOTES
    ! -----
    ! SYMMETRIC TENSOR ORDERING : XX, YY, ZZ, XY, YZ, ZX
    ! --------------------------------------------------------------------- !
    USE TENSALG, ONLY: I6
    INTEGER, INTENT(IN) :: NPROPS
    REAL(DP), INTENT(IN) :: PROPS(NPROPS), F(3,3)
    REAL(DP), INTENT(INOUT) :: SIG(6), C(6,6)
    INTEGER :: I, J
    REAL(DP) :: EE(6), EEP(3), BBP(3), BBN(3,3)
    REAL(DP) :: C10, D1, EG, EK, EG23, PR
    REAL(DP) :: JAC, SCALE, FB(3,3), BB(6), TRBBAR

    ! ELASTIC PROPERTIES
    C10 = PROPS(1)
    D1 = PROPS(2)

    ! JACOBIAN AND DISTORTION TENSOR
    JAC = F(1,1) * F(2,2) * F(3,3) - F(1,2) * F(2,1) * F(3,3) + F(1,2) * F(2&
         &,3) * F(3,1) + F(1,3) * F(3,2) * F(2,1) - F(1,3) * F(3,1) * F(2,2) &
         &- F(2,3) * F(3,2) * F(1,1)
    SCALE = JAC **(-ONE / THREE)
    FB = SCALE * F

    ! DEVIATORIC LEFT CAUCHY-GREEN DEFORMATION TENSOR
    BB(1) = FB(1,1) * FB(1,1) + FB(1,2) * FB(1,2) + FB(1,3) * FB(1,3)
    BB(2) = FB(2,1) * FB(2,1) + FB(2,2) * FB(2,2) + FB(2,3) * FB(2,3)
    BB(3) = FB(3,1) * FB(3,1) + FB(3,2) * FB(3,2) + FB(3,3) * FB(3,3)
    BB(4) = FB(2,1) * FB(1,1) + FB(2,2) * FB(1,2) + FB(2,3) * FB(1,3)
    BB(5) = FB(3,1) * FB(2,1) + FB(3,2) * FB(2,2) + FB(3,3) * FB(2,3)
    BB(6) = FB(3,1) * FB(1,1) + FB(3,2) * FB(1,2) + FB(3,3) * FB(1,3)
    TRBBAR = SUM(BB(1:3)) / THREE
    EG = TWO * C10 / JAC
    EK = TWO / D1 * (TWO * JAC - ONE)
    PR = TWO / D1 * (JAC - ONE)

    ! CAUCHY STRESS
    SIG = EG * (BB - TRBBAR * I6) + PR * I6

    ! SPATIAL STIFFNESS
    EG23 = EG * TWO / THREE
    C(1,1) =  EG23 * (BB(1) + TRBBAR) + EK
    C(1,2) = -EG23 * (BB(1) + BB(2)-TRBBAR) + EK
    C(1,3) = -EG23 * (BB(1) + BB(3)-TRBBAR) + EK
    C(1,4) =  EG23 * BB(4) / TWO
    C(1,5) = -EG23 * BB(5)
    C(1,6) =  EG23 * BB(6) / TWO
    C(2,2) =  EG23 * (BB(2) + TRBBAR) + EK
    C(2,3) = -EG23 * (BB(2) + BB(3)-TRBBAR) + EK
    C(2,4) =  EG23 * BB(4) / TWO
    C(2,5) =  EG23 * BB(5) / TWO
    C(2,6) = -EG23 * BB(6)
    C(3,3) =  EG23 * (BB(3) + TRBBAR) + EK
    C(3,4) = -EG23 * BB(4)
    C(3,5) =  EG23 * BB(5) / TWO
    C(3,6) =  EG23 * BB(6) / TWO
    C(4,4) =  EG * (BB(1) + BB(2)) / TWO
    C(4,5) =  EG * BB(6) / TWO
    C(4,6) =  EG * BB(5) / TWO
    C(5,5) =  EG * (BB(1) + BB(3)) / TWO
    C(5,6) =  EG * BB(4) / TWO
    C(6,6) =  EG * (BB(2) + BB(3)) / TWO
    FORALL(I=1:6,J=1:6,J<I) C(I,J) = C(J,I)
    RETURN
  END SUBROUTINE NEOHOOKE
END MODULE HYPERELASTIC

! *************************************************************************** !

SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD, RPL,DDSDDT,DRPLDE,DRPLDT,&
     & STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME, NDI,NSHR,NTENS&
     &,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT, CELENT,F0,F1,NOEL,NPT,LAYER&
     &,KSPT,KSTEP,KINC)
  ! -------------------------------------------------------------------------
  !  !
  !     UMAT INTERFACE FOR FINITE STRAIN PROPELLANT MODEL
  ! -------------------------------------------------------------------------
  !  !
  !     CANNOT BE USED FOR PLANE STRESS
  !     PROPS(1)  - C10
  !     PROPS(2)  - K1
  ! -------------------------------------------------------------------------
  !  !
  USE HYPERELASTIC, ONLY: DP, ZERO, ONE, TWO, THREE, HYPEREL, &
       NUMERICAL_JACOBIAN, NEOHOOKE, PUSHALL
  IMPLICIT NONE

  ! --- PASSED ARGUMENTS
  ! -------------------------------------------------------------------------
  !  !
  CHARACTER*8, INTENT(IN) :: CMNAME
  INTEGER, INTENT(IN) :: NDI, NSHR, NTENS, NSTATV, NPROPS
  INTEGER, INTENT(IN) :: NOEL, NPT, LAYER, KSPT, KSTEP, KINC
  REAL(KIND=DP), INTENT(IN) :: SSE, SPD, SCD, RPL, DRPLDT, TIME(2), DTIME
  REAL(KIND=DP), INTENT(IN) :: TEMP, DTEMP, PNEWDT, CELENT
  REAL(KIND=DP), INTENT(INOUT) :: STRESS(NTENS), STATEV(NSTATV)
  REAL(KIND=DP), INTENT(INOUT) :: DDSDDE(NTENS, NTENS)
  REAL(KIND=DP), INTENT(INOUT) :: DDSDDT(NTENS), DRPLDE(NTENS)
  REAL(KIND=DP), INTENT(IN) :: STRAN(NTENS), DSTRAN(NTENS)
  REAL(KIND=DP), INTENT(IN) :: PREDEF(1), DPRED(1), PROPS(NPROPS), COORDS(3)
  REAL(KIND=DP), INTENT(IN) :: DROT(3, 3), F0(3, 3), F1(3, 3)
  ! --- LOCAL VARIABLES
  INTEGER, PARAMETER :: NFLDV=1, INCMPFLAG=1
  REAL(KIND=DP) :: FIELDV(NFLDV), DFIELDV(NFLDV), ERR
  REAL(KIND=DP) :: PK2(6), CVOL(6,6), CISO(6,6), CFAC(2)
  REAL(KIND=DP), ALLOCATABLE :: SNEO(:), CNEO(:,:), S(:), C(:,:), CNUM(:,:)
  INTEGER :: INTV(1), DBGFLG, I
  CHARACTER*120 :: MSG
  REAL(KIND=DP) :: REALV(1)
  CHARACTER*8 :: CHARV(1)
  ! ---------------------------------------------------------------- UMAT --- !
  DDSDDE = 0._DP
  CALL HYPEREL(NPROPS, PROPS, TEMP, F1, NOEL, CMNAME, INCMPFLAG, &
       NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, PK2, CVOL, CISO)
  CFAC = ZERO
  CALL PUSHALL(F1, PK2, CVOL, CISO, CFAC, STRESS, DDSDDE)

  DBGFLG = 0
  SELECT CASE (DBGFLG)
  CASE (1)
     ! NUMERICAL JACOBIAN
     CALL NUMERICAL_JACOBIAN(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
          NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, F1, DDSDDE)
  CASE (2)
     ALLOCATE(SNEO(6))
     CALL NEOHOOKE(2, (/PROPS(1),PROPS(2)/), F1, SNEO, DDSDDE)
     DEALLOCATE(SNEO)
  CASE (3)

     ALLOCATE(SNEO(6))
     ALLOCATE(CNEO(6,6))
     ALLOCATE(CNUM(6,6))
     ALLOCATE(S(6))
     ALLOCATE(C(6,6))
     C = DDSDDE
     S = STRESS
     WHERE (ABS(C) < EPSILON(ONE))
        C = ONE
     END WHERE
     WHERE (ABS(S) < EPSILON(ONE))
        S = ONE
     END WHERE

     ! NUMERICAL JACOBIAN
     CALL NUMERICAL_JACOBIAN(NPROPS, PROPS, TEMP, NOEL, CMNAME, INCMPFLAG, &
          NSTATV, STATEV, NFLDV, FIELDV, DFIELDV, F1, CNUM)

     ! NEO HOOKE JACOBIAN
     CALL NEOHOOKE(2, (/PROPS(1),PROPS(2)/), F1, SNEO, CNEO)
     ERR = MAXVAL(((SNEO - STRESS) / S))
     IF (ERR > 1.E-8_DP) THEN
        WRITE(MSG, "(A,ES9.3)") "ERROR IN STRESS: ", ERR
        CALL STDB_ABQERR(-1, MSG, INTV, REALV, CHARV)
     END IF

     ERR = MAXVAL(ABS((CNEO - DDSDDE) / C))
     IF (ERR > 1.E-8_DP) THEN
        WRITE(MSG, "(A,ES9.3)") "ERROR IN STIFFNESS: ", ERR
        CALL STDB_ABQERR(-1, MSG, INTV, REALV, CHARV)
     END IF

     DEALLOCATE(SNEO, CNEO, CNUM, S, C)
  END SELECT


END SUBROUTINE UMAT
