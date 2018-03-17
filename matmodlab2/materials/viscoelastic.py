from numpy import reshape, dot, zeros, asarray, array, exp
from .addon import AddonModel
from ..core.tensor import inv, deviatoric_part, matrix_rep, push, pull, \
    array_rep
from ..core.logio import logger

class ViscoelasticModel(AddonModel):
    """
    Addon viscoelastic model that adds viscoelastic relaxation to otherwise
    "instantaneous" model response.

    Parameters
    ----------
    wlf : array_like
        wlf[0] is the WLF C1 coefficient
        wlf[1] is the WLF C2 coefficient
        wlf[2] is the WLF Tref coefficient
    prony : array_like
        prony[i,0] is the ith Prony shear coefficient
        prony[i,1] is the ith Prony shear relax time coefficient

    State Dependent Variables
    -------------------------
         (1) : WLF AVERAGE SHIFT FACTOR
         (2) : AVERAGE NUMERICAL SHIFT FACTOR
       (3:8) : INSTANTANEOUS DEVIATORIC PK2 STRESS COMPONENTS AT START OF
               CURRENT TIME STEP WRITTEN USING THE INITIAL CONFIGURATION AS
               THE REFERENCE STATE
      (9:14) : XX,YY,ZZ,XY,YZ,ZX COMPONENTS OF VISCOELASTIC DEVIATORIC 2ND
               PIOLA KIRCHHOFF (PK2) STRESS FOR 1ST PRONY TERM USING THE
               INITIAL CONFIGURATION AS THE REFERENCE STATE
     (15:20) : VISCO DEV PK2 STRESS FOR 2ND PRONY TERM
     (21:26) : VISCO DEV PK2 STRESS FOR 3RD PRONY TERM
     (27:32) : VISCO DEV PK2 STRESS FOR 4TH PRONY TERM
     (33:38) : VISCO DEV PK2 STRESS FOR 5TH PRONY TERM
     (39:44) : VISCO DEV PK2 STRESS FOR 6TH PRONY TERM
     (45:50) : VISCO DEV PK2 STRESS FOR 7TH PRONY TERM
     (51:56) : VISCO DEV PK2 STRESS FOR 8TH PRONY TERM
     (57:62) : VISCO DEV PK2 STRESS FOR 9TH PRONY TERM
     (63:68) : VISCO DEV PK2 STRESS FOR 10TH PRONY TERM

    """
    name = '__viscoelastic__'
    def __init__(self, wlf, prony):

        # check data
        prony = asarray(prony)
        if len(prony.shape) != 2:
            logger.error('expected prony to be a 2D array')

        if len(prony) > 10:
            logger.error('At most 10 shear relaxation coefficients '
                          'can be specified')

        Goo = 1. - sum(prony[:,0])
        if Goo < 0.:
            logger.error('expected sum of shear Prony coefficients, '
                          'including infinity term to be one')

        # starting location of G and T Prony terms
        # setup viscoelastic params
        self.params = zeros(24)

        n = len(prony[:,0])
        I, J = (4, 14)
        self.params[I:I+n] = prony[:,0]
        self.params[J:J+n] = prony[:,1]

        # Ginf
        self.params[3] = Goo

        # wlf coefficients
        if wlf is not None:
            self.params[0] = wlf[0] # C1
            self.params[1] = wlf[1] # C2
            self.params[2] = wlf[2] # REF TEMP

        # Check sum of prony series coefficients
        psum  = sum(self.params[I:I+n])
        if any(self.params[I:I+n] < 0.):
            logger.error('Expected all shear Prony series coefficients > 0')

        if abs(psum) < 1e-10:
            logger.warning('Sum of normalized shear prony series coefficients\n'
                            'including infinity term is zero. normalized infinity\n'
                            'coefficient set to one for elastic response.')
            self.params[3] = 1.

        elif abs(psum - 1.) > 1e-3:
            message = ('Expected sum of normalized shear prony series\n'
                       'coefficients including inf term to be 1,\n'
                       'got {0}'.format(psum))
            if abs(psum - 1) < .03:
                logger.warning(message)
            else:
                logger.error(message)

        # Verify that all relaxation times are positive
        for (i, param) in enumerate(self.params[J:], start=J):
            if param <= 0.:
                logger.warning('Shear relaxation time term <=0, SETTING TO 1')
                self.params[i] = 1.

        # Allocate storage for visco data
        self.sdv_names = []

        # Shift factors
        self.sdv_names.extend(["SHIFT_{0}".format(i+1) for i in range(2)])

        # Instantaneous deviatoric PK2
        m = {0: "XX", 1: "YY", 2: "ZZ", 3: "XY", 4: "YZ", 5: "XZ"}
        self.sdv_names.extend(["TE_{0}".format(m[i]) for i in range(6)])

        # Visco elastic model supports up to 10 Prony series terms,
        # allocate storage for stress corresponding to each
        nprony = 10
        for l in range(nprony):
            for i in range(6):
                self.sdv_names.append("H{0}_{1}".format(l+1, m[i]))

        self.num_sdv = len(self.sdv_names)

    def sdvini(self, statev):
        nvisco = len(self.sdv_names)
        statev = zeros(nvisco)
        statev[:2] = 1.
        return statev

    def eval(self, kappa, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, initial_temp=0., **kwds):
        """Evaluate the viscoelastic relaxation model

        stress is updated in place
        """

        cfac = zeros(2)

        # Get the shift factors (stored in statev)
        self.shiftfac(dtime, time, temp, dtemp, F, statev)

        # change reference state on sodev from configuration at end of current
        # time step to initial configuration
        F = F.reshape((3,3))
        C = array_rep(dot(F.T, F), (6,))
        sigo = stress.copy()
        pk2o = pull(F, sigo)
        pk2odev = deviatoric_part(pk2o, metric=C)

        # reduced time step
        dtred = dtime / statev[1] / statev[0]

        # loop over the prony terms
        I, J = (4, 14)
        for k in range(10):
            j = k * 6
            # compute needed viscoelastic factors
            ratio = dtred / self.params[J+k]
            e = exp(-ratio)

            if ratio > 1e-3:
              # explicit calculation of (1-exp(-ratio))/ratio
                s = (1. - e) / ratio
            else:
               # taylor series calculation of (1 - exp(-ratio))/ratio
                s = 1. - .5 * ratio + 1. / 6. * ratio ** 2

           # update the viscoelastic state variable history for kth prony term
            for l in range(6):
                statev[8+j+l] = (e * statev[8+j+l] +
                                 self.params[4+k] * (s - e) * statev[2+l] +
                                 self.params[4+k] * (1 - s) * pk2odev[l])
            cfac[0] += (1 - s) * self.params[4+k]

        # compute decaying deviatoric stress
        pk2dev = zeros(6)
        for l in range(6):
            for k in range(10):
                j = k * 6
                pk2dev[l] += statev[8+j+l]

        # change reference state on decaying portion of deviatoric stress from
        # initial configuration to configuration at end of current time step
        sdev = push(F, pk2dev)

        # eliminate the pressure arising from the reference state changes used
        # in computing the decaying portion of the deviatoric stress
        tr = sum(sdev[:3])
        if abs(tr) > 1e-16:
            sdev[0] -= tr / 3.
            sdev[1] -= tr / 3.
            sdev[2] = -(sdev[0] + sdev[1])

        # compute total deviatoric stress
        sodev = deviatoric_part(sigo)
        sdev = sodev - sdev

        # total stress
        stress[:] = sdev.copy()
        pressure = -sum(sigo[:3]) / 3.
        stress[:3] -= pressure

        # instantaneous deviatoric stress with original configuration as
        # reference state
        statev[2:8] = pk2odev

        return cfac

    def shiftfac(self, dtime, time, temp, dtemp, F, statev):

        # retrieve the WLF parameters - thermal analysis
        C1, C2, Tref   = self.params[:3]
        temp_new = temp + dtemp

        # Evaluate the WLF shift factor at the average temp of the step
        at = 1. # Default shift factor
        ts_flag = 1
        if ts_flag:
            tempavg = .5 * (temp + temp_new)
            tempdiff = tempavg - Tref
            if abs(C1) < 1e-10:
                at = 1.
            elif C2 + tempdiff <= 1e-30:
                at = 1e-30
            else:
                log_at = C1 * tempdiff / (C2 + tempdiff)
                if log_at > 30.:
                    at = 1e30
                elif log_at < -30.:
                    at = 1e-30
                else:
                    at = 10. ** log_at

        # Store the numerical shift factor for WLF
        statev[0] = 1. / at
        statev[1] = 1.
