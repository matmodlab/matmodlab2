import numpy as np

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import (symsq, det, invariants, inv, push,
                           dyad, symshuffle, II1, II5, I6)

class PolynomialHyperelasticMaterial(Material):
    name = "polyhyper"

    def __init__(self, **params):
        param_names = ['C10','C01','C20','C11','C02','C30','C21',
                       'C12','C03','D1', 'D2', 'D3']
        self.params = np.zeros(len(param_names))
        for (i, name) in enumerate(param_names):
            value = params.pop(name, 0.)
            self.params[i] = value
        if params:
            ignored = ', '.join(params.keys())
            logger.warning('The following parameters were '
                           'ignored: {0}'.format(ignored))

    def eval(self, time, dtime, temp, dtemp, F0, F,
             stran, d, stress, statev, **kwargs):
        """Compute updated stress given the updated deformation"""

        # Right deformation and invariants
        C = symsq(F)
        I1, I2, I3 = invariants(C)
        Jac = np.sqrt(I3)

        scale = Jac ** (-1. / 3.)

        IB1 = I1 * scale ** 2
        IB2 = I2 * scale ** 4

        # elastic energy
        u, du, d2u, d3u = uhyper(self.params, IB1, IB2, Jac)

        # PK2 stress
        Ci = inv(C)
        B = np.zeros((3, 6))
        B[0,:] = scale ** 2 * (I6 - I1 * Ci / 3.)
        B[1,:] = scale ** 4 * (I1 * I6 - C - 2. * I2 * Ci / 3.)
        B[2,:] = .5 * Jac * Ci

        # second Piola-Kirchhoff stress: derivative of energy wrt C
        T = np.zeros(6)
        for i in range(6):
            T[i] = 2.* np.sum(du * B[:, i], axis=0)

        # Cauchy stress
        stress = push(F, T)

        # The elastic stiffness
        CICI = dyad(Ci, Ci)
        CIC = dyad(Ci, C)
        CCI = dyad(C, Ci)
        LCICI = symshuffle(Ci, Ci)
        ICI = dyad(I6, Ci)
        CII = dyad(Ci, I6)

        # DA/DC
        DA = np.zeros((3,6))
        DA[0,:] = d2u[0] * B[0,:] + d2u[3] * B[1,:] + d2u[4] * B[2,:]
        DA[1,:] = d2u[3] * B[0,:] + d2u[1] * B[1,:] + d2u[5] * B[2,:]
        DA[2,:] = d2u[4] * B[0,:] + d2u[5] * B[1,:] + d2u[2] * B[2,:]

        # DB/DC
        DB = np.zeros((3,6,6))

        TERM1 = CII + ICI
        TERM2 = I1 * (LCICI + CICI / 3.)
        DB[0,:,:] = 1. / 3. * scale ** 2 * (-TERM1 + TERM2)

        TERM1 = 3. / 2. * (II1 - II5) + (CIC + CCI)
        TERM2 = -I1 * (CII + ICI) + I2 * (LCICI + 2. / 3. * CICI)
        DB[1,:,:] = 2. /  3. * scale ** 4 * (TERM1 + TERM2)

        DB[2,:,:] = 1. / 4. * Jac * (CICI - 2. * LCICI)

        # ISOCHORIC PART
        Liso = np.zeros((6,6))
        for I in range(2):
            TERM1 = 4. * (dyad(DA[I,:], B[I,:]) + du[I] * DB[I,:,:])
            Liso += TERM1
        Liso = .5 * (Liso + Liso.T)

        # VOLUMETRIC PART
        Lvol = 4. * (dyad(DA[2,:], B[2,:]) + du[2] * DB[2,:,:])
        Lvol = .5 * (Lvol + Lvol.T)

        # the constitutive equations for the hyperelastic material gives the
        # stress in the spatial configuration (cauchy stress). however, the
        # constitituve response is in terms of the lie derivative, or truesdell
        # rate, of the kirchoff stress. the jaummann rate is an alternative lie
        # derivative, given by
        #                              _   .
        #                              s = s + s.w - w.s
        #       _
        # where s is the jaummann rate of the cauchy stress s. using the jaummann
        # rate, the constitutive response in the corotated frame is given by
        #                                 _
        #                                 s = D:d
        #       _
        # where c is the jaummann tangent stiffness tensor and is defined
        # implicity from
            #                           _
        #                           s = c:d + d.s + s.d
        # from which
        #
        #           D = c + .5 (iik sjl + iil sjk + sik ijl + sil ijk)
        L = Liso + Lvol
        C = push(F, L)
        TERM1 = symshuffle(I6, stress)
        TERM2 = symshuffle(stress, I6)
        D = C + (TERM1 + TERM2)

        # Return it all
        return stress, statev, D

def uhyper(c, IB1, IB2, Jac):
    """HYPERELASTIC POLYNOMIAL MODEL"""
    # Energies
    u = np.zeros(2)
    u[1] += c[0] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 0)
    u[1] += c[1] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 1)
    u[1] += c[2] * ((IB1 - 3.) ** 2) * ((IB2 - 3.) ** 0)
    u[1] += c[3] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 1)
    u[1] += c[4] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 2)
    u[1] += c[5] * ((IB1 - 3.) ** 3) * ((IB2 - 3.) ** 0)
    u[1] += c[6] * ((IB1 - 3.) ** 2) * ((IB2 - 3.) ** 1)
    u[1] += c[7] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 2)
    u[1] += c[8] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 3)
    if (c[9] > 1e-14): u[0] += 1. / c[9] * (Jac - 1.) ** 2
    if (c[10] > 1e-14): u[0] += 1. / c[10] * (Jac - 1.) ** 4
    if (c[11] > 1e-14): u[0] += 1. / c[11] * (Jac - 1.) ** 6
    u[0] = u[0] + u[1]

    # First Derivatives
    du = np.zeros(3)
    du[0] += 1.0 * c[0] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 0)
    du[0] += 2.0 * c[2] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 0)
    du[0] += 1.0 * c[3] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 1)
    du[0] += 3.0 * c[5] * ((IB1 - 3.) ** 2) * ((IB2 - 3.) ** 0)
    du[0] += 2.0 * c[6] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 1)
    du[0] += 1.0 * c[7] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 2)

    du[1] += 1.0 * c[1] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 0)
    du[1] += 1.0 * c[3] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 0)
    du[1] += 2.0 * c[4] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 1)
    du[1] += 1.0 * c[6] * ((IB1 - 3.) ** 2) * ((IB2 - 3.) ** 0)
    du[1] += 2.0 * c[7] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 1)
    du[1] += 3.0 * c[8] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 2)

    if (c[9] > 1e-14): du[2] += 2.0 / c[9] * (Jac - 1.) ** 1
    if (c[10] > 1e-14): du[2] += 4.0 / c[10] * (Jac - 1.) ** 3
    if (c[11] > 1e-14): du[2] += 6.0 / c[11] * (Jac - 1.) ** 5

    # Second Derivatives
    ddu = np.zeros(6)
    ddu[0] += 2.0 * c[2] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 0)
    ddu[0] += 6.0 * c[5] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 0)
    ddu[0] += 2.0 * c[6] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 1)


    ddu[1] += 2.0 * c[4] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 0)
    ddu[1] += 6.0 * c[8] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 1)
    ddu[1] += 2.0 * c[7] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 0)

    ddu[3] += 1.0 * c[3] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 0)
    ddu[3] += 2.0 * c[6] * ((IB1 - 3.) ** 1) * ((IB2 - 3.) ** 0)
    ddu[3] += 2.0 * c[7] * ((IB1 - 3.) ** 0) * ((IB2 - 3.) ** 1)

    if (c[9] > 1e-14): ddu[2] += 2.0 / c[9] * (Jac - 1.) ** 0
    if (c[10] > 1e-14): ddu[2] += 12.0 / c[10] * (Jac - 1.) ** 2
    if (c[11] > 1e-14): ddu[2] += 30.0 / c[11] * (Jac - 1.) ** 4

    # Third Derivatives
    dddu = np.zeros(6)
    if (c[10] > 1e-14): dddu[5] += 24.0 / c[10] * (Jac - 1.) ** 1
    if (c[11] > 1e-14): dddu[5] += 120.0 / c[11] * (Jac - 1.) ** 3

    return u, du, ddu, dddu
