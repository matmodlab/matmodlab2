import numpy as np

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import VOIGT, deviatoric_part, isotropic_part, \
    invariants, I6, magnitude

class PlasticMaterial(Material):
    """Implements linear pressure dependent plasticity

    Parameters
    ----------
    **kwds : dict
        Material parameters.  Recognized parameters are the bulk
        modules `K`, shear modules `G`, yield strength in shear `A1`,
        and pressure dependence term `A4`.

    """
    name = 'pyplastic'
    def __init__(self, **parameters):

        # Linear elastic bulk modulus
        K = parameters.get('K', 0.)

        # Linear elastic shear modulus
        G = parameters.get('G', 0.)

        # A1: Intersection of the yield surface with the sqrt(J2) axis (pure
        # shear).
        # sqrt(J2) = r / sqrt(2); r = sqrt(2*J2)
        # sqrt(J2) = q / sqrt(3); q = sqrt(3*J2)
        A1 = parameters.get('A1', 0.)

        # Pressure dependence term.
        #   A4 = -d(sqrt(J2)) / d(I1)
        # always positive
        A4 = parameters.get('A4', 0.)

        if abs(A1) <= 1.E-12:
            A1 = 1.0e99

        # Check the input parameters
        errors = 0
        if K <= 0.0:
            errors += 1
            logger.error('Bulk modulus K must be positive')
        if G <= 0.0:
            errors += 1
            logger.error('Shear modulus G must be positive')
        nu = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)
        if nu > 0.5:
            errors += 1
            logger.error('Poisson\'s ratio > .5')
        if nu < -1.0:
            errors += 1
            logger.error('Poisson\'s ratio < -1.')
        if nu < 0.0:
            logger.warn('#negative Poisson\'s ratio')
        if A1 <= 0.0:
            errors += 1
            logger.error('A1 must be positive nonzero')
        if A4 < 0.0:
            errors += 1
            logger.error('A4 must be non-negative')
        if errors:
            raise ValueError('stopping due to previous errors')

        # Save the new parameters
        self.params = {'K': K, 'G': G, 'A1': A1, 'A4': A4}

        # Register State Variables
        self.sdv_names = ['EP_XX', 'EP_YY', 'EP_ZZ', 'EP_XY', 'EP_XZ', 'EP_YZ',
                          'I1', 'ROOTJ2', 'YROOTJ2', 'ISPLASTIC']
        self.num_sdv = len(self.sdv_names)

    def eval(self, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        """Compute updated stress given strain increment """

        sigsave = np.copy(stress)

        # Define helper functions and unload params/state vars
        A1 = self.params['A1']
        A4 = self.params['A4']
        idx = lambda x: self.sdv_names.index(x.upper())
        ep = statev[idx('EP_XX'):idx('EP_YZ')+1]

        # Compute the trial stress and invariants
        stress = stress + self.dot_with_elastic_stiffness(d / VOIGT * dtime)
        i1, rootj2 = invariants(stress, mechanics=True)
        if rootj2 - (A1 - A4 * i1) <= 0.0:
            statev[idx('ISPLASTIC')] = 0.0
        else:
            statev[idx('ISPLASTIC')] = 1.0

            s = deviatoric_part(stress)
            N = np.sqrt(2.) * A4 * I6 + s / magnitude(s)
            N = N / np.sqrt(6.0 * A4 ** 2 + 1.0)
            P = self.dot_with_elastic_stiffness(N)

            # 1) Check if linear drucker-prager
            # 2) Check if trial stress is beyond the vertex
            # 3) Check if trial stress is in the vertex
            i1_p, rootj2_p = invariants(P, mechanics=True)
            if (A4 != 0.0 and
                    i1 > A1 / A4 and
                    rootj2 / (i1 - A1 / A4) < rootj2_p / i1_p):
                dstress = stress - A1 / A4 / 3.0 * I6
                # convert all of the extra strain into plastic strain
                ep += isotropic_part(dstress) / (3.0 * self.params['K'])
                ep += deviatoric_part(dstress) / (2.0 * self.params['G'])
                stress = A1 / A4 / 3.0 * I6
            else:
                # not in vertex; do regular return
                lamb = ((rootj2 - A1 + A4 * i1) / (A4 * i1_p + rootj2_p))
                stress = stress - lamb * P
                ep += lamb * N

            # Save the updated plastic strain
            statev[idx('EP_XX'):idx('EP_YZ')+1] = ep

        i1, rootj2 = invariants(stress, mechanics=True)
        statev[idx('I1')] = i1
        statev[idx('ROOTJ2')] = rootj2
        statev[idx('YROOTJ2')] = A1 - A4 * i1

        return stress, statev, None

    def dot_with_elastic_stiffness(self, A):
        return (3.0 * self.params['K'] * isotropic_part(A) +
                2.0 * self.params['G'] * deviatoric_part(A))
