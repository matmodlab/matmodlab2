import numpy as np

from .logio import logger
from .material import Material
from .mmlabpack import VOIGT, dot, dev, iso, mag, invariants, I6

ROOT2 = np.sqrt(2.)
ROOT23 = np.sqrt(2./3.)

class ElasticMaterial(Material):
    """Implements linear elasticity

    Parameters
    ----------
    **kwds : dict
        Material parameters.  Recognized parameters are the Young's
        modules `E` and Poisson's ratio `Nu`

    """
    name = 'pyelastic'
    def __init__(self, **parameters):
        E = parameters['E']
        assert E > 0.
        Nu = parameters['Nu']
        assert -1. < Nu < .5
        self.G = E / 2. / (1. + Nu)
        self.K = E / 3. / (1. - 2. * Nu)

    def eval(self, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        K3 = 3. * self.K
        G2 = 2. * self.G
        Lam = (K3 - G2) / 3.
        # elastic stiffness
        ddsdde = np.zeros((6,6))
        ddsdde[np.ix_(range(3), range(3))] = Lam
        ddsdde[range(3),range(3)] += G2
        ddsdde[range(3,6),range(3,6)] = self.G
        # stress update
        stress = stress + dot(ddsdde, d * dtime)
        return stress, statev, ddsdde

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

            s = dev(stress)
            N = np.sqrt(2.) * A4 * I6 + s / mag(s)
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
                ep += iso(dstress) / (3.0 * self.params['K'])
                ep += dev(dstress) / (2.0 * self.params['G'])
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
        return (3.0 * self.params['K'] * iso(A) +
                2.0 * self.params['G'] * dev(A))

class VonMises(Material):
    name = 'vonmises'
    def __init__(self, **kwargs):
        '''Set up the von Mises material

        '''
        # Check inputs
        K = kwargs.get('K', 0.)
        G = kwargs.get('G', 0.)
        H = kwargs.get('H', 0.)
        Y0 = kwargs.get('Y0', 0.)
        BETA = kwargs.get('BETA', 0.)

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
            logger.warn('negative Poisson\'s ratio')
        if abs(Y0) <= 1.E-12:
            Y0 = 1.0e99
        if errors:
            raise ValueError('stopping due to previous errors')

        self.params = {'K': K, 'G': G, 'Y0': Y0, 'H': H, 'BETA': BETA}

        # Register State Variables
        self.sdv_names = ['EQPS', 'Y',
                          'BS_XX', 'BS_YY', 'BS_ZZ', 'BS_XY', 'BS_XZ', 'BS_YZ',
                          'SIGE']
        self.num_sdv = len(self.sdv_names)

    def sdvini(self, statev):
        Y0 = self.params['Y0']
        return np.array([0.0, Y0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def eval(self, time, dtime, temp, dtemp, F0, F,
             stran, d, stress, statev, **kwargs):
        """Compute updated stress given strain increment"""

        idx = lambda x: self.sdv_names.index(x.upper())
        bs = np.array([statev[idx('BS_XX')],
                       statev[idx('BS_YY')],
                       statev[idx('BS_ZZ')],
                       statev[idx('BS_XY')],
                       statev[idx('BS_YZ')],
                       statev[idx('BS_XZ')]])
        yn = statev[idx('Y')]

        de = d / VOIGT * dtime

        iso = de[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        dev = de - iso

        stress_trial = stress + 3.0 * self.params['K'] * iso + 2.0 * self.params['G'] * dev

        xi_trial = stress_trial - bs
        xi_trial_eqv = self.eqv(xi_trial)

        if xi_trial_eqv <= yn:
            statev[idx('SIGE')] = xi_trial_eqv
            return stress_trial, statev, None
        else:
            N = xi_trial - xi_trial[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            N = N / (ROOT23 * xi_trial_eqv)
            deqps = (xi_trial_eqv - yn) / (3.0 * self.params['G'] + self.params['H'])
            dps = 1. / ROOT23 * deqps * N

            stress_final = stress_trial - 2.0 * self.params['G'] / ROOT23 * deqps * N

            bs = bs + 2.0 / 3.0 * self.params['H'] * self.params['BETA'] * dps

            statev[idx('EQPS')] += deqps
            statev[idx('Y')] += self.params['H'] * (1.0 - self.params['BETA']) * deqps
            statev[idx('BS_XX')] = bs[0]
            statev[idx('BS_YY')] = bs[1]
            statev[idx('BS_ZZ')] = bs[2]
            statev[idx('BS_XY')] = bs[3]
            statev[idx('BS_YZ')] = bs[4]
            statev[idx('BS_XZ')] = bs[5]
            statev[idx('SIGE')] = self.eqv(stress_final - bs)
            return stress_final, statev, None


    def eqv(self, sig):
        # Returns sqrt(3 * rootj2) = sig_eqv = q
        s = sig - sig[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return 1. / ROOT23 * np.sqrt(np.dot(s[:3], s[:3]) + 2 * np.dot(s[3:], s[3:]))
