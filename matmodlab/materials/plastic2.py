import numpy as np

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import VOIGT, deviatoric_part, double_dot, \
    symmetric_dyad, magnitude

ROOT2 = np.sqrt(2.)
ROOT3 = np.sqrt(3.)
ROOT23 = np.sqrt(2./3.)
TOLER = 1e-8

class NonhardeningPlasticMaterial(Material):
    name = "nonhardening-plastic"

    def __init__(self, **parameters):
        """Set up the Plastic material """

        # Check inputs
        E = parameters.get('E', 0.)
        Nu = parameters.get('Nu', 0.)
        Y = parameters.get('Y', 0.)
        errors = 0
        if E <= 0.0:
            errors += 1
            logger.error("Young's modulus E must be positive")
        if Nu > 0.5:
            errors += 1
            logger.error("Poisson's ratio > .5")
        if Nu < -1.0:
            errors += 1
            logger.error("Poisson's ratio < -1.")
        if Nu < 0.0:
            logger.warn("#---- WARNING: negative Poisson's ratio")
        if Y < 0:
            errors += 1
            logger.error('Yield strength must be positive')
        if Y < 1e-12:
            # zero strength -> assume the user wants elasticity
            logger.info('Zero strength detected, setting it to a larg number')
            Y = 1e60

        if errors:
            raise ValueError("stopping due to previous errors")

        self.params = {'E': E, 'Nu': Nu, 'Y': Y}

        # At this point, the parameters have been checked.  Now request
        # allocation of solution dependent variables.  The only variable
        # is the equivalent plastic strain
        self.num_sdv = 1
        self.sdv_names = ['EP_Equiv']

    def eval(self, time, dtime, temp, dtemp, F0, F,
             stran, d, stress, X, **kwargs):
        """Compute updated stress given strain increment"""

        #  material properties
        Y = self.params['Y']
        E = self.params['E']
        Nu = self.params['Nu']

        # Input parameter is yield in tension -> convert to yield in shear
        k = Y / ROOT3

        # Get the bulk, shear, and Lame constants
        K = E / 3. / (1. - 2. * Nu)
        G = E / 2. / (1. + Nu)

        K3 = 3. * K
        G2 = 2. * G
        G3 = 3. * G
        Lam = (K3 - G2) / 3.

        # elastic stiffness
        C = np.zeros((6,6))
        C[np.ix_(range(3), range(3))] = Lam
        C[range(3),range(3)] += G2
        C[range(3,6),range(3,6)] = G

        # Trial stress
        de = d * dtime
        T = stress + np.dot(C, de)

        # check yield
        S = deviatoric_part(T)
        RTJ2 = magnitude(S) / ROOT2
        f = RTJ2 - k

        if f <= TOLER:
            # Elastic loading, return what we have computed
            return T, X, C

        # Calculate the flow direction, projection direction
        M = S / ROOT2 / RTJ2
        N = S / ROOT2 / RTJ2
        A = 2 * G * M

        # Newton iterations to find Gamma
        Gamma = 0
        Ttrial = T.copy()
        for i in range(20):

            # Update all quantities
            dGamma = f * ROOT2 / double_dot(N, A)
            Gamma += dGamma

            T = Ttrial - Gamma * A
            S = deviatoric_part(T)
            RTJ2 = magnitude(S) / ROOT2
            f = RTJ2 - k

            # Calculate the flow direction, projection direction
            M = S / ROOT2 / RTJ2
            N = S / ROOT2 / RTJ2
            A = 2 * G * M
            Q = 2 * G * N

            if abs(dGamma + 1.) < TOLER + 1.:
                break

        else:
            raise RuntimeError('Newton iterations failed to converge')

        # Elastic strain rate and equivalent plastic strain
        dT = T - stress
        dep = Gamma * M
        dee = de - dep
        deqp = ROOT2 / ROOT3 * Gamma

        # Elastic stiffness
        D = C - 1 / double_dot(N, A) * symmetric_dyad(Q, A)

        # Equivalent plastic strain
        X[0] += deqp

        return T, X, D
