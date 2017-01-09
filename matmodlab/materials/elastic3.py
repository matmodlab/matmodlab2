from numpy import dot, zeros, ix_, eye
from ..core.logio import logger
from ..core.material import Material
from ..core.mmlabpack import logm, powm, asarray, \
    polar_decomp, iso, dev, VOIGT

class ElasticMaterialTotal(Material):
    name = "elastic3"
    def __init__(self, **parameters):
        """Set up the Elastic material"""

        # Check inputs
        E = parameters.get('E', 0.)
        Nu = parameters.get('Nu', 0.)
        k = parameters.get('k', 0.)

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
        if errors:
            raise ValueError("stopping due to previous errors")
        self.params = {'E': E, 'Nu': Nu, 'k': k}

    def eval(self, time, dtime, temp, dtemp, F0, F,
             stran, d, stress, statev, **kwargs):
        """Compute updated stress given strain increment"""

        # elastic properties
        k = self.params['k']
        E = self.params['E']
        Nu = self.params['Nu']

        # Get the bulk, shear, and Lame constants
        K = E / 3. / (1. - 2. * Nu)
        G = E / 2. / (1. + Nu)

        K3 = 3. * K
        G2 = 2. * G
        Lam = (K3 - G2) / 3.

        # elastic stiffness
        ddsdde = zeros((6,6))
        ddsdde[ix_(range(3), range(3))] = Lam
        ddsdde[range(3),range(3)] += G2
        ddsdde[range(3,6),range(3,6)] = G

        R, U = polar_decomp(F.reshape(3,3))
        if abs(k) <= 1e-12:
            e = logm(U)
        else:
            e = 1. / 2 / k * (powm(U, 2*k) - eye(3))

        # convert strain to an array
        e = asarray(e, 6)

        # stress update
        stress = K3 * iso(e) + G2 * dev(e)

        return stress, statev, ddsdde
