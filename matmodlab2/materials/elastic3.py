from numpy import zeros, ix_
from ..core.logio import logger
from ..core.material import Material
from ..core.deformation import strain_from_stretch
from ..core.tensor import polar_decomp, isotropic_part, deviatoric_part, array_rep


class ElasticMaterialTotal(Material):
    name = "elastic3"

    def __init__(self, **parameters):
        """Set up the Elastic material"""

        # Check inputs
        E = parameters.get("E", 0.0)
        Nu = parameters.get("Nu", 0.0)
        k = parameters.get("k", 0.0)

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
            logger.warning("#---- WARNING: negative Poisson's ratio")
        if errors:
            raise ValueError("stopping due to previous errors")
        self.params = {"E": E, "Nu": Nu, "k": k}

    def eval(self, time, dtime, temp, dtemp, F0, F, stran, d, stress, statev, **kwargs):
        """Compute updated stress given strain increment"""

        # elastic properties
        k = self.params["k"]
        E = self.params["E"]
        Nu = self.params["Nu"]

        # Get the bulk, shear, and Lame constants
        K = E / 3.0 / (1.0 - 2.0 * Nu)
        G = E / 2.0 / (1.0 + Nu)

        K3 = 3.0 * K
        G2 = 2.0 * G
        Lam = (K3 - G2) / 3.0

        # elastic stiffness
        ddsdde = zeros((6, 6))
        ddsdde[ix_(range(3), range(3))] = Lam
        ddsdde[range(3), range(3)] += G2
        ddsdde[range(3, 6), range(3, 6)] = G

        R, U = polar_decomp(F.reshape(3, 3))
        e = strain_from_stretch(array_rep(U, (6,)), k)

        # stress update
        stress = K3 * isotropic_part(e) + G2 * deviatoric_part(e)

        return stress, statev, ddsdde
