import numpy as np

from ..core.material import Material


class ElasticMaterial(Material):
    """Implements linear elasticity

    Parameters
    ----------
    **kwds : dict
        Material parameters.  Recognized parameters are the Young's
        modules `E` and Poisson's ratio `Nu`

    """

    name = "pyelastic"

    def __init__(self, **parameters):
        E = parameters["E"]
        assert E > 0.0
        Nu = parameters["Nu"]
        assert -1.0 < Nu < 0.5
        self.G = E / 2.0 / (1.0 + Nu)
        self.K = E / 3.0 / (1.0 - 2.0 * Nu)

    def eval(self, time, dtime, temp, dtemp, F0, F, strain, d, stress, statev, **kwds):
        K3 = 3.0 * self.K
        G2 = 2.0 * self.G
        Lam = (K3 - G2) / 3.0
        # elastic stiffness
        ddsdde = np.zeros((6, 6))
        ddsdde[np.ix_(range(3), range(3))] = Lam
        ddsdde[range(3), range(3)] += G2
        ddsdde[range(3, 6), range(3, 6)] = self.G
        # stress update
        stress = stress + np.dot(ddsdde, d * dtime)
        return stress, statev, ddsdde
