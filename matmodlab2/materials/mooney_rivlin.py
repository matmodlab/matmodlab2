from numpy import dot, trace

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import array_rep, det, I6


class MooneyRivlinMaterial(Material):
    name = "mooney-rivlin"

    def __init__(self, **parameters):
        """Set up the Mooney Rivlin material"""
        param_names = ["C10", "C01", "D1"]
        self.params = {}
        for param_name in param_names:
            value = parameters.pop(param_name, 0.0)
            self.params[param_name] = value
        if parameters:
            unused = ", ".join(parameters.keys())
            logger.warning("Unused parameters: {0}".format(unused))

        # Check inputs
        C10 = self.params["C10"]
        C01 = self.params["C01"]
        D1 = self.params["D1"]

        errors = 0
        if D1 <= 0.0:
            errors += 1
            logger.error("D1 must be > 0")

        G = 2.0 * (C10 + C01)
        if G <= 0:
            errors += 1
            logger.error("2 (C10 + C01) > 0")

        if errors:
            raise ValueError("stopping due to previous errors")

    def eval(self, time, dtime, temp, dtemp, F0, F, stran, d, stress, statev, **kwargs):
        """Compute updated stress given the updated deformation"""

        # elastic properties
        C10 = self.params["C10"]
        C01 = self.params["C01"]
        D1 = self.params["D1"]

        # elastic stiffness
        ddsdde = None

        # Reshape the deformation gradient
        F = F.reshape(3, 3)
        Jac = det(F)

        # left Cauchy deformation
        B = dot(F, F.T)
        Bsq = dot(B, B)

        incompressible = D1 > 1e4 * (C10 + C01)
        if incompressible:
            # enforce incompressibility
            Jac = 1

        # Invariants of B
        I1 = trace(B)
        I2 = 0.5 * (I1**2 - trace(Bsq))

        # Invariants of Cbar
        scale = sign(abs(Jac) ** (1.0 / 3.0), Jac)
        I1B = I1 / (scale**2)
        I2B = I2 / (scale**4)

        # convert symmetric tensors to arrays
        BBsq = array_rep(Bsq, (6,)) / scale**4
        BB = array_rep(B, (6,)) / scale**2

        if not incompressible:
            p = -2.0 / D1 * (Jac - 1.0)
        else:
            p = 0.0
        pb = p + 2.0 / 3.0 / Jac * (C10 * I1B + 2.0 * C01 * I2B)
        stress = 2.0 / Jac * ((C10 + C01 * I1B) * BB - C01 * BBsq) - pb * I6

        return stress, statev, ddsdde


def sign(x, y):
    return x if y > 0.0 else -x
