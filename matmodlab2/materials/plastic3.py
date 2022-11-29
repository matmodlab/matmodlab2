from numpy import zeros, ix_, sqrt

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import dyad, deviatoric_part, double_dot, magnitude

TOLER = 1e-8
ROOT3, ROOT2 = sqrt(3.0), sqrt(2.0)


class HardeningPlasticMaterial(Material):
    name = "hardening-plastic"

    def __init__(self, **parameters):
        """Set up the Plastic material"""
        param_names = ["E", "Nu", "Y0", "Y1", "m"]
        self.params = {}
        for (i, name) in enumerate(param_names):
            self.params[name] = parameters.pop(name, 0.0)
        if parameters:
            unused = ", ".join(parameters.keys())
            logger.warning("Unused parameters: {0}".format(unused))

        # Check inputs
        E = self.params["E"]
        Nu = self.params["Nu"]
        Y0 = self.params["Y0"]
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
        if Y0 < 0:
            errors += 1
            logger.error("Yield strength must be positive")
        if Y0 < 1e-12:
            # zero strength -> assume the user wants elasticity
            logger.warning("Zero strength detected, setting it to a larg number")
            self.params["Y0"] = 1e60

        if errors:
            raise ValueError("stopping due to previous errors")

        # At this point, the parameters have been checked.  Now request
        # allocation of solution dependent variables.  The only variable
        # is the equivalent plastic strain

        self.num_sdv = 1
        self.sdv_names = ["EP_Equiv"]

    def Y(self, Y0, Y1, m, eqps):
        Y = Y0
        if eqps > 1e-12:
            Y += Y1 * eqps**m
        return Y

    def eval(self, time, dtime, temp, dtemp, F0, F, stran, d, stress, X, **kwargs):
        """Compute updated stress given strain increment"""

        #  material properties
        Y0 = self.params["Y0"]
        Y1 = self.params["Y1"]
        E = self.params["E"]
        Nu = self.params["Nu"]
        m = self.params["m"]
        if m < 1e-10:
            # if m = 0, assume linear hardening
            m = 1.0
        eqps = X[0]

        # Get the bulk, shear, and Lame constants
        K = E / 3.0 / (1.0 - 2.0 * Nu)
        G = E / 2.0 / (1.0 + Nu)

        K3 = 3.0 * K
        G2 = 2.0 * G
        # G3 = 3.0 * G
        Lam = (K3 - G2) / 3.0

        # elastic stiffness
        C = zeros((6, 6))
        C[ix_(range(3), range(3))] = Lam
        C[range(3), range(3)] += G2
        C[range(3, 6), range(3, 6)] = G

        # Trial stress
        de = d * dtime
        T = stress + double_dot(C, de)

        # check yield
        S = deviatoric_part(T)
        RTJ2 = magnitude(S) / ROOT2
        f = RTJ2 - self.Y(Y0, Y1, m, eqps) / ROOT3

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
            dfdy = -1.0 / ROOT3
            dydG = ROOT2 / ROOT3 * Y1
            hy = ROOT2 / ROOT3 * Y1
            if Y1 > 1e-8 and eqps > 1e-8:
                hy *= m * ((self.Y(Y0, Y1, m, eqps) - Y0) / Y1) ** ((m - 1.0) / m)
                dydG *= m * eqps ** (m - 1.0)

            dGamma = f * ROOT2 / (double_dot(N, A) - dfdy * dydG)
            Gamma += dGamma

            T = Ttrial - Gamma * A
            S = deviatoric_part(T)
            RTJ2 = magnitude(S) / ROOT2
            eqps += ROOT2 / ROOT3 * dGamma

            f = RTJ2 - self.Y(Y0, Y1, m, eqps) / ROOT3

            # Calculate the flow direction, projection direction
            M = S / ROOT2 / RTJ2
            N = S / ROOT2 / RTJ2
            A = 2 * G * M
            Q = 2 * G * N

            if abs(dGamma + 1.0) < TOLER + 1.0:
                break

        else:
            raise RuntimeError("Newton iterations failed to converge")

        # Elastic strain rate and equivalent plastic strain
        # dT = T - stress
        # dep = Gamma * M
        # dee = de - dep
        deqp = ROOT2 / ROOT3 * Gamma

        # Elastic stiffness
        H = -2.0 * dfdy * hy / ROOT2
        D = C - 1 / (double_dot(N, A) + H) * dyad(Q, A)

        # Equivalent plastic strain
        X[0] += deqp
        # print X[0]
        # print eqps
        # assert abs(X[0] - eqps) + 1 < 1.1e-5, 'Bad plastic strain integration'

        return T, X, D
