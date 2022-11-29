import numpy as np

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import VOIGT

ROOT2 = np.sqrt(2.0)
ROOT3 = np.sqrt(3.0)
ROOT23 = np.sqrt(2.0 / 3.0)
TOLER = 1e-8


class VonMisesMaterial(Material):
    name = "vonmises"

    def __init__(self, **kwargs):
        """Set up the von Mises material"""
        # Check inputs
        K = kwargs.get("K", 0.0)
        G = kwargs.get("G", 0.0)
        H = kwargs.get("H", 0.0)
        Y0 = kwargs.get("Y0", 0.0)
        BETA = kwargs.get("BETA", 0.0)

        errors = 0
        if K <= 0.0:
            errors += 1
            logger.error("Bulk modulus K must be positive")
        if G <= 0.0:
            errors += 1
            logger.error("Shear modulus G must be positive")
        nu = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)
        if nu > 0.5:
            errors += 1
            logger.error("Poisson's ratio > .5")
        if nu < -1.0:
            errors += 1
            logger.error("Poisson's ratio < -1.")
        if nu < 0.0:
            logger.warning("negative Poisson's ratio")
        if abs(Y0) <= 1.0e-12:
            Y0 = 1.0e99
        if errors:
            raise ValueError("stopping due to previous errors")

        self.params = {"K": K, "G": G, "Y0": Y0, "H": H, "BETA": BETA}

        # Register State Variables
        self.sdv_names = [
            "EQPS",
            "Y",
            "BS_XX",
            "BS_YY",
            "BS_ZZ",
            "BS_XY",
            "BS_XZ",
            "BS_YZ",
            "SIGE",
        ]
        self.num_sdv = len(self.sdv_names)

    def sdvini(self, statev):
        Y0 = self.params["Y0"]
        return np.array([0.0, Y0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def eval(self, time, dtime, temp, dtemp, F0, F, stran, d, stress, statev, **kwargs):
        """Compute updated stress given strain increment"""

        idx = lambda x: self.sdv_names.index(x.upper())
        bs = np.array(
            [
                statev[idx("BS_XX")],
                statev[idx("BS_YY")],
                statev[idx("BS_ZZ")],
                statev[idx("BS_XY")],
                statev[idx("BS_YZ")],
                statev[idx("BS_XZ")],
            ]
        )
        yn = statev[idx("Y")]

        de = d / VOIGT * dtime

        iso = de[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        dev = de - iso

        stress_trial = (
            stress + 3.0 * self.params["K"] * iso + 2.0 * self.params["G"] * dev
        )

        xi_trial = stress_trial - bs
        xi_trial_eqv = self.eqv(xi_trial)

        if xi_trial_eqv <= yn:
            statev[idx("SIGE")] = xi_trial_eqv
            return stress_trial, statev, None
        else:
            N = xi_trial - xi_trial[:3].sum() / 3.0 * np.array(
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            )
            N = N / (ROOT23 * xi_trial_eqv)
            deqps = (xi_trial_eqv - yn) / (3.0 * self.params["G"] + self.params["H"])
            dps = 1.0 / ROOT23 * deqps * N

            stress_final = stress_trial - 2.0 * self.params["G"] / ROOT23 * deqps * N

            bs = bs + 2.0 / 3.0 * self.params["H"] * self.params["BETA"] * dps

            statev[idx("EQPS")] += deqps
            statev[idx("Y")] += self.params["H"] * (1.0 - self.params["BETA"]) * deqps
            statev[idx("BS_XX")] = bs[0]
            statev[idx("BS_YY")] = bs[1]
            statev[idx("BS_ZZ")] = bs[2]
            statev[idx("BS_XY")] = bs[3]
            statev[idx("BS_YZ")] = bs[4]
            statev[idx("BS_XZ")] = bs[5]
            statev[idx("SIGE")] = self.eqv(stress_final - bs)
            return stress_final, statev, None

    def eqv(self, sig):
        # Returns sqrt(3 * rootj2) = sig_eqv = q
        s = sig - sig[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return 1.0 / ROOT23 * np.sqrt(np.dot(s[:3], s[:3]) + 2 * np.dot(s[3:], s[3:]))
