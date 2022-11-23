import numpy as np
from matmodlab2.core.logio import logger, StopFortran
from matmodlab2.core.tensor import VOIGT
from matmodlab2.core.material import Material

lib = None


class UMat(Material):
    """Constitutive model class for the umat model"""

    name = "umat"

    def __init__(self, parameters, depvar=1):
        """initialize the material state"""
        global lib
        try:
            import _umat

            lib = _umat
        except ImportError:
            raise RuntimeError("umat module must first be built")

        self.ordering = [0, 1, 2, 3, 5, 4]
        self.params = np.array(parameters)

        # depvar allowed to be an integer (number of SDVs) or a list (names of
        # SDVs)
        xkeys = lambda n: ["SDV{0}".format(i + 1) for i in range(depvar)]
        try:
            depvar, sdv_keys = len(depvar), depvar
        except TypeError:
            depvar = max(depvar, 1)
            sdv_keys = xkeys(depvar)

        # depvar must be at least 1 (cannot pass reference to empty list)
        assert depvar >= 1
        # statev = np.zeros(depvar)

        self.num_sdv = depvar
        self.sdv_names = sdv_keys

    def sdvini(self, statev):
        # initialize the model
        coords = np.zeros(3, order="F")
        noel = npt = layer = kspt = 1
        statev = np.zeros(self.num_sdv)
        lib.sdvini(
            statev,
            coords,
            noel,
            npt,
            layer,
            kspt,
            logger.info,
            logger.warning,
            StopFortran,
        )
        return statev

    def eval(self, time, dtime, temp, dtemp, F0, F, stran, d, stress, statev, **kwargs):

        # abaqus defaults
        cmname = "{0:8s}".format("umat")
        dfgrd0 = np.reshape(F0, (3, 3), order="F")
        dfgrd1 = np.reshape(F, (3, 3), order="F")
        dstran = d * dtime
        ddsdde = np.zeros((6, 6), order="F")
        ddsddt = np.zeros(6, order="F")
        drplde = np.zeros(6, order="F")
        predef = np.zeros(1, order="F")
        dpred = np.zeros(1, order="F")
        coords = np.zeros(3, order="F")
        drot = np.eye(3)
        ndi = nshr = 3
        spd = scd = rpl = drpldt = pnewdt = 0.0
        noel = npt = layer = kspt = kinc = 1
        rho = 1.0
        sse = np.dot(stress, stran * VOIGT) / rho
        celent = 1.0
        kstep = 1
        time = np.array([time, time])
        # abaqus ordering
        stress = stress[self.ordering]
        # abaqus passes engineering strain
        dstran = dstran[self.ordering]
        stran = stran[self.ordering]

        lib.umat(
            stress,
            statev,
            ddsdde,
            sse,
            spd,
            scd,
            rpl,
            ddsddt,
            drplde,
            drpldt,
            stran,
            dstran,
            time,
            dtime,
            temp,
            dtemp,
            predef,
            dpred,
            cmname,
            ndi,
            nshr,
            self.num_sdv,
            self.params,
            coords,
            drot,
            pnewdt,
            celent,
            dfgrd0,
            dfgrd1,
            noel,
            npt,
            layer,
            kspt,
            kstep,
            kinc,
            logger.info,
            logger.warning,
            StopFortran,
        )

        stress = stress[self.ordering]
        ddsdde = ddsdde[self.ordering, [[i] for i in self.ordering]]
        if abs(pnewdt) > 1e-12:
            logger.warning("Matmodlab does not support cutbacks")
        return stress, statev, ddsdde
