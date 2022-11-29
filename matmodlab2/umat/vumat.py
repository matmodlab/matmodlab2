import numpy as np
from matmodlab2.core.logio import logger, StopFortran
from matmodlab2.core.tensor import polar_decomp, inv, array_rep, det
from matmodlab2.core.material import Material

lib = None


class VUMat(Material):
    """Constitutive model class for the umat model"""

    name = "vumat"

    def __init__(self, parameters, depvar=1, density=1.):
        """initialize the material state"""
        global lib
        try:
            import _vumat

            lib = _vumat
        except ImportError:
            raise RuntimeError("vumat module must first be built")

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

        self.num_sdv = depvar
        self.sdv_names = sdv_keys
        self.initial_density = density

    def eval(self, time, dtime, temp, dtemp, F0, F, stran, d, stress, statev, **kwargs):

        # abaqus defaults
        cmname = "{0:8s}".format("vumat")
        ndir = nshr = 3
        nblock = nfieldv = lanneal = 1
        char_length = 1.0
        nstatev = self.num_sdv
        nprops = len(self.params)
        jac = det(F)
        rho = self.initial_density / jac

        coordMp = np.zeros((nblock, 3), order="F")
        char_length = np.ones(nblock, order="F")
        props = np.zeros(nprops, order="F")
        density = np.ones(nblock, order="F") * rho
        strain_inc = np.ones((nblock, ndir + nshr), order="F")
        rel_spin_inc = np.ones((nblock, nshr), order="F")
        temp_old = np.ones(nblock, order="F") * temp
        stretch_old = np.zeros((nblock, ndir + nshr), order="F")
        defgrad_old = np.zeros((nblock, ndir + nshr + nshr), order="F")
        field_old = np.zeros((nblock, nfieldv), order="F")
        stress_old = np.zeros((nblock, ndir + nshr), order="F")
        state_old = np.zeros((nblock, nstatev), order="F")
        ener_intern_old = np.ones(nblock, order="F")
        ener_inelas_old = np.ones(nblock, order="F")
        temp_new = np.ones(nblock, order="F") * (temp + dtemp)
        stretch_new = np.zeros((nblock, ndir + nshr), order="F")
        defgrad_new = np.zeros((nblock, ndir + nshr + nshr), order="F")
        field_new = np.asarray(field_old)
        stress_new = np.zeros((nblock, ndir + nshr), order="F")
        state_new = np.zeros((nblock, nstatev), order="F")
        ener_intern_new = np.zeros(nblock, order="F")
        ener_inelas_new = np.zeros(nblock, order="F")

        Rn, Un = polar_decomp(F0)
        Rp, Up = polar_decomp(F)
        dF = np.reshape(F - F0, (3, 3))
        dR = np.reshape(Rp - Rn, (3, 3))
        O = np.dot(dR, Rp.T)
        L = np.dot(dF, np.reshape(inv(F), (3,3)))
        W = 0.5 * (L - L.T)
        dW = W - O

        props[:] = self.params
        strain_inc[0, :] = d * dtime
        rel_spin_inc[0, :] = [dW[2, 1], dW[0, 2], dW[1, 0]]
        stretch_old[0, :] = array_rep(Un, (6,))
        defgrad_old[0, :] = np.reshape(F0, (3, 3), order="F").flatten()
        stress_old[0, :] = stress
        state_old[:] = statev
        stretch_new[0, :] = array_rep(Up, (6,))
        defgrad_new[0, :] = np.reshape(F, (3, 3), order="F").flatten()
        step_time = total_time = time

        stress_new, state_new, ener_intern_new, ener_inelas_new = lib.vumat(
            ndir,
            lanneal,
            step_time,
            total_time,
            dtime,
            cmname,
            coordMp,
            char_length,
            props,
            density,
            strain_inc,
            rel_spin_inc,
            temp_old,
            stretch_old,
            defgrad_old,
            field_old,
            stress_old,
            state_old,
            ener_intern_old,
            ener_inelas_old,
            temp_new,
            stretch_new,
            defgrad_new,
            field_new,
            logger.info,
            logger.warning,
            StopFortran,
        )

        stress = stress_new[0, :]
        statev = state_new[:]
        return stress, statev, None
