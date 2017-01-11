import warnings
import numpy as np
from copy import deepcopy as copy
from .environ import environ
from .tensor import VOIGT
import matmodlab.core.matfuncs as matfuncs

def update_deformation(farg, darg, dt, k):
    """Update the deformation gradient and strain

    Parameters
    ----------
    farg : ndarray
        The deformation gradient
    darg : ndarray
        The symmetric part of the velocity gradient
    dt : float
        The time increment
    k : int or real
        The Seth-Hill parameter

    Returns
    -------
    f : ndarray
        The updated deformation gradient
    e : ndarray
        The updated strain

    Notes
    -----
    From the value of the Seth-Hill parameter kappa, current strain E,
    deformation gradient F, and symmetric part of the velocit gradient d,
    update the strain and deformation gradient.
    Deformation gradient is given by

                 dFdt = L*F                                             (1)

    where F and L are the deformation and velocity gradients, respectively.
    The solution to (1) is

                 F = F0*exp(Lt)

    Solving incrementally,

                 Fc = Fp*exp(Lp*dt)

    where the c and p refer to the current and previous values, respectively.

    With the updated F, Fc, known, the updated stretch is found by

                 U = (trans(Fc)*Fc)**(1/2)

    Then, the updated strain is found by

                 E = 1/k * (U**k - I)

    where k is the Seth-Hill strain parameter.
    """
    f0 = farg.reshape((3, 3))
    d = matfuncs.array_to_mat(darg / VOIGT)[0]
    ff = np.dot(matfuncs.expm(d * dt), f0)
    u = matfuncs.sqrtm(np.dot(ff.T, ff))
    if k == 0:
        eps = matfuncs.logm(u)
    else:
        eps = 1.0 / k * (powm(u, k) - np.eye(3, 3))

    if matfuncs.determinant(ff) <= 0.0:
        raise Exception("negative jacobian encountered")

    f = np.reshape(ff, (9,))
    e = matfuncs.mat_to_array(eps, (6,)) * VOIGT

    return f, e

def defgrad_from_strain(E, kappa, flatten=1):
    """Compute the deformation gradient from the strain measure

    Parameters
    ----------
    E : ndarray (6,)
        Strain measure
    kappa : int or float
        Seth-Hill strain parameter
    flatten : bool, optional
        If True (default), return a flattened array

    Returns
    -------
    F : ndarray
        The deformation gradient

    """
    R = np.eye(3)
    I = np.eye(3)
    E, _ = matfuncs.array_to_mat(E / VOIGT)
    if kappa == 0:
        U = matfuncs.expm(E)
    else:
        U = powm(k * E + I, 1. / k)
    F = np.dot(R, U)
    if matfuncs.determinant(F) <= 0.0:
        raise Exception("negative jacobian encountered")
    if flatten:
        return F.flatten()
    return F

def strain_from_defgrad(farg, kappa):
    """Compute the strain measure from the deformation gradient

    Parameters
    ----------
    farg : ndarray (9,)
        Deformation gradient
    kappa : int or float
        Seth-Hill strain parameter
    flatten : bool, optional
        If True (default), return a flattened array

    Returns
    -------
    E : ndarray
        The strain measure

    Notes
    -----
    Update strain by

                 E = 1/k * (U**k - I)

    where k is the Seth-Hill strain parameter.
    """
    f = farg.reshape((3, 3))
    u = matfuncs.sqrtm(np.dot(f.T, f))
    if kappa == 0:
        eps = matfuncs.logm(u)
    else:
        eps = 1.0 / kappa * (matfuncs.powm(u, kappa) - np.eye(3, 3))

    if matfuncs.determinant(f) <= 0.0:
        raise Exception("negative jacobian encountered")

    e = matfuncs.mat_to_array(eps, (6,)) * VOIGT

    return e

def strain_from_stretch(u, k):
    """Convert the 3x3 stretch tensor to a strain tensor using the
    Seth-Hill parameter k and return a 6x1 array"""
    mat, orig_shape = matfuncs.array_to_mat(u)
    assert orig_shape == (6,)
    if abs(k) > 1e-12:
        mat = 1. / k * matfuncs.powm(mat, k) - np.eye(3)
    else:
        mat = matfuncs.logm(mat)
    return matfuncs.mat_to_array(mat, orig_shape)
