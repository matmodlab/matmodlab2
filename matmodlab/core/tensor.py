import warnings
import numpy as np
from copy import deepcopy as copy
from .environ import environ
import matmodlab.core.matfuncs as matfuncs

VOIGT = np.array([1, 1, 1, 2, 2, 2], dtype=np.float64)
I6 = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
epsilon = np.finfo(float).eps

def stretch_to_strain(u, k):
    """Convert the 3x3 stretch tensor to a strain tensor using the
    Seth-Hill parameter k and return a 6x1 array"""
    mat, orig_shape = matfuncs.array_to_mat(u)
    assert orig_shape == (6,)
    if abs(k) > 1e-12:
        mat = 1. / k * matfuncs.powm(mat, k) - np.eye(3)
    else:
        mat = matfuncs.logm(mat)
    return matfuncs.mat_to_array(mat, orig_shape)

def symarray(A):
    """Convert a 3x3 matrix to a 6x1 array representing a symmetric matix."""
    mat = (A + A.T) / 2.0
    return mat[([0,1,2,0,1,0],[0,1,2,1,2,2])]

def rate_of_matrix_function(A, Adot, f, fprime):
    """Find the rate of the tensor A

    Parameters
    ----------
    A : ndarray (3,3)
        A diagonalizable tensor
    Adot : ndarray (3,3)
        Rate of A
    f : callable
    fprime : callable
        Derivative of f

    Returns
    -------
    Ydot : ndarray (3,3)

    Notes
    -----
    For a diagonalizable tensor A (the strain) which has a quasi-arbitrary
    spectral expansion

    .. math::
        A = \sum_{i=1}^3 \lambda_i P_{i}

    and if a second tensor Y is a principal function of A, defined by

    .. math::
        Y = \sum_{i=1}^3 f(\lambda_i) P_i,

    compute the time rate \dot{Y}. Algorithm taken from Brannon's
    Tensor book, from the highlighted box near Equation (28.404) on
    page 550.

    """

    # Compute the eigenvalues and eigenprojections.
    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_projs = [np.outer(eig_vecs[:, i], eig_vecs[:, i]) for i in [0, 1, 2]]

    # Assemble the rate of Y.
    Ydot = np.zeros((3, 3))
    for eigi, proji in zip(eig_vals, eig_projs):
        for eigj, projj in zip(eig_vals, eig_projs):
            if eigi == eigj:
                gamma = fprime(eigi)
            else:
                gamma = (f(eigi) - f(eigj)) / (eigi - eigj)
            Ydot += gamma * np.dot(proji, np.dot(Adot, projj))

    return Ydot

def rate_of_strain_to_rate_of_deformation(dedt, e, k, disp=0):
    """ Compute symmetric part of velocity gradient given depsdt

    Parameters
    ----------
    dedt : ndarray
        Strain rate
    e : ndarray
        Strain
    k : int or float
        Seth-Hill parameter
    disp : bool, optional
        If True, return both d and dU

    Returns
    -------
    d : ndarray
        Symmetric part of the velocity gradient
    dU : ndarray
        Rate of stretch (if disp is True)

    Notes
    -----
    Velocity gradient L is given by
                L = dFdt * Finv
                  = dRdt*I*Rinv + R*dUdt*Uinv*Rinv
    where F, I, R, U are the deformation gradient, identity, rotation, and
    right stretch tensor, respectively. d*dt and *inv are the rate and
    inverse or *, respectively,

    The stretch U is given by
                 if k != 0:
                     U = (k*E + I)**(1/k)
                 else:
                     U = exp(E)
    and its rate (dUdt) is calculated using `rate_of_matrix_function`.
       Then
                 L = dot(dUdt, inv(U))
                 d = sym(L)
                 w = skew(L)

    """

    eps = matfuncs.array_to_mat(e / VOIGT)[0]
    depsdt = matfuncs.array_to_mat(dedt / VOIGT)[0]

    # Calculate the right stretch (U) and its rate
    if abs(k) <= 1e-12:
        u = matfuncs.expm(eps)
        dudt = rate_of_matrix_function(eps, depsdt, np.exp, np.exp)
    else:
        u = matfuncs.powm(k*eps+np.eye(3), 1./k)
        f = lambda x: (k * x + 1.0) ** (1.0 / k)
        fprime = lambda x: (k * x + 1.0) ** (1.0 / k - 1.0)
        dudt = rate_of_matrix_function(eps, depsdt, f, fprime)

    # Calculate the velocity gradient (L) and its symmetric part
    l = np.dot(dudt, matfuncs.inverse(u))
    d = (l + l.T) / 2.0

    d = symarray(d) * VOIGT
    if not disp:
        return d
    return d, dudt

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

def isotropic_part(A):
    """Return isotropic part of A"""
    if A.shape == (6,):
        return np.sum(A[:3]) / 3. * np.array([1.,1.,1.,0.,0.,0.])
    elif A.shape == (3,3):
        return np.trace(A) / 3. * np.eye(3)
    else:
        raise ValueError('Unknown shape')

def deviatoric_part(A):
    """Return deviatoric part of A"""
    return A - isotropic_part(A)

def symmetric_dyad(A, B):
    """Compute the symmetric dyad AB_ij = A_i B_j"""
    A = np.asarray(A)
    B = np.asarray(B)
    assert A.shape == (3,)
    assert A.shape == B.shape
    return np.array([A[0] * B[0], A[1] * B[1], A[2] * B[2],
                     A[0] * B[1], A[1] * B[2], A[0] * B[2]])

def invariants(A, n=None, mechanics=False):
    if mechanics:
        i1 = matfuncs.trace(A)
        rootj2 = magnitude(deviatoric_part(A)) / np.sqrt(2.)
        return i1, rootj2

    A = matfuncs.array_to_mat(A)[0]
    asq = np.dot(A, A)
    deta = matfuncs.determinant(A)
    tra = np.trace(A)

    b = np.zeros(5)
    b[0] = tra
    b[1] = .5 * (tra ** 2 - np.trace(asq))
    b[2] = deta
    if n in None:
        return b[:3]

    b[3] = np.dot(np.dot(n, A), n)
    b[4] = np.dot(np.dot(n, asq), n)

    return b

def magnitude(A):
    """Return magnitude of A"""
    mat = matfuncs.array_to_mat(A)[0]
    return np.sqrt(np.sum(np.dot(mat, mat)))

def double_dot(A, B):
    """Return A:B"""
    A_mat, A_shape = matfuncs.array_to_mat(A)
    B_mat, B_shape = matfuncs.array_to_mat(B)
    assert A.shape == (3,3)
    assert B.shape == A.shape
    return np.sum(A * B)

def polar_decomp(F):
    F = F.reshape(3,3)
    I = np.eye(3)
    R = F.copy()
    for j in range(20):
        R = .5 * np.dot(R, 3. * I - np.dot(R.T, R))
        if (np.amax(np.abs(np.dot(R.T, R) - I)) < 1.e-6):
            U = np.dot(R.T, F)
            return R, U
    raise RuntimeError('Fast polar decompositon failed to converge')
