import warnings
import numpy as np
import scipy.linalg
from .logio import logger
from .environ import environ
try:
    import matmodlab2.core._matfuncs_sq3
    la = matmodlab2.core._matfuncs_sq3.linalg
    fortran_linalg = True
    logger.debug('Using fortran linalg')
except ImportError:
    fortran_linalg = False
    la = scipy.linalg
    logger.debug('Using scipy.linalg')

def set_linalg_library(lib):
    global la

    if lib == 'default':
        if fortran_linalg:
            la = matmodlab2.core._matfuncs_sq3.linalg
        else:
            la = scipy.linalg
        return

    if lib == 'scipy':
        la = scipy.linalg
        return

    if lib == 'fortran':
        if fortran_linalg:
            la = matmodlab2.core._matfuncs_sq3.linalg
        else:
            la = scipy.linalg
            logger.warning('Fortran linalg not imported')

        return

epsilon = np.finfo(float).eps

def apply_fun_to_diag(mat, fun):
    ix = ([0,1,2],[0,1,2])
    mat2 = np.zeros((3,3))
    mat2[ix] = fun(mat[ix])
    return mat2

def det(mat):
    """Determinant of A"""
    assert mat.shape == (3,3)
    if is_diagonal(mat):
        ix = ([0,1,2],[0,1,2])
        return np.prod(mat[ix])
    else:
        return la.det(mat)

def trace(mat):
    """Return trace of A"""
    assert mat.shape == (3,3)
    ix = ([0,1,2],[0,1,2])
    return np.sum(mat[ix])

def inv(mat):
    """Inverse of A"""
    assert mat.shape == (3,3)
    if is_diagonal(mat):
        ix = ([0,1,2],[0,1,2])
        if any(abs(mat[ix])<=epsilon):
            raise np.linalg.LinAlgError('singular matrix')
        return apply_fun_to_diag(mat, lambda x: 1. / x)
    else:
        mat2 = la.inv(mat)
    return mat2

def expm(mat):
    """Compute the matrix exponential of a 3x3 matrix"""
    assert mat.shape == (3,3)
    if is_diagonal(mat):
        return apply_fun_to_diag(mat, np.exp)
    else:
        mat2 = la.expm(mat)
    return mat2

def logm(mat):
    """Compute the matrix logarithm of a 3x3 matrix"""
    assert mat.shape == (3,3)
    if is_diagonal(mat):
        return apply_fun_to_diag(mat, np.log)
    else:
        mat2 = la.logm(mat)
    return mat2

def powm(mat, t):
    """Compute the matrix power of a 3x3 matrix"""
    assert mat.shape == (3,3)
    if is_diagonal(mat):
        return apply_fun_to_diag(mat, lambda x: x ** t)
    else:
        mat2 = scipy.linalg.fractional_matrix_power(mat, t)
    return mat2

def sqrtm(mat):
    """Compute the square root of a 3x3 matrix"""
    assert mat.shape == (3,3)
    if is_diagonal(mat):
        return apply_fun_to_diag(mat, np.sqrt)
    else:
        mat2 = la.sqrtm(mat)
    return mat2

def is_diagonal(A):
    """Determines if a matrix is diagonal."""
    return np.all(np.abs(A[([0,0,1,1,2,2],[1,2,0,2,0,1])])<=epsilon)

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

def polar_decomp(F):
    F = F.reshape(3,3)
    try:
        R, U, ierr = la.polar_decomp(F)
        if not ierr:
            return R, U
    except AttributeError:
        I = np.eye(3)
        R = F.copy()
        for j in range(20):
            R = .5 * np.dot(R, 3. * I - np.dot(R.T, R))
            if (np.amax(np.abs(np.dot(R.T, R) - I)) < 1.e-6):
                U = np.dot(R.T, F)
                return R, U
    try:
        R, V = scipy.linalg.qr(F)
        U = np.dot(R.T, np.dot(V, R))
        return R, U
    except:
        raise RuntimeError('Fast polar decompositon failed')

def solve(A, b):
    return scipy.linalg.solve(A, b)

def lstsq(A, b):
    return np.linalg.lstsq(A, b)
