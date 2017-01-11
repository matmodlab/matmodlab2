import warnings
import numpy as np
import scipy.linalg
from .logio import logger
try:
    import matmodlab.core._matfuncs_sq3
    la = matmodlab.core._matfuncs_sq3.linalg
    logger.info('Using fortran matfuncs')
except ImportError:
    la = scipy.linalg
    logger.info('Using scipy.linalg matfuncs')

epsilon = np.finfo(float).eps

def determinant(A):
    """Determinant of A"""
    mat, orig_shape = array_to_mat(A)
    if isdiag(mat):
        ix = ([0,1,2],[0,1,2])
        return np.prod(mat[ix])
    else:
        return la.det(mat)

def trace(A):
    """Return trace of A"""
    mat, orig_shape = array_to_mat(A)
    ix = ([0,1,2],[0,1,2])
    return np.sum(mat[ix])

def inverse(A):
    """Inverse of A"""
    mat, orig_shape = array_to_mat(A)
    if isdiag(mat):
        ix = ([0,1,2],[0,1,2])
        if any(abs(mat[ix])<=epsilon):
            raise np.linalg.LinAlgError('singular matrix')
        mat2 = np.zeros((3,3))
        mat2[ix] = 1. / mat[ix]
    else:
        mat2 = la.inv(mat)
    return mat_to_array(mat2, orig_shape)

def expm(A):
    """Compute the matrix exponential of a 3x3 matrix"""
    mat, orig_shape = array_to_mat(A)
    if isdiag(mat):
        ix = ([0,1,2],[0,1,2])
        mat2 = np.zeros((3,3))
        mat2[ix] = np.exp(mat[ix])
    else:
        mat2 = la.expm(mat)
    return mat_to_array(mat2, orig_shape)

def logm(A):
    """Compute the matrix logarithm of a 3x3 matrix"""
    mat, orig_shape = array_to_mat(A)
    if isdiag(mat):
        ix = ([0,1,2],[0,1,2])
        mat2 = np.zeros((3,3))
        mat2[ix] = np.log(mat[ix])
    else:
        mat2 = la.logm(mat)
    return mat_to_array(mat2, orig_shape)

def powm(A, t):
    """Compute the matrix power of a 3x3 matrix"""
    mat, orig_shape = array_to_mat(A)
    if isdiag(mat):
        ix = ([0,1,2],[0,1,2])
        mat2 = np.zeros((3,3))
        mat2[ix] = mat[ix] ** t
    else:
        mat2 = scipy.linalg.fractional_matrix_power(mat, t)
    return mat_to_array(mat2, orig_shape)

def sqrtm(A):
    """Compute the square root of a 3x3 matrix"""
    mat, orig_shape = array_to_mat(A)
    if isdiag(mat):
        ix = ([0,1,2],[0,1,2])
        mat2 = np.zeros((3,3))
        mat2[ix] = np.sqrt(mat[ix])
    else:
        mat2 = la.sqrtm(mat)
    return mat_to_array(mat2, orig_shape)

def array_to_mat(A):
    """Convert array to matrix"""
    if A.shape == (6,):
        ix1 = ([0,1,2,0,1,0,1,2,2],[0,1,2,1,2,2,0,0,1])
        ix2 = [0,1,2,3,4,5,3,5,4]
        mat = np.zeros((3,3))
        mat[ix1] = A[ix2]
        orig_shape = (6,)
    elif A.shape == (9,):
        mat = np.reshape(A, (3,3))
        orig_shape = (9,)
    elif A.shape == (3,3):
        mat = np.array(A)
        orig_shape = (3,3)
    else:
        raise ValueError('Unknown shape')
    return mat, orig_shape

def mat_to_array(mat, shape):
    """Reverse of array_to_mat"""
    if shape == (6,):
        mat = .5 * (mat + mat.T)
        return mat[([0,1,2,0,1,0],[0,1,2,1,2,2])]
    if shape == (9,):
        return mat.flatten()
    if shape == (3,3):
        return mat
    raise ValueError('Unknown shape')

def isdiag(A):
    """Determines if a matrix is diagonal."""
    return np.all(np.abs(A[([0,0,1,1,2,2],[1,2,0,2,0,1])])<=epsilon)
