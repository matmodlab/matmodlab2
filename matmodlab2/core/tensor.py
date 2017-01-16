import warnings
import numpy as np
from copy import deepcopy as copy
from .environ import environ
import matmodlab2.core.linalg as la

VOIGT = np.array([1., 1., 1., 2., 2., 2.])
I6 = np.array([1., 1., 1., 0., 0., 0.])
epsilon = np.finfo(float).eps
SYMMETRIC_COMPONENTS = ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ']
TENSOR_COMPONENTS = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']

def isotropic_part(A):
    """Return isotropic part of A"""
    if is_symmetric_rep(A):
        return np.sum(A[:3]) / 3. * np.array([1.,1.,1.,0.,0.,0.])
    elif A.shape == (3,3):
        return np.trace(A) / 3. * np.eye(3)
    else:
        raise ValueError('Unknown shape')

def deviatoric_part(A):
    """Return deviatoric part of A"""
    return A - isotropic_part(A)

def is_symmetric_rep(A):
    return A.shape == (6,)

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
        i1 = trace(A)
        rootj2 = magnitude(deviatoric_part(A)) / np.sqrt(2.)
        return i1, rootj2

    A = matrix_rep(A, 0)
    asq = np.dot(A, A)
    deta = la.det(A)
    tra = la.trace(A)

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
    mat = matrix_rep(A, 0)
    return np.sqrt(np.sum(np.dot(mat, mat)))

def double_dot(A, B):
    """Return A:B"""
    A_mat, A_shape = matrix_rep(A)
    B_mat, B_shape = matrix_rep(B)
    assert A.shape == (3,3)
    assert B.shape == A.shape
    return np.sum(A * B)

def polar_decomp(F):
    return la.polar_decomp(F)

def det(A):
    """Determinant of tensor A"""
    mat, orig_shape = matrix_rep(A)
    return la.det(mat)

def trace(A):
    """Return trace of A"""
    mat, orig_shape = matrix_rep(A)
    ix = ([0,1,2],[0,1,2])
    return np.sum(mat[ix])

def inv(A):
    """Inverse of A"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.inv(mat)
    return array_rep(mat2, orig_shape)

def expm(A):
    """Compute the matrix exponential of a 3x3 matrix"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.expm(mat)
    return array_rep(mat2, orig_shape)

def logm(A):
    """Compute the matrix logarithm of a 3x3 matrix"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.logm(mat)
    return array_rep(mat2, orig_shape)

def powm(A, t):
    """Compute the matrix power of a 3x3 matrix"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.powm(mat)
    return array_rep(mat2, orig_shape)

def sqrtm(A):
    """Compute the square root of a 3x3 matrix"""
    mat, orig_shape = matrix_rep(A)
    mat2 = la.sqrtm(mat)
    return array_rep(mat2, orig_shape)

def matrix_rep(A, disp=1):
    """Convert array to matrix"""
    orig_shape = A.shape
    if orig_shape == (6,):
        ix1 = ([0,1,2,0,1,0,1,2,2],[0,1,2,1,2,2,0,0,1])
        ix2 = [0,1,2,3,4,5,3,5,4]
        mat = np.zeros((3,3))
        mat[ix1] = A[ix2]
    elif orig_shape == (9,):
        mat = np.reshape(A, (3,3))
    elif orig_shape == (3,3):
        mat = np.array(A)
    else:
        raise ValueError('Unknown shape')
    if not disp:
        return mat
    return mat, orig_shape

def array_rep(mat, shape):
    """Reverse of matrix_rep"""
    if shape == (6,):
        mat = .5 * (mat + mat.T)
        return mat[([0,1,2,0,1,0],[0,1,2,1,2,2])]
    if shape == (9,):
        return np.flatten(mat)
    if shape == (3,3):
        return np.array(mat)
    raise ValueError('Unknown shape')

def isdiag(A):
    """Determines if a matrix is diagonal."""
    return np.all(np.abs(A[([0,0,1,1,2,2],[1,2,0,2,0,1])])<=epsilon)
