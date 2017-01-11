# -*- coding: utf-8 -*-
"""
This file contains tests for matfuncs.py
"""
import pytest
import numpy as np
from testing_utils import isclose

try:
    import matmodlab
except ImportError:
    matmodlab = None

import matmodlab.core.matfuncs as mf

# NOTE: In many of the tests to follow, only a trivial tensor is sent to test
# the matrix function. This is by design. Since all of the matrix functions are
# just wrappers to scipy functions, these tests just test the wrapper (the
# ability to call the function with a 3x3 "matrix" or 6x1 array)

def test_array_to_mat():
    """Test array to matrix conversion"""
    a = np.array([1,4,6,2,5,3])
    A = np.array([[1,2,3],[2,4,5],[3,5,6]])
    ax, shape = mf.array_to_mat(a)
    assert np.allclose(ax, A)
    assert shape == (6,)
    Ax, shape = mf.array_to_mat(A)
    assert np.allclose(Ax, A)
    assert shape == (3,3)
    Ax = mf.mat_to_array(A, (6,))
    assert np.allclose(Ax, a)
    Ax = mf.mat_to_array(A, (3,3))
    assert np.allclose(Ax, A)

def test_determinant():
    """Determinant of A"""
    x, y, z = 1., 2., 3.
    a = np.array([x, y, z, 0, 0, 0])
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    print(mf.determinant(a))
    assert isclose(mf.determinant(a), x*y*z)
    assert isclose(mf.determinant(A), x*y*z)

def test_trace():
    a_ident = np.array([1., 1., 1., 0., 0., 0.])
    A_ident = np.eye(3)
    assert isclose(mf.trace(a_ident), 3.)
    assert isclose(mf.trace(A_ident), 3.)

    a_dev = np.array([1., -.5, -.5, 0., 0., 0.])
    A_dev = np.array([[1., 0., 0.], [0., -.5, 0.], [0., 0., -.5]])
    assert isclose(mf.trace(a_dev), 0.)
    assert isclose(mf.trace(A_dev), 0.)

def test_inverse():
    """Inverse of A"""
    x, y, z = 1., 2., 3.
    fun = lambda _: 1./ _
    mf_fun = mf.inverse
    a = np.array([x, y, z, 0, 0, 0])
    a_fun = np.array([fun(x), fun(y), fun(z), 0, 0, 0])
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    A_fun = np.array([[fun(x), 0, 0], [0, fun(y), 0], [0, 0, fun(z)]])
    assert np.allclose(mf_fun(a), a_fun)
    assert np.allclose(mf_fun(A), A_fun)

def test_expm():
    """Compute the matrix exponential of a 3x3 matrix"""
    x, y, z = 1., 2., 3.
    fun = np.exp
    mf_fun = mf.expm
    a = np.array([x, y, z, 0, 0, 0])
    a_fun = np.array([fun(x), fun(y), fun(z), 0, 0, 0])
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    A_fun = np.array([[fun(x), 0, 0], [0, fun(y), 0], [0, 0, fun(z)]])
    assert np.allclose(mf_fun(a), a_fun)
    assert np.allclose(mf_fun(A), A_fun)

def test_expm_logm_consistency():
    A = random_symmetric_positive_definite_matrix()
    B = mf.expm(A)
    assert np.allclose(mf.logm(B), A)

def test_logm_expm_consistency():
    A = random_symmetric_positive_definite_matrix()
    A = A.reshape(3,3)
    B = mf.logm(A)
    assert np.allclose(mf.expm(B), A)

def test_logm():
    """Compute the matrix logarithm of a 3x3 matrix"""
    x, y, z = 1., 2., 3.
    fun = np.log
    mf_fun = mf.logm
    a = np.array([x, y, z, 0, 0, 0])
    a_fun = np.array([fun(x), fun(y), fun(z), 0, 0, 0])
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    A_fun = np.array([[fun(x), 0, 0], [0, fun(y), 0], [0, 0, fun(z)]])
    assert np.allclose(mf_fun(a), a_fun)
    assert np.allclose(mf_fun(A), A_fun)

def test_powm():
    """Compute the matrix power of a 3x3 matrix"""
    x, y, z = 1., 2., 3.
    m = 1.2
    fun = lambda _: _ ** m
    mf_fun = lambda _, m: mf.powm(_, m)
    a = np.array([x, y, z, 0, 0, 0])
    a_fun = np.array([fun(x), fun(y), fun(z), 0, 0, 0])
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    A_fun = np.array([[fun(x), 0, 0], [0, fun(y), 0], [0, 0, fun(z)]])
    assert np.allclose(mf.powm(a, m), a_fun)
    assert np.allclose(mf.powm(A, m), A_fun)
    A = np.random.rand(9).reshape(3,3)
    B = mf.powm(A, .5)
    assert np.allclose(np.dot(B,B), A)

def test_sqrtm():
    """Compute the square root of a 3x3 matrix"""
    x, y, z = 1., 2., 3.
    fun = np.sqrt
    mf_fun = mf.sqrtm
    a = np.array([x, y, z, 0, 0, 0])
    a_fun = np.array([fun(x), fun(y), fun(z), 0, 0, 0])
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    A_fun = np.array([[fun(x), 0, 0], [0, fun(y), 0], [0, 0, fun(z)]])
    assert np.allclose(mf_fun(a), a_fun)
    assert np.allclose(mf_fun(A), A_fun)
    A = random_symmetric_positive_definite_matrix()
    B = mf_fun(A)
    assert np.allclose(np.dot(B,B), A)

def random_symmetric_positive_definite_matrix():
    theta = np.random.uniform(0, 2*np.pi, 1)[0]
    a = np.random.rand(3)
    a = a / np.sqrt(np.dot(a,a))
    aa = np.outer(a,a)
    I = np.eye(3)
    A = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = I + 2*np.sin(theta/2.)**2*(aa-I)+np.sin(theta)*A
    L = np.zeros((3,3))
    L[([0,1,2],[0,1,2])] = np.random.rand(3)
    X = np.dot(np.dot(R, L), R.T)
    return (X + X.T) / 2.

if __name__ == '__main__':
    test_import()
