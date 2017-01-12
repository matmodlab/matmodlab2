# -*- coding: utf-8 -*-
"""
This file contains tests for matfuncs.py
"""
import pytest
import numpy as np
from testing_utils import *

try:
    import matmodlab2.core._matfuncs_sq3
    la = matmodlab2.core._matfuncs_sq3.linalg
except ImportError:
    la = None
pytestmark = pytest.mark.skipif(la is None, reason='Fortran linalg not imported')

def test_inv():
    """Inverse of A"""
    A = random_matrix()
    Ai = la.inv(A)
    assert np.allclose(np.dot(Ai, A), np.eye(3))

def test_expm():
    """Compute the matrix exponential of a 3x3 matrix"""
    x, y, z = 1., 2., 3.
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    exp_A = np.array([[np.exp(x), 0, 0], [0, np.exp(y), 0], [0, 0, np.exp(z)]])
    assert np.allclose(la.expm(A), exp_A)

def test_logm():
    """Compute the matrix logarithm of a 3x3 matrix"""
    x, y, z = 1., 2., 3.
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    log_A = np.array([[np.log(x), 0, 0], [0, np.log(y), 0], [0, 0, np.log(z)]])
    assert np.allclose(log_A, la.logm(A))

def test_expm_logm_consistency():
    A = random_symmetric_positive_definite_matrix()
    B = la.expm(A)
    assert np.allclose(la.logm(B), A)

def test_logm_expm_consistency():
    A = random_symmetric_positive_definite_matrix()
    A = A.reshape(3,3)
    B = la.logm(A)
    assert np.allclose(la.expm(B), A)

def test_powm():
    """Compute the matrix power of a 3x3 matrix"""
    m = 1.2
    x, y, z = 1., 2., 3.
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    pow_A = np.array([[x**m, 0, 0], [0, y**m, 0], [0, 0, z**m]])
    assert np.allclose(la.powm(A, m), pow_A)
    A = random_symmetric_positive_definite_matrix()
    B = la.powm(A, .5)
    assert np.allclose(np.dot(B,B), A)
    A = random_symmetric_positive_definite_matrix()
    B = la.powm(A, 2)
    assert np.allclose(np.dot(A,A), B)

def test_sqrtm():
    """Compute the square root of a 3x3 matrix"""
    x, y, z = np.random.rand(3)
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    sqrt_A = np.array([[np.sqrt(x), 0, 0],
                       [0, np.sqrt(y), 0],
                       [0, 0, np.sqrt(z)]])
    assert np.allclose(la.sqrtm(A), sqrt_A)
    A = random_symmetric_positive_definite_matrix()
    B = la.sqrtm(A)
    assert np.allclose(np.dot(B,B), A)

if __name__ == '__main__':
    test_import()
