# -*- coding: utf-8 -*-
"""
This file contains tests for tensor.py
"""

import sys
import pathlib
import pytest
import numpy as np
from testing_utils import isclose

# Ensure that 'matmodlab2' is imported from parent directory.
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent))
import matmodlab2
import matmodlab2.core.tensor as tens

def test_isotropic_part():
    a_ident = np.array([1., 1., 1., 0., 0., 0.])
    A_ident = np.eye(3)
    iso_a_ident = tens.isotropic_part(a_ident)
    iso_A_ident = tens.isotropic_part(A_ident)
    assert np.allclose(iso_a_ident, a_ident)
    assert np.allclose(iso_A_ident, A_ident)
    dev_a_ident = tens.deviatoric_part(a_ident)
    dev_A_ident = tens.deviatoric_part(A_ident)
    assert np.allclose(dev_a_ident, 0)
    assert np.allclose(dev_A_ident, 0)

def test_deviatoric_part():
    a_dev = np.array([1., -.5, -.5, 0., 0., 0.])
    A_dev = np.array([[1., 0., 0.], [0., -.5, 0.], [0., 0., -.5]])
    iso_a_dev = tens.isotropic_part(a_dev)
    iso_A_dev = tens.isotropic_part(A_dev)
    assert np.allclose(iso_a_dev, 0)
    assert np.allclose(iso_A_dev, 0)
    dev_a_dev = tens.deviatoric_part(a_dev)
    dev_A_dev = tens.deviatoric_part(A_dev)
    assert np.allclose(dev_a_dev, a_dev)
    assert np.allclose(dev_A_dev, A_dev)

def test_invariants_mechanics_dev():
    a_dev = np.array([1., 0., -1., 2., 4., 3.])
    A_dev = np.array([[1.0, 2.0, 3.0],
                      [2.0, 0.0, 4.0],
                      [3.0, 4.0,-1.0]])
    a_i1 = 0.
    a_rootj2 = np.sqrt(30)
    print(tens.magnitude(a_dev), np.sqrt(60.))
    print(tens.magnitude(A_dev), np.sqrt(60.))
    # Do it with 6x1
    i1, rootj2, j3 = tens.invariants(a_dev, 'mechanics')
    assert np.isclose(a_i1, i1)
    assert np.isclose(a_rootj2, rootj2)
    assert np.isclose(36.0, j3)
    # Now do it with 3x3
    i1, rootj2, j3 = tens.invariants(A_dev, 'mechanics')
    assert np.isclose(a_i1, i1)
    assert np.isclose(a_rootj2, rootj2)
    assert np.isclose(36.0, j3)

def test_invariants_mechanics_iso():
    a_iso = np.array([4.5, 4.5, 4.5, 0., 0., 0.])
    A_iso = np.eye(3) * 4.5
    a_i1 = 3 * 4.5
    # Do it with 6x1
    i1, rootj2, j3 = tens.invariants(a_iso, 'mechanics')
    assert np.isclose(a_i1, i1)
    assert np.isclose(0., rootj2)
    assert np.isclose(0., j3)
    # Now do it with 3x3
    i1, rootj2, j3 = tens.invariants(A_iso, 'mechanics')
    assert np.isclose(a_i1, i1)
    assert np.isclose(0., rootj2)
    assert np.isclose(0., j3)

def test_invariants_lode():
    def f(x):
        fac = -1.0 + 2.0 * x
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, fac, 0.0],
                         [0.0, 0.0,-1.0]])
    z, r, theta, lode = tens.invariants(f(.5), 'lode')
    assert np.isclose(       0.0, theta)
    z, r, theta, lode = tens.invariants(f(1.), 'lode')
    assert np.isclose(-np.pi/6.0, theta)
    z, r, theta, lode = tens.invariants(f(0.), 'lode')
    assert np.isclose( np.pi/6.0, theta)

def test_magnitude():
    a_ident = np.array([1., 1., 1., 0., 0., 0.])
    A_ident = np.eye(3)
    assert isclose(tens.magnitude(a_ident), np.sqrt(3.))
    assert isclose(tens.magnitude(A_ident), np.sqrt(3.))

    mag = np.sqrt(1 + 2. * .5 ** 2)
    a_dev = np.array([1., -.5, -.5, 0., 0., 0.])
    A_dev = np.array([[1., 0., 0.], [0., -.5, 0.], [0., 0., -.5]])
    assert isclose(tens.magnitude(a_dev), mag)
    assert isclose(tens.magnitude(A_dev), mag)

def test_matrix_rep():
    """Test array to matrix conversion"""
    a = np.array([1,4,6,2,5,3])
    A = np.array([[1,2,3],[2,4,5],[3,5,6]])
    ax, shape = tens.matrix_rep(a)
    assert np.allclose(ax, A)
    assert shape == (6,)
    Ax, shape = tens.matrix_rep(A)
    assert np.allclose(Ax, A)
    assert shape == (3,3)
    Ax = tens.array_rep(A, (6,))
    assert np.allclose(Ax, a)
    Ax = tens.array_rep(A, (3,3))
    assert np.allclose(Ax, A)

def test_trace():
    a_ident = np.array([1., 1., 1., 0., 0., 0.])
    A_ident = np.eye(3)
    assert isclose(tens.trace(a_ident), 3.)
    assert isclose(tens.trace(A_ident), 3.)

    a_dev = np.array([1., -.5, -.5, 0., 0., 0.])
    A_dev = np.array([[1., 0., 0.], [0., -.5, 0.], [0., 0., -.5]])
    assert isclose(tens.trace(a_dev), 0.)
    assert isclose(tens.trace(A_dev), 0.)

def test_det():
    """Determinant of A"""
    x, y, z = 1., 2., 3.
    a = np.array([x, y, z, 0, 0, 0])
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    assert isclose(tens.det(a), x*y*z)
    assert isclose(tens.det(A), x*y*z)

def test_inv():
    """Inverse of A"""
    x, y, z = 1., 2., 3.
    fun = lambda _: 1./ _
    tens_fun = tens.inv
    a = np.array([x, y, z, 0, 0, 0])
    a_fun = np.array([fun(x), fun(y), fun(z), 0, 0, 0])
    A = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]])
    A_fun = np.array([[fun(x), 0, 0], [0, fun(y), 0], [0, 0, fun(z)]])
    assert np.allclose(tens_fun(a), a_fun)
    assert np.allclose(tens_fun(A), A_fun)


if __name__ == '__main__':
    test_import()
