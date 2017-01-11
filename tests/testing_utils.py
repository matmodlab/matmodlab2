"""General testing utilities"""
import os
import glob
import numpy as np
from pandas import read_table
from matmodlab.core.numerix import *

def teardown_module():
    """Remove generated files after a test"""
    for ext in ('.log', '.exo', '.dat', '.npz'):
        for filename in glob.glob('*'+ext):
            os.remove(filename)

def isclose(a, b, tol=1e-8):
    """Is b close to a?"""
    return abs(a - b) <= tol

def same_as_baseline(mps, baseline=None, interp=0,
                     diff_tolerance=1.5e-6, fail_tolerance=1e-4, floor=1e-12):
    """Is the response from the MaterialPointSimulator object mps the same as the
    baseline?"""

    df1 = mps.df
    if baseline is None:
        baseline = os.path.join('data', mps.jobid+'.base_dat')
    df2 = read_table(baseline, sep='\s+')

    t1 = df1['Time']
    t2 = df2['Time']

    if not interp:
        # interpolation will not be used when comparing values, so the
        # timesteps must be equal
        if t1.shape[0] != t2.shape[0]:
            raise Exception('Number of timesteps differ')

        if not np.allclose(t1, t2, atol=fail_tolerance, rtol=fail_tolerance):
            raise Exception('Timestep size in File1 and File2 differ')

    vars_to_compare = [x for x in df1.columns
                       if x in df2.columns and x != 'Time']

    passed, diffed, failed = [], [], []
    for key in vars_to_compare:
        d1 = np.where(df1[key] <= floor, 0., df1[key])
        d2 = np.where(df2[key] <= floor, 0., df2[key])
        if not interp:
            if np.allclose(d1, d2, atol=fail_tolerance, rtol=fail_tolerance):
                passed.append(key)
                continue

        if amag(d1) < 1.e-10 and amag(d2) < 1.e-10:
            passed.append(key)
            continue

        rms, nrms = rms_error(t1, d1, t2, d2)
        if nrms < diff_tolerance:
            passed.append(key)

        elif nrms < fail_tolerance:
            diffed.append(key)

        else:
            failed.append(key)

        continue

    if len(passed) == len(vars_to_compare):
        return True

    return False

def responses_are_same(a, b, vars, diff_tolerance=5e-3, fail_tolerance=1e-2):
    T = a[:, 0]
    t = b[:, 0]
    passed, diffed, failed = 0, 0, 0
    for col in range(1, len(vars)):
        X = a[:, col]
        x = b[:, col]
        nrms = rms_error(T, X, t, x, disp=0)
        if nrms < diff_tolerance:
            passed += 1
        elif nrms < fail_tolerance:
            diffed += 1
        else:
            failed += 1
    return passed == len(vars[1:])

def random_matrix():
    return np.random.rand(9).reshape(3,3)

def random_symmetric_positive_definite_matrix():
    R = random_rotation_matrix()
    L = random_diagonal_matrix()
    X = np.dot(np.dot(R, L), R.T)
    return (X + X.T) / 2.

def random_diagonal_matrix():
    L = np.zeros((3,3))
    L[([0,1,2],[0,1,2])] = np.random.rand(3)
    return L

def random_rotation_matrix():
    theta = np.random.uniform(0, 2*np.pi, 1)[0]
    a = np.random.rand(3)
    a = a / np.sqrt(np.dot(a,a))
    aa = np.outer(a,a)
    I = np.eye(3)
    A = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = I + 2*np.sin(theta/2.)**2*(aa-I)+np.sin(theta)*A
    return R
