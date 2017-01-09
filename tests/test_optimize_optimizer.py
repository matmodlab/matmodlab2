"""Test the optimization capabilities.

Test is a trivial linear elastic material exercised through a path of uniaxial
stress. The tests check that the optimized Young's modulus is close to the
specified value

"""
import pytest
import numpy as np
from matmodlab import *
from matmodlab.optimize import Optimizer, OptimizeVariable
from testing_utils import *

E = 10.

def func(x, xnames, evald, job, *args):
    parameters = {'E': x[0], 'Nu': .1}
    mps = MaterialPointSimulator(job)
    mps.material = ElasticMaterial(**parameters)
    mps.add_step('ESS', [.1, 0., 0.])
    mps.add_step('ESS', [0., 0., 0.])
    mps.run()
    sxx = np.array(mps.df['S.XX'])
    exx = np.array(mps.df['E.XX'])
    youngs = []
    for (i, e) in enumerate(exx):
        if abs(e) < 1e-12:
            continue
        youngs.append(sxx[i] / e)
    youngs = np.average(youngs)
    return youngs - E

def run_method(method):
    E = OptimizeVariable('E', 8, bounds=(7, 12))
    xinit = [E,]
    optimizer = Optimizer(method, func, xinit, method=method,
                          maxiter=25, tolerance=1.e-4)
    optimizer.run()
    return optimizer.xopt

@pytest.mark.cobyla
@pytest.mark.optimize
def test_optimize_cobyla():
    xopt = run_method('cobyla')
    err = (xopt[0] - E)
    assert err < 1e-4

@pytest.mark.powell
@pytest.mark.optimize
def test_powell():
    xopt = run_method('powell')
    err = (xopt[0] - E)
    assert err < 1e-4

@pytest.mark.simplex
@pytest.mark.optimize
def test_simplex():
    xopt = run_method('simplex')
    err = (xopt[0] - E)
    assert err < 1e-4
