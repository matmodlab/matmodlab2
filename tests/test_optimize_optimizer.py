"""Test the optimization capabilities.

Test is a trivial linear elastic material exercised through a path of uniaxial
stress. The tests check that the optimized Young's modulus is close to the
specified value

"""
import os
import glob
import pytest
import shutil
import numpy as np
from matmodlab2 import *
from matmodlab2.optimize import Optimizer, OptimizeVariable
import testing_utils as tu


def teardown_module():
    tu.teardown_module()
    for dirname in glob.glob("*.eval"):
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)


E = 10.0


def func(x, xnames, evald, job, *args):
    parameters = {"E": x[0], "Nu": 0.1}
    mps = MaterialPointSimulator(job)
    mps.material = ElasticMaterial(**parameters)
    mps.run_step("ESS", [0.1, 0.0, 0.0])
    mps.run_step("ESS", [0.0, 0.0, 0.0])
    sxx = np.array(mps.df["S.XX"])
    exx = np.array(mps.df["E.XX"])
    youngs = []
    for (i, e) in enumerate(exx):
        if abs(e) < 1e-12:
            continue
        youngs.append(sxx[i] / e)
    youngs = np.average(youngs)
    return youngs - E


def run_method(method):
    E = OptimizeVariable("E", 8, bounds=(7, 12))
    xinit = [
        E,
    ]
    optimizer = Optimizer(
        method, func, xinit, method=method, maxiter=25, tolerance=1.0e-4
    )
    optimizer.run()
    return optimizer.xopt


def test_optimize_cobyla():
    xopt = run_method("cobyla")
    err = xopt[0] - E
    assert err < 1e-4


def test_powell():
    xopt = run_method("powell")
    err = xopt[0] - E
    assert err < 1e-4


def test_simplex():
    xopt = run_method("simplex")
    err = xopt[0] - E
    assert err < 1e-4
