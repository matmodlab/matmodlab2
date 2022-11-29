# Hyperelastic parameters, D1 set to a large number to force incompressibility
import pytest
import numpy as np

from matmodlab2 import MaterialPointSimulator, PolynomialHyperelasticMaterial
from testing_utils import *


def test_polynomial_hyperelastic():
    parameters = {"D1": 1.0e-15, "C10": 1e6, "C01": 0.1e6}

    # stretch to 300%
    lam = np.linspace(0.5, 3, 50)

    # Set up the simulator
    mps = MaterialPointSimulator("test1")
    mps.material = PolynomialHyperelasticMaterial(**parameters)

    # Drive the *incompressible* material through a path of uniaxial stress by
    # prescribing the deformation gradient.
    Fij = lambda x: (x, 0, 0, 0, 1 / np.sqrt(x), 0, 0, 0, 1 / np.sqrt(x))
    mps.run_step("F", Fij(lam[0]), frames=10)
    mps.run_step("F", Fij(1), frames=1)
    mps.run_step("F", Fij(lam[-1]), frames=20)

    # analytic solution for true and engineering stress
    C10, C01 = parameters["C10"], parameters["C01"]
    s = 2 * C01 * lam - 2 * C01 / lam**2 + 2 * C10 * lam**2 - 2 * C10 / lam

    lam_ = np.exp(mps.get("E.XX"))
    ss = mps.get("S.XX") - mps.get("S.ZZ")
    print(ss)

    # check the actual solutions
    assert abs(np.amax(ss) - np.amax(s)) / np.amax(s) < 1e-6
    assert abs(np.amin(ss) - np.amin(s)) / np.amin(s) < 1e-6
