import pytest

import numpy as np
from matmodlab2 import *
from testing_utils import *


def test_viscoelastic():

    mps = MaterialPointSimulator("viscoelastic", initial_temp=75)

    parameters = {"D1": 1.0e-15, "C10": 1e6, "C01": 0.1e6}

    wlf = [75, 35, 50]
    prony = np.array(
        [
            [0.35, 600.0],
            [0.15, 20.0],
            [0.25, 30.0],
            [0.05, 40.0],
            [0.05, 50.0],
            [0.15, 60.0],
        ]
    )

    material = PolynomialHyperelasticMaterial(**parameters)
    material.Expansion(1e-5)
    material.Viscoelastic(wlf, prony)
    mps.material = material

    mps.add_step("ESS", (0.1, 0.0, 0.0), increment=1.0, temperature=75.0, frames=10)
    mps.add_step("EEE", (0.0, 0.0, 0.0), increment=50.0, temperature=95.0, frames=50)

    try:
        # test passes if it runs
        mps.run()
    except BaseException:
        raise Exception("viscoelastic failed to run")
