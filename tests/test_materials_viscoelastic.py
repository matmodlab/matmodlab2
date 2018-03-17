import pytest

import numpy as np
from matmodlab2 import *
from testing_utils import *

@pytest.mark.material
@pytest.mark.viscoelastic
def test_viscoelastic():

    mps = MaterialPointSimulator('viscoelastic', initial_temp=75)

    parameters = {'D1': 1.e-15, 'C10': 1e6, 'C01': .1e6}

    wlf = [75, 35, 50]
    prony =  np.array([[.35, 600.], [.15, 20.], [.25, 30.],
                       [.05, 40.], [.05, 50.], [.15, 60.]])

    material = PolynomialHyperelasticMaterial(**parameters)
    material.Expansion(1e-5)
    material.Viscoelastic(wlf, prony)
    mps.material = material

    mps.add_step('ESS', (.1, 0., 0.), increment=1.,
                 temperature=75., frames=10)
    mps.add_step('EEE', (0., 0., 0.), increment=50.,
                 temperature=95., frames=50)

    try:
        # test passes if it runs
        mps.run()
    except BaseException:
        raise Exception('viscoelastic failed to run')
