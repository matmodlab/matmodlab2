import pytest

import numpy as np
from matmodlab2 import MaterialPointSimulator, PolynomialHyperelasticMaterial, MooneyRivlinMaterial
from matmodlab2.fitting.hyperopt import hyperopt
from matmodlab2.fitting.hyperfit import HyperFit
from matmodlab2.core.numerix import rms_error, calculate_bounded_area, get_area

@pytest.mark.slow
@pytest.mark.hyperfit
def test_hyperfit():

    # Run a simulation to generate data
    parameters = {'D1': 1.5e-5, 'C10': 1e6, 'C01': .1e6}

    # Set up the simulator
    mps = MaterialPointSimulator('test1')
    mps.material = PolynomialHyperelasticMaterial(**parameters)
    #mps.material = MooneyRivlinMaterial(**parameters)

    # Drive the *incompressible* material through a path of uniaxial stress by
    # prescribing the deformation gradient.
    mps.add_step('ESS', (1., 0, 0), frames=50)

    # Now fit the model
    dtype = 'Uniaxial Data'
    e1 = mps.get('E.XX')
    s1 = mps.get('S.XX')
    opt = hyperopt('Uniaxial Data', e1, s1, order=2, i2dep=False)

    mps = MaterialPointSimulator('test2')
    parameters['C10'] = opt.popt[0]
    parameters['C01'] = opt.popt[1]
    mps.material = PolynomialHyperelasticMaterial(**parameters)
    mps.add_step('ESS', (1., 0, 0), frames=50)
    s2 = mps.get('S.XX')
    t = mps.get('Time')
    assert rms_error(t, s1, t, s2, 0) < .01
