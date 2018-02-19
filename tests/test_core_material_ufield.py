import pytest

import numpy as np
from matmodlab2 import *
from testing_utils import *

from matmodlab2.core.logio import logger
from matmodlab2.core.material import Material
from matmodlab2.core.tensor import VOIGT, deviatoric_part, isotropic_part

class UfieldDependentMaterial(Material):
    name = 'ufield-mat'
    def __init__(self, K, G, num_ufield):
        self.num_ufield = num_ufield
        self.params = {'K': K, 'G': G}
        self.sdv_names = ['UFIELD{0}'.format(i) for i in range(num_ufield)]
        self.sdv_names.extend(['DUFIELD{0}'.format(i) for i in range(num_ufield)])
        self.num_sdv = len(self.sdv_names)
    def eval(self, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        ufield = kwds['ufield']
        # ufield: Array of interpolated values of user defined variables
        # at the start of the increment.
        dufield = kwds['dufield']
        # dufield: Array of increments of user field variables
        de = d / VOIGT * dtime
        stress += (3.0 * self.params['K'] * isotropic_part(de) +
                   2.0 * self.params['G'] * deviatoric_part(de))
        for i in range(self.num_ufield):
            statev[i] = ufield[i]+dufield[i]
            statev[i+self.num_ufield] = dufield[i]
        return stress, statev, None
    def sdvini(self, statev):
        for i in range(self.num_ufield):
            statev[i] = float(i+1)
            statev[i+self.num_ufield] = 0.
        return statev

@pytest.mark.ufield
@pytest.mark.material
def test_ufield_scalar():
    E, Nu = 500, .45
    K = E / 3. / (1. - 2. * Nu)
    G = E / (2.0 * (1.0 + Nu))

    # Number of user defined field variables
    mps = MaterialPointSimulator('Job-1', ufield=1.)
    assert mps.num_ufield == 1

    mps.material = UfieldDependentMaterial(K, G, 1)

    # Run a step.  The value of the ufield should be same as the initial value
    # since it is not passed.
    mps.run_step('EEE', (.01,0,0))
    out = mps.get2('UFIELD0', 'DUFIELD0')
    assert np.allclose(out[:,0], 1.)
    assert np.allclose(out[:,1], 0.)

    mps.run_step('EEE', (.02,0,0), ufield=1.)
    assert np.allclose(out[:,0], 1.)
    assert np.allclose(out[:,1], 0.)

    mps.run_step('EEE', (.03,0,0), ufield=2., frames=10)
    out = mps.get2('UFIELD0', 'DUFIELD0')
    assert np.allclose(out[3:,0], np.linspace(1.1, 2, 10))
    assert np.allclose(out[3:,1], .1)

    try:
        mps.run_step('EEE', (.01,0,0), ufield=[2., 1.])
        assert False, 'Should have failed because of bad ufield shape'
    except:
        pass

    return

@pytest.mark.ufield
@pytest.mark.material
def test_ufield_vector():
    E, Nu = 500, .45
    K = E / 3. / (1. - 2. * Nu)
    G = E / (2.0 * (1.0 + Nu))

    # Number of user defined field variables
    mps = MaterialPointSimulator('Job-1', ufield=[1., 2.])
    assert mps.num_ufield == 2

    mps.material = UfieldDependentMaterial(K, G, 2)

    # Run a step.  The value of the ufield should be same as the initial value
    # since it is not passed.
    mps.run_step('EEE', (.01,0,0))
    out = mps.get2('UFIELD0', 'UFIELD1', 'DUFIELD0', 'DUFIELD1')
    assert np.allclose(out[:,0], 1.)
    assert np.allclose(out[:,1], 2.)
    assert np.allclose(out[:,2], 0.)
    assert np.allclose(out[:,3], 0.)

    mps.run_step('EEE', (.02,0,0), ufield=[1., 2.])
    out = mps.get2('UFIELD0', 'UFIELD1', 'DUFIELD0', 'DUFIELD1')
    assert np.allclose(out[:,0], 1.)
    assert np.allclose(out[:,1], 2.)
    assert np.allclose(out[:,2], 0.)
    assert np.allclose(out[:,3], 0.)

    mps.run_step('EEE', (.03,0,0), ufield=[2., 1.], frames=10)
    out = mps.get2('UFIELD0', 'UFIELD1', 'DUFIELD0', 'DUFIELD1')
    assert np.allclose(out[3:,0], np.linspace(1.1, 2, 10))
    assert np.allclose(out[3:,1], np.linspace(1.9, 1, 10))
    assert np.allclose(out[3:,2], .1)
    assert np.allclose(out[3:,3], -.1)

    try:
        mps.run_step('EEE', (.02,0,0), ufield=1.)
        assert False, 'Should have failed because of bad ufield shape'
    except:
        pass

    return
