"""Materials do not need to subclass the matmodlab2.core.material.Material base
class, they need only provide an `eval` method. This test verifies that that is
the case.

"""
import pytest

import numpy as np
from matmodlab2 import *
from testing_utils import *

class MyMaterial:
    def __init__(self, **parameters):
        E = parameters['E']
        Nu = parameters['Nu']
        self.G = E / 2. / (1. + Nu)
        self.K = E / 3. / (1. - 2. * Nu)
    def eval(self, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        K3, G2 = 3. * self.K, 2. * self.G
        Lam = (K3 - G2) / 3.
        ddsdde = np.zeros((6,6))
        ddsdde[np.ix_(range(3), range(3))] = Lam
        ddsdde[range(3),range(3)] += G2
        ddsdde[range(3,6),range(3,6)] = self.G
        stress = stress + np.dot(ddsdde, d * dtime)
        return stress, statev, ddsdde

@pytest.mark.user_material
def test_uniaxial_strain():
    K = 9.980040E+09
    G = 3.750938E+09
    E = 9. * K * G / (3. * K + G)
    Nu  = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    parameters = {'K': K, 'G': G, 'E': E, 'Nu': Nu}
    pathtable = [[1.0, 0.0, 0.0],
                 [2.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0]]
    mps = MaterialPointSimulator('Job')
    material = MyMaterial(**parameters)
    mps.assign_material(material)
    for c in pathtable:
        mps.add_step('E', c, scale=-0.5)
    H = K + 4. / 3. * G
    Q = K - 2. / 3. * G
    a = mps.get2('E.XX', 'S.XX', 'S.YY', 'S.ZZ')
    eps_xx = mps.data[:,4]
    assert np.allclose(a[:,2], a[:,3])
    assert np.allclose(a[:,1], H * a[:,0])
    assert np.allclose(a[:,2], Q * a[:,0])
    assert np.allclose(eps_xx, a[:,0])
