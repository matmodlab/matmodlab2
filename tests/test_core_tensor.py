# -*- coding: utf-8 -*-
"""
This file contains tests for tensor.py
"""

import sys
import pathlib
import pytest
import numpy as np
from testing_utils import isclose

# Ensure that 'matmodlab' is imported from parent directory.
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent))

try:
    import matmodlab
except ImportError:
    matmodlab = None

import matmodlab.core.tensor as tens

def vec_isclose(name, comp, gold, rtol=1.0e-12, atol=1.0e-12):
    print("===== {0}".format(name))
    print("comp:", comp)
    print("gold:", gold)
    print("diff:", gold - comp)
    PASS = np.allclose(comp, gold, rtol=rtol, atol=atol)
    print("PASS" if PASS else "FAIL")
    return PASS

tovoigt = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

deformation_measures_db = [
     {"name": "Uniaxial Extension",
      "eps": np.array([0.042857142857142857143,0,0,0,0,0]),
      "depsdt": np.array([0.10000000000000000000,0,0,0,0,0]),
      "subtests": [
           {
            "k": 2,
            "u": np.array([1.0419761445034553738,1.0000000000000000000,1.0000000000000000000,0,0,0]),
            "dudt": np.array([0.095971486993739310740,0,0,0,0,0]),
            "d": np.array([0.092105263157894736842,0,0,0,0,0]),
           },
           {
           "k": 1,
           "u": np.array([1.0428571428571428571,1.0000000000000000000,1.0000000000000000000,0,0,0]),
           "dudt": np.array([0.10000000000000000000,0,0,0,0,0]),
           "d": np.array([0.095890410958904109589,0,0,0,0,0]),
           },
           {
           "k": 0,
           "u": np.array([1.0437887715175541853,1.0000000000000000000,1.0000000000000000000,0,0,0]),
           "dudt": np.array([0.10437887715175541853,0,0,0,0,0]),
           "d": np.array([0.10000000000000000000,0,0,0,0,0]),
           },
           {
           "k": -1,
           "u": np.array([1.0447761194029850746,1.0000000000000000000,1.0000000000000000000,0,0,0]),
           "dudt": np.array([0.10915571396747605257,0,0,0,0,0]),
           "d": np.array([0.10447761194029850746,0,0,0,0,0]),
           },
           {
           "k": -2,
           "u": np.array([1.0458250331675944350,1.0000000000000000000,1.0000000000000000000,0,0,0]),
           "dudt": np.array([0.11438711300270564133,0,0,0,0,0]),
           "d": np.array([0.10937500000000000000,0,0,0,0,0]),
           },
          ],
     },
     {"name": "Uniaxial Extension with rotation",
      "eps": np.array([0.026196877156206737235,0.016660265700936119908,0,0.020891312403896220150,0,0]),
      "depsdt": np.array([-0.0045059468741139683829,0.10450594687411396838,0,0.063726469853100399588,0,0]),
      "subtests": [
           {
            "k": 2,
            "u": np.array([1.0256583576911247384,1.0163177868123306353,1.0000000000000000000,0.020461857461098139159,0,0]),
            "dudt": np.array([-0.0056192451222061811013,0.10159073211594549184,0,0.061454775148472809312,0,0]),
            "d": np.array([-0.0066876940055755266344,0.098792957163470263477,0,0.059274595960483676859,0,0]),
           },
           {
            "k": 1,
            "u": np.array([1.0261968771562067372,1.0166602657009361199,1.0000000000000000000,0.020891312403896220150,0,0]),
            "dudt": np.array([-0.0045059468741139683829,0.10450594687411396838,0,0.063726469853100399588,0,0]),
            "d": np.array([-0.0056693735828201687630,0.10155978454172427835,0,0.061415383576480024658,0,0]),
           },
           {
            "k": 0,
            "u": np.array([1.0267663449262200007,1.0170224265913341846,1.0000000000000000000,0.021345447796308002806,0,0]),
            "dudt": np.array([-0.0032560207940279426371,0.10763489794578336117,0,0.066186651517750065998,0,0]),
            "d": np.array([-0.0045260401459293278687,0.10452604014592932787,0,0.063731056011271402912,0,0]),
           },
           {
            "k": -1,
            "u": np.array([1.0273698716557383822,1.0174062477472466924,1.0000000000000000000,0.021826744302578140456,0,0]),
            "dudt": np.array([-0.0018481668596687927090,0.11100388082714484528,0,0.068860299997538432155,0,0]),
            "d": np.array([-0.0032383326989564664762,0.10771594463925497394,0,0.066244519079865882721,0,0]),
           },
           {
            "k": -2,
            "u": np.array([1.0280110311733133167,1.0178140019942811183,1.0000000000000000000,0.022338051955872830687,0,0]),
            "dudt": np.array([-0.00025673980976010909772,0.11464385281246575042,0,0.071777050608761226760,0,0]),
            "d": np.array([-0.0017829682784827673453,0.11115796827848276735,0,0.068982906840537447349,0,0]),
           },
      ],
     },
    ]

@pytest.mark.parametrize('a', np.linspace(0.0, 1.0e+1, 10))
@pytest.mark.parametrize('t', np.linspace(0.0, 1.0e+1, 10))
def test_deformation_measures_from_strain_uni_strain(a, t):
    """ Verify that we are converting from strain to D correctly. """

    # Setup
    eps = np.array([a * t, 0.0, 0.0, 0.0, 0.0, 0.0])
    depsdt = np.array([a, 0.0, 0.0, 0.0, 0.0, 0.0])
    d_g = np.array([a, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Test
    d = tens.rate_of_strain_to_rate_of_deformation(depsdt*tovoigt, eps*tovoigt, 0)
    assert vec_isclose("D", d, d_g*tovoigt)

    # Teardown
    pass

def test_deformation_measures_from_strain_dissertation_test():
    """ Verify that we are converting from strain to D correctly. """

    a = 0.5
    t = 0.1

    # Setup (inputs)
    st = np.sin(np.pi * t)
    ct = np.cos(np.pi * t)
    sht = np.sinh(a * t)
    eat = np.exp(a * t)

    eps = np.array([a * t * np.cos(np.pi * t / 2.0) ** 2,
                    a * t * np.sin(np.pi * t / 2.0) ** 2,
                    0.0,
                    a * t * np.sin(np.pi * t) / 2.0,
                    0.0, 0.0])
    depsdt = np.array([a / 2.0 * (1.0 + ct - np.pi * t * st),
                       a / 2.0 * (1.0 - ct + np.pi * t * st),
                       0.0,
                       a / 2.0 * (np.pi * t * ct + st),
                       0.0, 0.0])

    # Setup (expected outputs)
    d_g = np.array([(a + a * ct - np.pi * st * sht) / 2.0,
                    (a - a * ct + np.pi * st * sht) / 2.0,
                    0.0,
                    (a * st + np.pi * ct * sht) / 2.0,
                    0.0, 0.0])

    # Test
    d = tens.rate_of_strain_to_rate_of_deformation(depsdt*tovoigt, eps*tovoigt, 0)
    assert vec_isclose("D", d, d_g*tovoigt)

    # Teardown
    pass

def test_deformation_measures_from_strain_dissertation_static():
    """ Verify that we are converting from strain to D correctly. """

    # Setup (inputs)
    eps=np.array([2.6634453918413015230,0.13875241035650067478,0,0.60791403008229297100,0,0])
    depsdt=np.array([-0.66687706806142212351,1.9745693757537298158,0,4.2494716756395844993,0,0])

    # Setup (expected outputs)
    d_g=np.array([-4.3525785227788080461,5.6602708304711157384,0,11.902909607738023219,0,0])

    # Test
    d = tens.rate_of_strain_to_rate_of_deformation(depsdt*tovoigt, eps*tovoigt, 0)
    assert vec_isclose("D", d, d_g*tovoigt)

    # Teardown
    pass

@pytest.mark.parametrize('db', deformation_measures_db)
@pytest.mark.parametrize('idx', [0, 1, 2, 3, 4])
def test_deformation_measures_from_strain_db(db, idx):
    """
    Test the deformation measures for various values of kappa.
    """

    # Setup (inputs)
    kappa = db['subtests'][idx]['k']
    print("Test name, kappa: {0}, {1}".format(db['name'], kappa))
    eps = db['eps']
    depsdt = db['depsdt']

    # Setup (expected outputs)
    d_g = db['subtests'][idx]['d']
    print("kappa=", kappa)

    # Test
    d = tens.rate_of_strain_to_rate_of_deformation(depsdt*tovoigt,
                                                  eps*tovoigt, kappa)
    assert vec_isclose("D", d, d_g*tovoigt)

    # Teardown
    pass

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

def test_mechanics_invariants_dev():
    a_dev = np.array([1., -.5, -.5, 0., 0., 0.])
    A_dev = np.array([[1., 0., 0.], [0., -.5, 0.], [0., 0., -.5]])
    a_i1 = 0.
    a_mag_a = np.sqrt((1 + 2. * .5 ** 2))
    a_rootj2 = a_mag_a / np.sqrt(2.)
    i1, rootj2 = tens.invariants(a_dev, mechanics=True)
    assert np.allclose(a_i1, i1)
    assert np.allclose(a_rootj2, rootj2)

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

if __name__ == '__main__':
    test_import()
