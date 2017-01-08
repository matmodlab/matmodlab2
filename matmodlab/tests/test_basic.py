import os
import glob
import numpy as np
from matmodlab import *

def teardown_module():
    for ext in ('.exo', '.log'):
        for filename in glob.glob('*'+ext):
            os.remove(filename)

def close(a, b, tol=1e-8):
    return abs(a - b) <= tol

def test_elastic_consistency():
    """Test the elastic and plastic materials for equivalence"""
    environ.SQA = True
    E = 10.
    Nu = .1
    G = E / 2. / (1. + Nu)
    K = E / 3. / (1. - 2. * Nu)

    jobid = 'Job-El'
    mps_el = MaterialPointSimulator(jobid)
    material = ElasticMaterial(E=E, Nu=Nu)
    mps_el.assign_material(material)
    mps_el.add_step('E'*6, [1,0,0,0,0,0], scale=.1, frames=1)
    mps_el.add_step('S'*6, [0,0,0,0,0,0], frames=5)
    mps_el.run()
    df_el = mps_el.df

    jobid = 'Job-Pl'
    mps_pl = MaterialPointSimulator(jobid)
    material = PlasticMaterial(K=K, G=G)
    mps_pl.assign_material(material)
    mps_pl.add_step('E'*6, [1,0,0,0,0,0], scale=.1, frames=1)
    mps_pl.add_step('S'*6, [0,0,0,0,0,0], frames=5)
    mps_pl.run()
    df_pl = mps_pl.df

    for key in ('S.XX', 'S.YY', 'S.ZZ', 'E.XX', 'E.YY', 'E.ZZ'):
        assert np.allclose(df_el[key], df_pl[key])

def test_steps():
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)

    # Volume strain
    s = mps.add_step('E', 1)
    assert close(np.sum(s.components), 1)

    # Uniaxial strain
    s = mps.add_step('EEE', [1, 0, 0], scale=.1)
    assert close(s.components[0], .1)

    # Pressure
    s = mps.add_step('SSS', [1, 1, 1], scale=1.)
    assert close(np.sum(s.components[:3]), 3)
    s = mps.add_step('SSS', [1, 1, 1], scale=2.)
    assert close(np.sum(s.components[:3]), 6)

    # Invalid deformation gradient
    try:
        s = mps.add_step('F', [1, 1, 1], scale=2.)
    except ValueError as e:
        assert e.args[0] == 'Must specify all 9 components of F'

    # Valid deformation gradients
    s = mps.add_step('F', [1.05, 0, 0, 0, 1, 0, 0, 0, 1], kappa=1)
    assert close(s.components[0], .05)
    s = mps.add_step('F', [1.05, 0, 0, 0, 1, 0, 0, 0, 1], kappa=0)
    assert close(s.components[0], np.log(1.05))

    try:
        s = mps.add_step('F', [0, 0, 0, 0, 1, 0, 0, 0, 1], kappa=0)
    except ValueError as e:
        assert e.args[0] == 'Negative or zero initial Jacobian'

    # Invalid deformation gradients (no rotations allowed)
    try:
        s = mps.add_step('F', [1.05, .05, 0, .05, 1, 0, 0, 0, 1], kappa=0)
    except ValueError as e:
        assert e.args[0] == 'QR decomposition of deformation gradient gave unexpected rotations (rotations are not yet supported)'

    # Displacement
    try:
        s = mps.add_step('U', [1.05])
    except ValueError as e:
        assert e.args[0] == 'Must specify all 3 components of U'

    s = mps.add_step('U', [.05, 0., 0.])
    assert close(s.components[0], np.log(1.05))
    assert close(np.sum(s.components[3:]), 0)

    s = mps.add_step('U', [0., .05, 0.])
    assert close(s.components[1], np.log(1.05))
    assert close(np.sum(s.components[3:]), 0)

    s = mps.add_step('U', [0., 0., .05])
    assert close(s.components[2], np.log(1.05))
    assert close(np.sum(s.components[3:]), 0)
