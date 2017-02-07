import os
import glob
import numpy as np
import pytest
from matmodlab2 import *
from testing_utils import *

def test_mps_assign_material():
    """Test correctness of assigning a material"""
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    try:
        mps.run_step('E', 1)
    except RuntimeError:
        pass
    else:
        raise Exception('RuntimeError not triggered')
    mps.material = ElasticMaterial(E=10, Nu=.1)
    assert mps.material.name.lower() == 'pyelastic'

@pytest.mark.pandas
def test_mps_db_exo():
    """Test the db"""
    # Run without writing db
    jobid = 'Job-nodb_exo'
    mps1 = MaterialPointSimulator(jobid, db_fmt='exo')
    mps1.material = ElasticMaterial(E=10, Nu=.1)
    mps1.run_step('E', 1)
    assert not os.path.isfile(jobid+'.exo')
    assert not os.path.isfile(jobid+'.log')
    mps1.dump()
    assert os.path.isfile(jobid+'.exo')

    # Test writing to a different filename and reading it
    filename = 'a_different_filename_exo'
    mps1.dump(filename)
    assert os.path.isfile(filename+'.exo')
    df = read_exodb(filename+'.exo')

    exx_1 = mps1.df['E.XX'].iloc[:]
    exx_2 = df['E.XX'].iloc[:]
    assert np.allclose(exx_1, exx_2)

@pytest.mark.pandas
def test_mps_db_npz():
    """Test the db"""
    # Run without writing db
    jobid = 'Job-nodb_npz'
    mps1 = MaterialPointSimulator(jobid, db_fmt='npz')
    mps1.material = ElasticMaterial(E=10, Nu=.1)
    mps1.run_step('E', 1)
    assert not os.path.isfile(jobid+'.npz')
    assert not os.path.isfile(jobid+'.log')
    mps1.dumpz()
    assert os.path.isfile(jobid+'.npz')

    # Test writing to a different filename and reading it
    filename = 'a_different_filename_npz'
    mps1.dumpz(filename)
    assert os.path.isfile(filename+'.npz')
    df = read_npzdb(filename+'.npz')

    exx_1 = mps1.df['E.XX'].iloc[:]
    exx_2 = df['E.XX'].iloc[:]
    assert np.allclose(exx_1, exx_2)

def test_mps_run_step_volume_strain():
    # Volume strain
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    s = mps.run_step('E', 1)
    assert isclose(np.sum(s.components), 1)

def test_mps_run_step_uni_strain():
    # Uniaxial strain
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    s = mps.run_step('EEE', [1, 0, 0], scale=.1)
    assert isclose(s.components[0], .1)

def test_mps_run_step_pres():
    # Pressure
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    s = mps.run_step('SSS', [1, 1, 1], scale=1.)
    assert isclose(np.sum(s.components[:3]), 3)
    s = mps.run_step('SSS', [1, 1, 1], scale=2.)
    assert isclose(np.sum(s.components[:3]), 6)

def test_mps_run_step_ivalid_F_1():
    # Invalid deformation gradient
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    s = mps.run_step('SSS', [1, 1, 1], scale=1.)
    try:
        s = mps.run_step('F', [1, 1, 1], scale=2.)
    except ValueError as e:
        assert e.args[0] == 'Must specify all 9 components of F'

def test_mps_run_step_valid_F_1():
    # Valid deformation gradients
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    s = mps.run_step('F', [1.05, 0, 0, 0, 1, 0, 0, 0, 1], kappa=1)
    assert isclose(s.components[0], .05)
    s = mps.run_step('F', [1.05, 0, 0, 0, 1, 0, 0, 0, 1], kappa=0)
    assert isclose(s.components[0], np.log(1.05))

def test_mps_run_step_invalid_F_2():
    # Invalid deformation gradient
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    try:
        s = mps.run_step('F', [0, 0, 0, 0, 1, 0, 0, 0, 1], kappa=0)
    except ValueError as e:
        assert e.args[0] == 'Negative or zero Jacobian'

def test_mps_run_step_invalid_F_3():
    # Invalid deformation gradients (no rotations allowed)
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    try:
        s = mps.run_step('F', [1.05, .05, 0, .05, 1, 0, 0, 0, 1], kappa=0)
    except ValueError as e:
        assert e.args[0] == 'QR decomposition of deformation gradient gave unexpected rotations (rotations are not yet supported)'

def test_mps_run_step_invalid_displacement_1():
    # Displacement
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    try:
        s = mps.run_step('U', [1.05])
    except ValueError as e:
        assert e.args[0] == 'Must specify all 3 components of U'

def test_mps_run_step_valid_displacement_1():
    # Displacement
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    s = mps.run_step('U', [.05, 0., 0.])
    assert isclose(s.components[0], np.log(1.05))
    assert isclose(np.sum(s.components[3:]), 0)

def test_mps_run_step_valid_displacement_2():
    # Displacement
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    s = mps.run_step('U', [0., .05, 0.])
    assert isclose(s.components[1], np.log(1.05))
    assert isclose(np.sum(s.components[3:]), 0)

def test_mps_run_step_valid_displacement_3():
    # Displacement
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    s = mps.run_step('U', [0., .05, 0.])
    s = mps.run_step('U', [0., 0., .05])
    assert isclose(s.components[2], np.log(1.05))
    assert isclose(np.sum(s.components[3:]), 0)

def test_mps_run_step_invalid_stress():
    # Displacement
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.material = ElasticMaterial(E=10, Nu=.1)
    try:
        s = mps.run_step('S', [1.], kappa=1)
    except ValueError as e:
        assert e.args[0] == 'Stress control requires kappa = 0'
    else:
        raise Exception('Nonzero kappa accepted but should not')
