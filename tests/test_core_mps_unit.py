import os
import glob
import numpy as np
from matmodlab import *
from testing_utils import *

def test_mps_assign_material():
    """Test correctness of assigning a material"""
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)
    mps.add_step('E', 1)
    # This should result in an error since the material has not been assigned!
    try:
        mps.run()
    except RuntimeError:
        pass
    else:
        raise Exception('RuntimeError not triggered')
    mps.material = ElasticMaterial(E=10, Nu=.1)
    mps.run()

def test_mps_db_exo():
    """Test the db"""
    # Run without writing db
    jobid = 'Job-nodb_exo'
    mps1 = MaterialPointSimulator(jobid, write_db=False, db_fmt='exo')
    mps1.add_step('E', 1)
    mps1.material = ElasticMaterial(E=10, Nu=.1)
    mps1.run()
    assert not os.path.isfile(jobid+'.exo')
    assert not os.path.isfile(jobid+'.log')
    mps1.dump()
    assert os.path.isfile(jobid+'.exo')

    # Run and write db at each step/frame
    jobid = 'Job-yesdb_exo'
    mps2 = MaterialPointSimulator(jobid, db_fmt='exo')
    mps2.add_step('E', 1)
    mps2.material = ElasticMaterial(E=10, Nu=.1)
    mps2.run()
    assert os.path.isfile(jobid+'.exo')
    assert os.path.isfile(jobid+'.log')

    # Test writing to a different filename and reading it
    filename = 'a_different_filename_exo'
    mps2.dump(filename)
    assert os.path.isfile(filename+'.exo')
    df = DatabaseFile(filename+'.exo')

    exx_1 = mps1.df['E.XX'].iloc[:]
    exx_2 = mps2.df['E.XX'].iloc[:]
    exx_3 = df['E.XX'].iloc[:]
    assert np.allclose(exx_1, exx_2)
    assert np.allclose(exx_1, exx_3)

def test_mps_db_npz():
    """Test the db"""
    # Run without writing db
    jobid = 'Job-nodb_npz'
    mps1 = MaterialPointSimulator(jobid, write_db=False, db_fmt='npz')
    mps1.add_step('E', 1)
    mps1.material = ElasticMaterial(E=10, Nu=.1)
    mps1.run()
    assert not os.path.isfile(jobid+'.npz')
    assert not os.path.isfile(jobid+'.log')
    mps1.dumpz()
    assert os.path.isfile(jobid+'.npz')

    # Run and write db at each step/frame
    jobid = 'Job-yesdb_npz'
    mps2 = MaterialPointSimulator(jobid, db_fmt='npz')
    mps2.add_step('E', 1)
    mps2.material = ElasticMaterial(E=10, Nu=.1)
    mps2.run()
    assert os.path.isfile(jobid+'.npz')
    assert os.path.isfile(jobid+'.log')

    # Test writing to a different filename and reading it
    filename = 'a_different_filename_npz'
    mps2.dumpz(filename)
    assert os.path.isfile(filename+'.npz')
    df = DatabaseFile(filename+'.npz')

    exx_1 = mps1.df['E.XX'].iloc[:]
    exx_2 = mps2.df['E.XX'].iloc[:]
    exx_3 = df['E.XX'].iloc[:]
    assert np.allclose(exx_1, exx_2)
    assert np.allclose(exx_1, exx_3)

def test_mps_add_step():
    jobid = 'Job-1'
    mps = MaterialPointSimulator(jobid)

    # Volume strain
    s = mps.add_step('E', 1)
    assert isclose(np.sum(s.components), 1)

    # Uniaxial strain
    s = mps.add_step('EEE', [1, 0, 0], scale=.1)
    assert isclose(s.components[0], .1)

    # Pressure
    s = mps.add_step('SSS', [1, 1, 1], scale=1.)
    assert isclose(np.sum(s.components[:3]), 3)
    s = mps.add_step('SSS', [1, 1, 1], scale=2.)
    assert isclose(np.sum(s.components[:3]), 6)

    # Invalid deformation gradient
    try:
        s = mps.add_step('F', [1, 1, 1], scale=2.)
    except ValueError as e:
        assert e.args[0] == 'Must specify all 9 components of F'

    # Valid deformation gradients
    s = mps.add_step('F', [1.05, 0, 0, 0, 1, 0, 0, 0, 1], kappa=1)
    assert isclose(s.components[0], .05)
    s = mps.add_step('F', [1.05, 0, 0, 0, 1, 0, 0, 0, 1], kappa=0)
    assert isclose(s.components[0], np.log(1.05))

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
    assert isclose(s.components[0], np.log(1.05))
    assert isclose(np.sum(s.components[3:]), 0)

    s = mps.add_step('U', [0., .05, 0.])
    assert isclose(s.components[1], np.log(1.05))
    assert isclose(np.sum(s.components[3:]), 0)

    s = mps.add_step('U', [0., 0., .05])
    assert isclose(s.components[2], np.log(1.05))
    assert isclose(np.sum(s.components[3:]), 0)
