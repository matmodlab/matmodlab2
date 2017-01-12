"""General testing utilities"""
import os
import sys
import glob
import inspect
import numpy as np
from matmodlab2.core.numerix import *
from matmodlab2.core.database import read_db

def teardown_module():
    """Remove generated files after a test"""
    for ext in ('.log', '.exo', '.dat', '.npz'):
        for filename in glob.glob('*'+ext):
            os.remove(filename)

def my_dirname():
    return os.path.dirname(inspect.stack()[1][1])

def isclose(a, b, tol=1e-8):
    """Is b close to a?"""
    return abs(a - b) <= tol

def find_db_from_jobid(jobid, data_dir, baseline=False):
    """Find the output for jobid"""
    data_dir = data_dir or os.getcwd()
    for ext in ('npz', 'dat', 'exo'):
        prefix = 'base_' if baseline else ''
        ext = '.{0}{1}'.format(prefix, ext)
        filename = os.path.join(data_dir, jobid + ext)
        if os.path.isfile(filename):
            break
    else:
        return None
    return read_db(filename)

class tee:
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        self.fh = open(self.filename, 'w')
        return self
    def write(self, string):
        self.fh.write(string)
        sys.stdout.write(string)
    def __exit__(self, *args):
        self.fh.close()

def afloor(a, floor):
    """Put a floor of zero on array"""
    return np.array(np.where(np.abs(a) <= floor, 0., a))

def same_as_baseline(jobid, simulation_df=None, baseline_df=None,
                     variables_to_compare=None, interp=0,
                     diff_tolerance=1.5e-6, fail_tolerance=1e-4, floor=1e-12,
                     istart=0):
    """Is the response from the simulation the same as the baseline?

    jobid : str
        The Job identification
    simulation_df : DataFrame
        DataFrame of simulation data, if not given the simulation data is
        looked for in ./jobid.npz, jobid.exo, jobid.dat in that order.
    baseline_df : DataFrame
        DataFrame of baseline data, if not given the simulation data is looked
        for in ./data/jobid.base_npz, ./data/jobid.base_exo,
        ./data/jobid.base_dat in that order.
    variables_to_compare : list of tuple
        Each entry in variables_to_compare, if not None, is:
           (key_1, key_2, diff_tolerance, fail_tolerance, floor)
        where key_1 is the key of the simulation variable and key_2 is the key
        in the baseline
    interp : bool
        User interpolation to determine if responses are the same
    *_tolerance : float
        Global tolerance
    istart : int
        The starting point of the data for comparison, default is 0

    """

    dirname_of_caller = os.path.dirname(inspect.stack()[1][1])
    if simulation_df is None:
        data_dir = dirname_of_caller
        simulation_df = find_db_from_jobid(jobid, data_dir)

    if baseline_df is None:
        data_dir = os.path.join(dirname_of_caller, 'data')
        baseline_df = find_db_from_jobid(jobid, data_dir, baseline=True)

    sim_time = np.array(simulation_df['Time'])[istart:]
    base_time = np.array(baseline_df['Time'])[istart:]

    if not interp:
        # interpolation will not be used when comparing values, so the
        # timesteps must be equal
        if sim_time.shape[0] != base_time.shape[0]:
            print(sim_time.shape[0], base_time.shape[0])
            raise Exception('Number of timesteps differ')

        if not np.allclose(sim_time, base_time,
                           atol=fail_tolerance, rtol=fail_tolerance):
            raise Exception('Timestep size in File1 and File2 differ')

    if variables_to_compare is None:
        variables_to_compare = [(x, x, diff_tolerance, fail_tolerance, floor)
                                for x in simulation_df.columns
                                if x in baseline_df.columns and x != 'Time']

    passed, diffed, failed = [], [], []
    for item in variables_to_compare:
        try:
            sim_key, base_key, v_diff_tolerance, v_fail_tolerance, v_floor = item
        except ValueError:
            raise ValueError('expected len(5) variable')

        sim_var_val = afloor(simulation_df[sim_key], v_floor)[istart:]
        base_var_val = afloor(baseline_df[base_key], v_floor)[istart:]
        if not interp:
            if np.allclose(sim_var_val, base_var_val,
                           atol=v_fail_tolerance, rtol=v_fail_tolerance):
                passed.append(sim_key)
                continue

        if amag(sim_var_val) < 1.e-10 and amag(base_var_val) < 1.e-10:
            # Zero
            passed.append(sim_key)
            continue

        rms, nrms = rms_error(sim_time, sim_var_val, base_time, base_var_val)
        if nrms < diff_tolerance:
            passed.append(sim_key)

        elif nrms < fail_tolerance:
            diffed.append((sim_key, rms, nrms))

        else:
            failed.append((sim_key, rms, nrms))

        continue

    if len(passed) == len(variables_to_compare):
        return True

    with tee(os.path.join(dirname_of_caller, jobid+'.failed')) as fh:
        fh.write('======== FAILED ==========\n')
        for item in failed:
            fh.write('{0}: rms={1}, nrms={2}\n'.format(*item))
        fh.write('======== DIFFED ==========\n')
        for item in diffed:
            fh.write('{0}: rms={1}, nrms={2}\n'.format(*item))

    return False

def responses_are_same(jobid, a, b, variables_to_compare,
                       diff_tolerance=5e-3, fail_tolerance=1e-2):
    T = a[:, 0]
    t = b[:, 0]
    passed, diffed, failed = [], [], []
    for (col, variable) in enumerate(variables_to_compare[1:], start=1):
        X = a[:, col]
        x = b[:, col]
        rms, nrms = rms_error(T, X, t, x)
        if nrms < diff_tolerance:
            passed.append(variable)
        elif nrms < fail_tolerance:
            diffed.append((variable, rms, nrms))
        else:
            failed.append((variable, rms, nrms))

    if len(passed) == len(variables_to_compare[1:]):
        return True

    dirname_of_caller = os.path.dirname(inspect.stack()[1][1])
    with tee(os.path.join(dirname_of_caller, jobid+'.failed')) as fh:
        fh.write('======== FAILED ==========\n')
        for item in failed:
            fh.write('{0}: rms={1}, nrms={2}\n'.format(*item))
        fh.write('======== DIFFED ==========\n')
        for item in diffed:
            fh.write('{0}: rms={1}, nrms={2}\n'.format(*item))

    return False

def random_matrix():
    return np.random.rand(9).reshape(3,3)

def random_symmetric_positive_definite_matrix():
    R = random_rotation_matrix()
    L = random_diagonal_matrix()
    X = np.dot(np.dot(R, L), R.T)
    return (X + X.T) / 2.

def random_diagonal_matrix():
    L = np.zeros((3,3))
    L[([0,1,2],[0,1,2])] = np.random.rand(3)
    return L

def random_rotation_matrix():
    theta = np.random.uniform(0, 2*np.pi, 1)[0]
    a = np.random.rand(3)
    a = a / np.sqrt(np.dot(a,a))
    aa = np.outer(a,a)
    I = np.eye(3)
    A = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = I + 2*np.sin(theta/2.)**2*(aa-I)+np.sin(theta)*A
    return R
