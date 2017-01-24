import os
import glob
import pytest
import random
import numpy as np

from matmodlab2 import *
from testing_utils import *

K = 9.980040E+09
G = 3.750938E+09
E = 9. * K * G / (3. * K + G)
Nu  = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
parameters = {'K': K, 'G': G, 'E': E, 'Nu': Nu}

@pytest.mark.elastic
@pytest.mark.material
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
    mps_el.run_step('E'*6, [1,0,0,0,0,0], scale=.1, frames=1)
    mps_el.run_step('S'*6, [0,0,0,0,0,0], frames=5)
    df_el = mps_el.df

    jobid = 'Job-Pl'
    mps_pl = MaterialPointSimulator(jobid)
    material = PlasticMaterial(K=K, G=G)
    mps_pl.assign_material(material)
    mps_pl.run_step('E'*6, [1,0,0,0,0,0], scale=.1, frames=1)
    mps_pl.run_step('S'*6, [0,0,0,0,0,0], frames=5)
    df_pl = mps_pl.df

    for key in ('S.XX', 'S.YY', 'S.ZZ', 'E.XX', 'E.YY', 'E.ZZ'):
        assert np.allclose(df_el[key], df_pl[key])

@pytest.mark.elastic
@pytest.mark.material
def test_uniaxial_strain():
    pathtable = [[1.0, 0.0, 0.0],
                 [2.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0]]
    mps = MaterialPointSimulator('elastic_unistrain')
    material = ElasticMaterial(**parameters)
    mps.assign_material(material)
    for c in pathtable:
        mps.run_step('E', c, scale=-0.5)
    H = K + 4. / 3. * G
    Q = K - 2. / 3. * G
    a = mps.get2('E.XX', 'S.XX', 'S.YY', 'S.ZZ')
    eps_xx = mps.data[:,4]
    assert np.allclose(a[:,2], a[:,3])
    assert np.allclose(a[:,1], H * a[:,0])
    assert np.allclose(a[:,2], Q * a[:,0])
    assert np.allclose(eps_xx, a[:,0])

@pytest.mark.elastic
@pytest.mark.material
def test_uniaxial_stress():
    pathtable = [[1.0, 0.0, 0.0],
                 [2.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0]]
    mps = MaterialPointSimulator('elastic_unistress')
    material = ElasticMaterial(**parameters)
    mps.assign_material(material)
    for c in pathtable:
        mps.run_step('SSS', c, frames=50, scale=-1.e6)
    a = mps.get2('E.XX', 'S.XX', 'S.YY', 'S.ZZ')
    assert np.allclose(a[:,2], 0)
    assert np.allclose(a[:,3], 0)
    diff = (a[:,1] - parameters['E'] * a[:,0]) / parameters['E']
    assert max(abs(diff)) < 1e-10

@pytest.mark.elastic
@pytest.mark.material
def test_uniaxial_strain_with_stress_control():
    pathtable = [[ -7490645504., -3739707392., -3739707392.],
                 [-14981291008., -7479414784., -7479414784.],
                 [ -7490645504., -3739707392., -3739707392.],
                 [           0.,           0.,           0.]]
    mps = MaterialPointSimulator('elastic_unistrain_stressc')
    material = ElasticMaterial(**parameters)
    mps.assign_material(material)
    for c in pathtable:
        mps.run_step('SSS', c, frames=250)
    a = mps.get2('E.XX', 'E.YY', 'E.ZZ', 'S.XX')
    assert np.allclose(a[:,1], 0)
    assert np.allclose(a[:,2], 0)
    H = K + 4. / 3. * G
    diff = (a[:,3] - H * a[:,0]) / H
    assert max(abs(diff)) < 1e-7

@pytest.mark.elastic
@pytest.mark.material
@pytest.mark.parametrize('realization', range(1,4))
def test_random_linear_elastic(realization):
    difftol = 5.e-08
    failtol = 1.e-07
    myvars = ('Time',
              'E.XX', 'E.YY', 'E.ZZ', 'E.XY', 'E.YZ', 'E.XZ',
              'S.XX', 'S.YY', 'S.ZZ', 'S.XY', 'S.YZ', 'S.XZ')
    jobid = 'rand_linear_elastic_{0}'.format(realization)
    mps = MaterialPointSimulator(jobid)
    NU, E, K, G, LAM = gen_rand_elast_params()
    material = ElasticMaterial(E=E, Nu=NU)
    mps.assign_material(material)
    analytic = gen_analytical_response(LAM, G)
    for (i, row) in enumerate(analytic[1:], start=1):
        incr = analytic[i, 0] - analytic[i-1, 0]
        mps.run_step('E', row[1:7], increment=incr, frames=10)
    simulation = mps.get2(*myvars)
    assert responses_are_same(jobid, analytic, simulation, myvars)

@pytest.mark.elastic
@pytest.mark.material
@pytest.mark.analytic
def test_supreme():
    ''' This test is 'supreme' because it compares the following values
    against the analytical solution:

    * Stress
    * Strain
    * Deformation gradient
    * Symmetric part of the velocity gradient

    This is meant to be a static test for linear elasticity. It's primary
    purpose is to be THE benchmark for linear elasticity as it checks each
    component of stress/strain as well as exercises key parts of the
    driver (like how it computes inputs).

    For uniaxial strain:

        | a  0  0 |                | exp(a)  0  0 |
    e = | 0  0  0 |            U = |   0     1  0 |
        | 0  0  0 |                |   0     0  1 |

        -1  | 1/exp(a)  0  0 |    dU   da | exp(a)  0  0 |
    U  = |    0      1  0 |    -- = -- |   0     0  0 |
            |    0      0  1 |    dt   dt |   0     0  0 |

            da | 1  0  0 |
    D = L = -- | 0  0  0 |
            dt | 0  0  0 |


    For pure shear

        | 0  a  0 |         1         | exp(2a)+1  exp(2a)-1  0 |   | 0  0  0 |
    e = | a  0  0 |     U = - exp(-a) | exp(2a)-1  exp(2a)+1  0 | + | 0  0  0 |
        | 0  0  0 |         2         |     0          0      0 |   | 0  0  1 |


        -1  1 | exp(-a) + exp(a)  exp(-a) - exp(a)  0 |
    U  = - | exp(-a) - exp(a)  exp(-a) + exp(a)  0 |
            2 |         0                 0         2 |


    dU   da / | exp(a)  exp(a)  0 |     \
    -- = -- | | exp(a)  exp(a)  0 | - U |
    dt   dt \ |   0       0     1 |     /

            da | 0  1  0 |
    D = L = -- | 1  0  0 |
            dt | 0  0  0 |

    '''
    difftol = 5.e-08
    failtol = 1.e-07
    jobid = 'supreme_linear_elastic'
    mps = MaterialPointSimulator(jobid)

    N = 25
    solfile = os.path.join(os.getcwd(), 'data', mps.jobid + '.base_dat')
    path, LAM, G, tablepath = generate_solution(solfile, N)

    # set up the material
    K = LAM + 2.0 * G / 3.0
    E = 9. * K * G / (3. * K + G)
    Nu  = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    params = {'E': E, 'Nu': Nu}
    material = ElasticMaterial(**params)
    mps.assign_material(material)

    for row in tablepath:
        mps.run_step('E', row, increment=1.0, frames=N)

    # check output with analytic (all shared variables)
    assert same_as_baseline(mps.jobid, mps.df)

def get_D_E_F_SIG(dadt, a, LAM, G, loc):
    # This is just an implementation of the above derivations.
    #
    # 'dadt' is the current time derivative of the strain
    # 'a' is the strain at the end of the step
    # 'LAM' and 'G' are the lame and shear modulii
    # 'loc' is the index for what's wanted (0,1) for xy

    if loc[0] == loc[1]:
        # axial
        E = np.zeros((3,3))
        E[loc] = a

        F = np.eye(3)
        F[loc] = np.exp(a)

        D = np.zeros((3,3))
        D[loc] = dadt

        SIG = LAM * a * np.eye(3)
        SIG[loc] = (LAM + 2.0 * G) * a
    else:
        # shear
        l0, l1 = loc

        E = np.zeros((3,3))
        E[l0, l1] = a
        E[l1, l0] = a

        fac = np.exp(-a) / 2.0
        F = np.eye(3)
        F[l0,l0] = fac * (np.exp(2.0 * a) + 1.0)
        F[l1,l1] = fac * (np.exp(2.0 * a) + 1.0)
        F[l0,l1] = fac * (np.exp(2.0 * a) - 1.0)
        F[l1,l0] = fac * (np.exp(2.0 * a) - 1.0)

        D = np.zeros((3,3))
        D[l0,l1] = dadt
        D[l1,l0] = dadt

        SIG = np.zeros((3,3))
        SIG[l0,l1] = 2.0 * G * a
        SIG[l1,l0] = 2.0 * G * a

    return D, E, F, SIG

def generate_solution(solfile, N):
    # solfile = filename to write analytical solution to
    # N = number of steps per leg
    a = 0.1                 # total strain increment for each leg
    LAM = 1.0e9             # Lame modulus
    G = 1.0e9               # Shear modulus
    T = [0.0]               # time
    E = [np.zeros((3,3))]   # strain
    SIG = [np.zeros((3,3))] # stress
    F = [np.eye(3)]         # deformation gradient
    D = [np.zeros((3,3))]   # symmetric part of velocity gradient

    #
    # Generate the analytical solution
    #
    # strains:    xx     yy     zz     xy     xz     yz
    for loc in [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]:
        t0 = T[-1]
        tf = t0 + 1.0
        for idx in range(1, N+1):
            fac = float(idx) / float(N)
            ret = get_D_E_F_SIG(a, fac * a, LAM, G, loc)
            T.append(t0 + fac)
            D.append(ret[0])
            E.append(ret[1])
            F.append(ret[2])
            SIG.append(ret[3])

        for idx in range(1, N+1):
            fac = float(idx) / float(N)
            ret = get_D_E_F_SIG(-a, (1.0 - fac) * a, LAM, G, loc)
            T.append(t0 + 1.0 + fac)
            D.append(ret[0])
            E.append(ret[1])
            F.append(ret[2])
            SIG.append(ret[3])

    #
    # Write the output
    #
    headers = ['Time',
               'E.XX', 'E.YY', 'E.ZZ', 'E.XY', 'E.YZ', 'E.XZ',
               'S.XX', 'S.YY', 'S.ZZ', 'S.XY', 'S.YZ', 'S.XZ',
               'F.XX', 'F.XY', 'F.XZ',
               'F.YX', 'F.YY', 'F.YZ',
               'F.ZX', 'F.ZY', 'F.ZZ',
               'D.XX', 'D.YY', 'D.ZZ', 'D.XY', 'D.YZ', 'D.XZ']
    symlist = lambda x: [x[0,0], x[1,1], x[2,2], x[0,1], x[1,2], x[0,2]]
    matlist = lambda x: list(np.reshape(x, 9))
    fmtstr = lambda x: '{0:>25s}'.format(x)
    fmtflt = lambda x: '{0:25.15e}'.format(x)

    with open(solfile, 'w') as FOUT:
        FOUT.write(''.join(map(fmtstr, headers)) + '\n')
        for idx in range(0, len(T)):
            vals = ([T[idx]] +
                     symlist(E[idx]) +
                     symlist(SIG[idx]) +
                     matlist(F[idx]) +
                     symlist(D[idx]))
            FOUT.write(''.join(map(fmtflt, vals)) + '\n')

    #
    # Pass the relevant data so the sim can run
    #

    # inputs    xx   yy   zz   xy   yz   xz
    path = '''
    0 0 222222 0.0  0.0  0.0  0.0  0.0  0.0
    1 1 222222 {0}  0.0  0.0  0.0  0.0  0.0
    2 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    3 1 222222 0.0  {0}  0.0  0.0  0.0  0.0
    4 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    5 1 222222 0.0  0.0  {0}  0.0  0.0  0.0
    6 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    7 1 222222 0.0  0.0  0.0  {0}  0.0  0.0
    8 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    9 1 222222 0.0  0.0  0.0  0.0  0.0  {0}
   10 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
   11 1 222222 0.0  0.0  0.0  0.0  {0}  0.0
   12 1 222222 0.0  0.0  0.0  0.0  0.0  0.0
    '''.format('{0:.1f}'.format(a))

    tablepath = ((  a, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0,   a, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0,   a, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0,   a, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0,   a),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 (0.0, 0.0, 0.0, 0.0,   a, 0.0),
                 (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    return path, LAM, G, tablepath

def get_stress(e11, e22, e33, e12, e23, e13, LAM, G):
    #standard hooke's law
    sig11 = (2.0 * G + LAM) * e11 + LAM * (e22 + e33)
    sig22 = (2.0 * G + LAM) * e22 + LAM * (e11 + e33)
    sig33 = (2.0 * G + LAM) * e33 + LAM * (e11 + e22)
    sig12 = 2.0 * G * e12
    sig23 = 2.0 * G * e23
    sig13 = 2.0 * G * e13
    return sig11, sig22, sig33, sig12, sig23, sig13

def gen_rand_elast_params():
    # poisson_ratio and young's modulus
    nu = random.uniform(-1.0 + 1.0e-5, 0.5 - 1.0e-5)
    E = max(1.0, 10 ** random.uniform(0.0, 12.0))

    # K and G are used for parameterization
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # LAM is used for computation
    LAM = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    return nu, E, K, G, LAM

def const_elast_params():
    K = 9.980040E+09
    G = 3.750938E+09
    LAM = K - 2.0 / 3.0 * G
    E   = 9.0 * K * G / (3.0 * K + G)
    NU  = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return NU, E, K, G, LAM

def gen_analytical_response(LAM, G, nlegs=4, test_type="PRINCIPAL"):
    stiff = (LAM * np.outer(np.array([1,1,1,0,0,0]), np.array([1,1,1,0,0,0])) +
             2.0 * G * np.identity(6))

    rnd = lambda: random.uniform(-0.01, 0.01)
    table = [np.zeros(1 + 6 + 6)]
    for idx in range(1, nlegs):
        if test_type == "FULL":
            strains = np.array([rnd(), rnd(), rnd(), rnd(), rnd(), rnd()])
        elif test_type == "PRINCIPAL":
            strains = np.array([rnd(), rnd(), rnd(), 0.0, 0.0, 0.0])
        elif test_type == "UNIAXIAL":
            strains = np.array([rnd(), 0.0, 0.0, 0.0, 0.0, 0.0])
        elif test_type == "BIAXIAL":
            tmp = rnd()
            strains = np.array([tmp, tmp, 0.0, 0.0, 0.0, 0.0])
        table.append(np.hstack(([idx], strains, np.dot(stiff, strains))))

    # returns a tablewith each row comprised of
    # time=table[0], strains=table[1:7], stresses=table[7:]
    return np.array(table)
