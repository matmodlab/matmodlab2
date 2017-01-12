import os
import glob
import pytest
import random
from math import *
import numpy as np

from matmodlab2 import *
from testing_utils import *
from matmodlab2.core.tensor import I6

@pytest.mark.material
@pytest.mark.drucker_prager
@pytest.mark.spherical
def test_spherical_drucker_prager():
    jobid = 'lin_druck_prag_spher'
    mps = MaterialPointSimulator(jobid)
    # Elastic modulii
    LAM = 1.0e9
    MU = 1.0e8
    K = LAM + 2.0 / 3.0 * MU
    # Intersects
    FAC = 1.0e6
    RINT = 1.0 * FAC
    ZINT = sqrt(2.0) * FAC
    parameters = {'K': K, 'G': MU,
                    'A1': RINT/sqrt(2.0), 'A4': RINT/sqrt(6.0)/ZINT}
    material = PlasticMaterial(**parameters)
    mps.assign_material(material)
    head, data = gen_spherical_everything(K, MU, RINT, ZINT)
    i = head.index('E.XX')
    j = head.index('E.XX') + 6
    t = head.index('Time')
    for (k, row) in enumerate(data[1:], start=1):
        dt = data[k][t] - data[k-1][t]
        mps.add_step('E', row[i:j], increment=dt, frames=1)
    mps.run()
    assert same_as_baseline(mps.jobid, mps.df, interp=1)

@pytest.mark.material
@pytest.mark.drucker_prager
@pytest.mark.random
@pytest.mark.analytic
@pytest.mark.parametrize('realization', range(1,4))
def test_random_linear_drucker_prager(realization):
    myvars = ('Time', 'E.XX', 'E.YY', 'E.ZZ', 'S.XX', 'S.YY', 'S.ZZ')
    jobid = 'linear_drucker_prager_rand_{0}'.format(realization)
    mps = MaterialPointSimulator(jobid)
    nu, E, K, G, LAM = gen_rand_elastic_params()
    A1, A4 = gen_rand_surface_params(K, G)
    parameters = {'K': K, 'G': G, 'A1': A1, 'A4': A4}
    material = PlasticMaterial(**parameters)
    mps.assign_material(material)
    strain = gen_strain_table(K, G, A1, A4)
    analytic = gen_analytic_solution(K, G, A1, A4, strain)
    for row in strain[1:]:
        mps.add_step('E', row, increment=1.0, frames=25)
    mps.run()
    simulation = mps.get2(*myvars)
    assert responses_are_same(jobid, analytic, simulation, myvars)

def gen_rand_elastic_params():
    # poisson_ratio and young's modulus
    nu = random.uniform(-1.0 + 1.0e-5, 0.5 - 1.0e-5)
    E = max(1.0, 10 ** random.uniform(0.0, 12.0))

    # K and G are used for parameterization
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # LAM is used for computation
    LAM = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return nu, E, K, G, LAM


def gen_rand_surface_params(K, G):
    A1 = 2.0 * G * random.uniform(0.0001, 0.5)
    A4 = np.tan(random.uniform(np.pi / 100.0, np.pi / 2.1))
    return A1, A4


def gen_strain_table(K, G, A1, A4):
    strain = gen_unit_strain(A1, A4)
    gamma = mag(dev(strain))

    # Figure out how much strain we'll need to achieve yield
    fac = A1 / (3. * K * A4 * np.sum(strain[:3]) + sqrt(2.) * G * gamma)
    strain = fac * strain

    strain_table = np.zeros((3, 6))
    strain_table[1] = strain
    strain_table[2] = 2.0 * strain
    return strain_table


def gen_unit_strain(A1, A4):
    # Generate a random strain deviator
    straingen = lambda: random.uniform(-1, 1)
    #devstrain = np.array([straingen(), straingen(), straingen(),
    #                      straingen(), straingen(), straingen()])
    devstrain = np.array([straingen(), straingen(), straingen(),
                          0., 0., 0.])
    while mag(dev(devstrain)) == 0.0:
        devstrain = np.array([straingen(), straingen(), straingen(),
                              0., 0., 0.])

    snorm = unit(dev(devstrain))
    return (sqrt(2.0) * A4 * I6 + snorm) / sqrt(6.0 * A4 ** 2 + 1.0)


def gen_analytic_solution(K, G, A1, A4, strain):

    flatten = lambda arg: [x for y in arg for x in y]

    stress = np.zeros((3, 6))
    stress[1] = 3.0 * K * iso(strain[1]) + 2.0 * G * dev(strain[1])
    # Stress stays constant while strain increases
    stress[2] = stress[1]

    state = []
    for i in range(3):
        state.append(flatten([[i], strain[i], stress[i]]))
    state = np.array(state)
    return state[:, [0, 1, 2, 3, 7, 8, 9]]


def iso(A):
    return np.sum(A[:3]) / 3.0 * I6


def dev(A):
    return A - iso(A)


def mag(A):
    return sqrt(np.dot(A[:3], A[:3]) + 2.0 * np.dot(A[3:], A[3:]))


def unit(A):
    return A / mag(A)


def gen_spherical_everything(K, MU, RINT, ZINT):
    # Shear strain
    ES = RINT / (2.0 * sqrt(2.0) * MU)

    # Spherical (volumetric) strain
    RNUM = 3.0 * K ** 2 * (RINT / ZINT) ** 2
    DNOM = 3.0 * K * (RINT / ZINT) ** 2 + 2.0 * MU
    TREPS = ZINT / (sqrt(3.0) * K - sqrt(3.0) * RNUM / DNOM)
    EV = TREPS / 3.0

    ##### Stress State
    MAX_SHEAR_STRESS = 2.0 * MU * ES
    MAX_HYDRO_STRESS = ZINT / sqrt(3.0)

    pathtable = [[0.0, 0.0, 0.0,  ES, 0.0, 0.0],
                 [ EV,  EV,  EV,  ES, 0.0, 0.0]]

    head = (["Time"] + ["E." + _ for _ in ["XX","YY","ZZ","XY","XZ","YZ"]]
                     + ["S." + _ for _ in ["XX","YY","ZZ","XY","XZ","YZ"]]
                     + ["N." + _ for _ in ["XX","YY","ZZ","XY","XZ","YZ"]]
                     + ["P." + _ for _ in ["XX","YY","ZZ","XY","XZ","YZ"]])
    data = []
    for t in np.linspace(0.0, 2.0, 100*2 + 1):
        exx = 0.0 if t < 1.0 else EV * (t - 1.0)
        exy = ES * min(1.0, t)
        sxx = 0.0 if t < 1.0 else MAX_HYDRO_STRESS * (t - 1.0)
        sxy = MAX_SHEAR_STRESS * t if t < 1.0 else MAX_SHEAR_STRESS * (2.0 - t)

        if t <= 1.0:
            nxx = 0.0
            nxy = 0.0
            pxx = 0.0
            pxy = 0.0
        else:
            fac = 1.0 / np.sqrt((RINT/ZINT)**2 + 1.0)
            nxx = fac * RINT / ZINT / np.sqrt(3.0)
            nxy = fac * 1.0 / np.sqrt(2.0)
            pxx = 3.0 * K * nxx
            pxy = 2.0 * MU * nxy

        data.append([t, exx, exx, exx, exy, 0.0, 0.0,
                        sxx, sxx, sxx, sxy, 0.0, 0.0,
                        nxx, nxx, nxx, nxy, 0.0, 0.0,
                        pxx, pxx, pxx, pxy, 0.0, 0.0,
                    ])

    if 0:
        with open("dum.dat", "w") as F:
            F.write("".join(["{0:>20s}".format(_) for _ in head])+"\n")
            for leg in data:
                F.write("".join(["{0:20.10e}".format(_) for _ in leg])+"\n")
    return head, data


def gen_spherical_path(K, MU, RINT, ZINT):
    # Shear strain
    ES = RINT / (2.0 * sqrt(2.0) * MU)

    # Spherical (volumetric) strain
    RNUM = 3.0 * K ** 2 * (RINT / ZINT) ** 2
    DNOM = 3.0 * K * (RINT / ZINT) ** 2 + 2.0 * MU
    TREPS = ZINT / (sqrt(3.0) * K - sqrt(3.0) * RNUM / DNOM)
    EV = TREPS / 3.0

    ##### Stress State
    MAX_SHEAR_STRESS = 2.0 * MU * ES
    MAX_HYDRO_STRESS = ZINT / sqrt(3.0)

    pathtable = [[0.0, 0.0, 0.0,  ES, 0.0, 0.0],
                 [ EV,  EV,  EV,  ES, 0.0, 0.0]]

    return pathtable
