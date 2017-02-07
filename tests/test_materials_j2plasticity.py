import os
import glob
import pytest
import random
import numpy as np

from matmodlab2 import *
from testing_utils import *

@pytest.mark.pandas
@pytest.mark.random
@pytest.mark.material
@pytest.mark.j2plasticity
@pytest.mark.parametrize('realization', range(1,4))
def test_random_j2_1(realization):
    myvars = ('Time',
                'E.XX', 'E.YY', 'E.ZZ', 'E.XY', 'E.YZ', 'E.XZ',
                'S.XX', 'S.YY', 'S.ZZ', 'S.XY', 'S.YZ', 'S.XZ')
    jobid = 'test_random_j2_1_{0}'.format(realization)
    mps = MaterialPointSimulator(jobid)
    NU, E, K, G, LAM, Y_SHEAR = gen_rand_params()
    parameters = {'K': K, 'G': G, 'A1': Y_SHEAR}
    material = PlasticMaterial(**parameters)
    mps.assign_material(material)
    analytic = gen_rand_analytical_resp_1(LAM, G, Y_SHEAR)
    for (i, row) in enumerate(analytic[1:], start=1):
        incr = analytic[i, 0] - analytic[i-1, 0]
        mps.run_step('E', row[1:7], increment=incr, frames=100)
    simulation = mps.get2(*myvars)
    assert responses_are_same(jobid, analytic, simulation, myvars)

@pytest.mark.random
@pytest.mark.material
@pytest.mark.j2plasticity
@pytest.mark.parametrize('realization', range(1,4))
def test_random_j2_2(realization):

    jobid = 'test_random_j2_2_{0}'.format(realization)
    mps = MaterialPointSimulator(jobid)

    # set up the model
    myvars = ('Time', 'E.XX', 'E.YY', 'E.ZZ', 'S.XX', 'S.YY', 'S.ZZ')

    # Set up the path and random material constants
    K = 150.0e9
    G = 100.0e9
    Y0 = 40.0e6

    LAM = K - 2.0 * G / 3.0
    E = 9.0 * K * G / (3.0 * K + G)
    NU = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)

    # set up the material
    parameters = {'K': K, 'G': G, 'Y0': Y0, 'H': 0., 'BETA': 0.}
    material = VonMisesMaterial(**parameters)
    mps.assign_material(material)

    # get the path and analytical solution
    pathtable, analytic = gen_rand_analytic_resp_2(NU, E, K, G, LAM, Y0)

    # Write the analytic solution for visualization comparison
    for row in pathtable:
        mps.run_step('E', row, increment=1., frames=100)

    # check output with analytic
    simulation = mps.get2(*myvars)
    assert responses_are_same(jobid, analytic, simulation, myvars)

@pytest.mark.pandas
@pytest.mark.material
@pytest.mark.j2plasticity
def test_j2_1():
    jobid = 'j2_plast'
    mps = MaterialPointSimulator(jobid)
    NU, E, K, G, LAM, Y = copper_params()
    YF, H, BETA = Y, 0, 0
    parameters = {'K': K, 'G': G, 'Y0': YF, 'H': H, 'BETA': 0}
    material = VonMisesMaterial(**parameters)
    mps.assign_material(material)
    path = gen_uniax_strain_path(Y, YF, G, LAM)
    for row in path:
        mps.run_step('E', row, increment=1.0, frames=25)
    assert same_as_baseline(mps.jobid, mps.df, interp=1)

@pytest.mark.pandas
@pytest.mark.material
@pytest.mark.j2plasticity
@pytest.mark.isotropic_hardening
def test_j2_iso_hard():
    jobid = 'j2_plast_iso_hard'
    mps = MaterialPointSimulator(jobid)
    NU, E, K, G, LAM, Y = copper_params()
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF, BETA = Y * (1.0 + HFAC), 0
    parameters = {'K': K, 'G': G, 'Y0': Y, 'H': H, 'BETA': BETA}
    material = VonMisesMaterial(**parameters)
    mps.assign_material(material)
    path = gen_uniax_strain_path(Y, YF, G, LAM)
    for row in path:
        mps.run_step('E', row, increment=1.0, frames=25)
    assert same_as_baseline(mps.jobid, mps.df, interp=1)

@pytest.mark.pandas
@pytest.mark.material
@pytest.mark.j2plasticity
@pytest.mark.kinematic_hardening
def test_j2_kin_hard():
    jobid = 'j2_plast_kin_hard'
    mps = MaterialPointSimulator(jobid)
    NU, E, K, G, LAM, Y = copper_params()
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF = Y * (1.0 + HFAC)
    BETA = 1.0
    parameters = {'K': K, 'G': G, 'Y0': Y, 'H': H, 'BETA': BETA}
    material = VonMisesMaterial(**parameters)
    mps.assign_material(material)
    path = gen_uniax_strain_path(Y, YF, G, LAM)
    for row in path:
        mps.run_step('E', row, increment=1.0, frames=25)
    assert same_as_baseline(mps.jobid, mps.df, interp=1)

@pytest.mark.pandas
@pytest.mark.material
@pytest.mark.j2plasticity
@pytest.mark.mixed_hardening
def test_j2_mix_hard():
    jobid = 'j2_plast_mix_hard'
    mps = MaterialPointSimulator(jobid)
    NU, E, K, G, LAM, Y = copper_params()
    HFAC = 1.0 / 10.0
    H = 3.0 * HFAC / (1.0 - HFAC) * G
    YF, BETA = Y * (1.0 + HFAC), .5
    parameters = {'K': K, 'G': G, 'Y0': Y, 'H': H, 'BETA': BETA}
    material = VonMisesMaterial(**parameters)
    mps.assign_material(material)
    path = gen_uniax_strain_path(Y, YF, G, LAM)
    for row in path:
        mps.run_step('E', row, increment=1.0, frames=25)
    assert same_as_baseline(mps.jobid, mps.df, interp=1)

def gen_rand_params():
    # poisson_ratio and young's modulus
    nu = random.uniform(-1.0 + 1.0e-5, 0.5 - 1.0e-5)
    E = max(1.0, 10 ** random.uniform(0.0, 12.0))

    # K and G are used for parameterization
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))

    # LAM is used for computation
    LAM = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Compute a realistic shear yield stress for our parameters
    Y_SHEAR = 2.0 * G * random.uniform(1.0e-6, 1.0e-1)

    return nu, E, K, G, LAM, Y_SHEAR

def gen_rand_analytical_resp_1(LAM, G, Y_SHEAR, test_type="PRINCIPAL"):
    # This function randomly comes up with a strain path
    # to do a simple test of von mises plasticity. Yield is achieved
    # right when T=1 (for simple debugging).

    # Populate the stiffness tensor
    stiff = (LAM * np.outer(np.array([1,1,1,0,0,0]), np.array([1,1,1,0,0,0])) +
             2.0 * G * np.identity(6))

    rnd = lambda: random.uniform(-0.01, 0.01)
    if test_type == "FULL":
        strain = np.array([rnd(), rnd(), rnd(), rnd(), rnd(), rnd()])
    elif test_type == "PRINCIPAL":
        strain = np.array([rnd(), rnd(), rnd(), 0.0, 0.0, 0.0])
    elif test_type == "UNIAXIAL":
        strain = np.array([rnd(), 0.0, 0.0, 0.0, 0.0, 0.0])
    elif test_type == "BIAXIAL":
        tmp = rnd()
        strain = np.array([tmp, tmp, 0.0, 0.0, 0.0, 0.0])

    strain_iso = strain[:3].sum() / 3.0 * np.array([1, 1, 1, 0, 0, 0])
    strain_dev = strain - strain_iso
    strain_dev_mag = np.sqrt(2.0 * np.dot(strain_dev, strain_dev) -
                             np.dot(strain_dev[:3], strain_dev[:3]))

    mag_fac = Y_SHEAR / (np.sqrt(2.0) * G * strain_dev_mag)
    strain1 = mag_fac * strain_iso + mag_fac * strain_dev
    strain2 = 2.0 * mag_fac * strain_iso + mag_fac * strain_dev

    table = np.vstack((np.hstack(([0.0], np.zeros(6), np.zeros(6))),
                       np.hstack(([1.0], strain1, np.dot(stiff, strain1))),
                       np.hstack(([2.0], 2.0 * strain1, np.dot(stiff, strain2)))))

    # returns a tablewith each row comprised of
    # time=table[0], strains=table[1:7], stresses=table[7:]
    return table

def gen_rand_analytic_resp_2(nu, E, K, G, LAM, Y0, disp=0):
    # Here we generate a random unit vector that will not be too close to
    # the hydrostat. This is accomplished by generating a random unit vector
    # in 3D space but require it to be at least theta_cutoff degrees from
    # the z-axis. We then rotate the vectors such that the old z-axis is now
    # the hydrostatic axis.
    #
    # We then compute the required magnitude of this strain so that the yield
    # surface is hit exactly halfway through the first leg.
    e0 = np.zeros(3)

    # random generation
    theta_cutoff = np.radians(30.0)
    theta_range = np.cos(theta_cutoff)
    theta = np.arccos(theta_range * (1.0 - 2.0 * random.random()))
    phi = 2.0 * np.pi * random.random()
    e1 = np.array([np.sin(theta) * np.cos(phi),
                   np.sin(theta) * np.sin(phi),
                   np.cos(theta)])

    # rotation
    e1 = np.dot(rotation_matrix(np.array([-1.0, 1.0, 0.0]),
                                np.arccos(1.0 / np.sqrt(3.0))), e1)

    # scale for desired intersection with yield surface
    e1dev = e1 - e1.sum() / 3.0 * np.ones(3)
    fac = 2.0 * (Y0 / np.sqrt(12.0 * G ** 2 * np.dot(e1dev, e1dev) / 2.0))
    e1 = fac * e1
    e1vol = e1.sum() / 3.0 * np.ones(3)
    e1dev = e1 - e1vol

    # e2 is just a rotation (of random magnitude not greater than 90degrees) of
    # the deviatoric part of e1 around the hydrostat while maintaining e1's
    # hydrostatic strain rate.
    e2_rot = np.radians(random.uniform(-60.0, 60.0))
    e2 = np.dot(rotation_matrix(np.ones(3), e2_rot), e1dev) #+ e1vol

    # generate the simulation steps
    pathtable = [[e1[0], e1[1], e1[2]],
                 [e1[0]+e2[0], e1[1]+e2[1], e1[2]+e2[2]]]

    # analytic solution
    analytic_solution = []
    for t in np.linspace(0, 2, 1000+1):
        if 0.0 <= t <= 1.0:
            fac = t
            cur_strain = (1.0 - fac) * e0 + fac * e1
        else:
            fac = t - 1.0
            cur_strain = (1.0 - fac) * e1 + fac * (e1 + e2)
        tmp = get_stress2(K, G, Y0, e1, e2, t)
        analytic_solution.append([t, cur_strain[0], cur_strain[1], cur_strain[2],
                                  tmp[0], tmp[1], tmp[2]])
    analytic_solution = np.array(analytic_solution)

    return pathtable, analytic_solution

def get_stress1(e11, e22, e33, e12, e23, e13, LAM, G, rootj2lim):
    YIELDING = False
    #standard hooke's law
    sig11 = (2.0 * G + LAM) * e11 + LAM * (e22 + e33)
    sig22 = (2.0 * G + LAM) * e22 + LAM * (e11 + e33)
    sig33 = (2.0 * G + LAM) * e33 + LAM * (e11 + e22)
    sig12 = 2.0 * G * e12
    sig23 = 2.0 * G * e23
    sig13 = 2.0 * G * e13

    rootj2 = get_rootj2(sig11, sig22, sig33, sig12, sig23, sig13)
    if rootj2 > rootj2lim:
        YIELDING = True
        sigmean = (sig11 + sig22 + sig33) / 3.0
        s11 = sig11 - sigmean
        s22 = sig22 - sigmean
        s33 = sig33 - sigmean
        dev_mag = np.sqrt(sum([_ ** 2 for _ in
                               [s11, s22, s33, sig12, sig23, sig13]]))
        fac = rootj2lim / dev_mag * np.sqrt(2.0)
        sig11 = sigmean + s11 * fac
        sig22 = sigmean + s22 * fac
        sig33 = sigmean + s33 * fac
        sig12 = sig12 * fac
        sig23 = sig23 * fac
        sig13 = sig13 * fac

    rootj2 = get_rootj2(sig11, sig22, sig33, sig12, sig23, sig13)

    return (sig11, sig22, sig33, sig12, sig23, sig13), YIELDING

def get_stress2(K, G, Y, e0, e1, t):
    e0vol = e0.sum() / 3.0 * np.ones(3)
    e0dev = e0 - e0vol

    e1vol = e1.sum() / 3.0 * np.ones(3)
    e1dev = e1 - e1vol

    # Y = sqrt(3 * J2)
    j2_of_e0 = np.dot(e0dev, e0dev) / 2.0
    final_q = 2.0 * G * np.sqrt(3.0 * j2_of_e0)

    fac = min(min(t, 1.0), Y / final_q)
    p0 = 3.0 * K * e0vol * min(t, 1.0)
    s0 = 2.0 * G * e0dev * fac
    if 0.0 <= t <= 1.0:
        return p0 + s0

    # R = sqrt(2 * J2) =  mag(s) at yield
    R = np.sqrt(2.0 / 3.0) * Y
    fac = t - 1.0

    ds = 2.0 * G * e1dev * fac
    cospsi = np.dot(s0, ds) / (np.linalg.norm(s0) * np.linalg.norm(ds))
    c = np.exp(-np.linalg.norm(ds) / R)
    beta = (1.0 - c ** 2 + (1.0 - c) ** 2 * cospsi) * R
    beta /= (2.0 * c * np.linalg.norm(ds))
    alpha = 2.0 * c / (1.0 + c ** 2 + (1.0 - c ** 2) * cospsi)

    p1 = p0 + 3.0 * K * e1vol * fac
    s1 =  alpha * (s0 + beta * ds)
    return p1 + s1

def copper_params():
    E   =   0.1100000E+12
    NU  =   0.3400000
    LAM =   0.87220149253731343283582089552238805970149253731343E+11
    G   =   0.41044776119402985074626865671641791044776119402985E+11
    K   =   0.11458333333333333333333333333333333333333333333333E+12
    USM =   0.16930970149253731343283582089552238805970149253731E+12
    Y   =   70.0E+6
    return NU, E, K, G, LAM, Y

def get_rootj2(sig11, sig22, sig33, sig12, sig23, sig13):
    rootj2 = ((sig11 - sig22) ** 2 +
              (sig22 - sig33) ** 2 +
              (sig33 - sig11) ** 2 +
              6.0 * (sig12 ** 2 + sig23 ** 2 + sig13 ** 2))
    return np.sqrt(rootj2 / 6.0)

def rotation_matrix(a, theta):
    ahat = a / np.linalg.norm(a)
    part1 = np.cos(theta) * np.eye(3)
    part2 = (1.0 - np.cos(theta)) * np.outer(ahat, ahat)
    part3 = np.sin(theta) * np.array([[0.0, -ahat[2], ahat[1]],
                                      [ahat[2], 0.0, -ahat[0]],
                                      [-ahat[1], ahat[0], 0.0]])
    return part1 + part2 + part3

def gen_uniax_strain_path(Y, YF, G, LAM):

    EPSY = Y / (2.0 * G) #  axial strain at yield
    SIGAY = (2.0 * G + LAM) * EPSY #  axial stress at yield
    SIGLY = LAM * EPSY #  lateral stress at yield

    EPSY2 = 2.0 * Y / (2.0 * G) #  final axial strain
    SIGAY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0 + YF # final axial stress
    SIGLY2 = ((2 * G + 3 * LAM) * EPSY2 - YF ) / 3.0 # final lateral stress

    return [[EPSY, 0.0, 0.0], [EPSY2, 0.0, 0.0]]
