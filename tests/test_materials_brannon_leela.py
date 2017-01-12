""" The acronym BL stands for 'Brannon, Leelavanichkul' with reference to
    the paper 'A multi-stage return algorithm for solving the classical damage
    component of constitutive models for rocks, ceramics, and other rock-like
    media', published on 25 September 2009 """
import os
import glob
import pytest
import numpy as np

from matmodlab2 import *
from testing_utils import *

@pytest.mark.material
@pytest.mark.analytic
@pytest.mark.multi_stage
def test_brannon_lellavanichkul_2():
    '''This function generates the analytical solution for example #2
    in the paper.

    The material parameters are set as:
        Yield in shear  = 165 MPa
        shear modulus   = 79 GPa
        poisson's ratio = 1/3  (not technically needed for simulation)

    '''
    jobid = 'brannon_leelavanichkul_2'
    mps = MaterialPointSimulator(jobid)

    # Yield in shear, shear modulus, and poisson_ratio
    Y = 165.0e6
    G = 79.0e9
    nu = 1.0 / 3.0

    # K and G are used for parameterization
    K = 2.0 * G * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))

    # Convert Y from 'yield in shear' to 'yield in tension'
    Y = Y * np.sqrt(3.0)

    # generate path
    #               time   e11         e22    e33
    strain_table = [[0.0,  0.0,        0.0,   0.0],
                    [1.0, -0.003,     -0.003, 0.006],
                    [2.0, -0.0103923,  0.0,   0.0103923]]

    N = 500  # number of interpolation points
    expanded = [[_] for _ in strain_table[0]]
    analytic_response = []
    for idx in range(0, len(strain_table) - 1):
        for jdx in range(0, len(strain_table[0])):
            start = strain_table[idx][jdx]
            end = strain_table[idx + 1][jdx]
            expanded[jdx] = expanded[jdx] + list(np.linspace(start, end, N))[1:]

    for idx in range(0, len(expanded[0])):
        t = expanded[0][idx]
        e1 = expanded[1][idx]
        e2 = expanded[2][idx]
        e3 = expanded[3][idx]
        sig = get_stress_2(t)
        sig11, sig22, sig33, sig12, sig23, sig13 = sig
        analytic_response.append([t, e1, e2, e3, sig11, sig22, sig33])
    analytic_response = np.array(analytic_response)

    # set up the material
    parameters = {'K': K, 'G': G, 'Y0': Y}
    material = VonMisesMaterial(**parameters)
    mps.assign_material(material)

    # set up the driver
    for idx, leg in enumerate(analytic_response):
        if idx == 0:
            continue
        incr = analytic_response[idx,0] - analytic_response[idx-1,0]
        mps.add_step('E', leg[1:4], increment=incr, frames=1)

    # run the model
    mps.run()

    # check output with analytic
    variables = ['Time',
                 'E.XX', 'E.YY', 'E.ZZ', 'S.XX', 'S.YY', 'S.ZZ']
    simulate_response = mps.get2(*variables)
    assert responses_are_same(jobid, analytic_response, simulate_response,
                              variables)

def get_stress_2(t):
    # this function evaluates equations #59, #60, and #61
    # !!!! Be Aware !!!
    # In equation #60 in the part 'if 1 < t <= 2' a minus
    #   sign should preceed the entire fraction.
    # In equation #61, also in the part 'if 1 < t <= 2'
    #   the last minus sign in the numerator should be a
    #   plus sign.
    if 0.0 <= t <= 0.20097:
        sig11 = -474.0 * t
        sig22 = -474.0 * t
        sig33 = 948.0 * t
    elif 0.20097 < t <= 1.0:
        sig11 = -95.26
        sig22 = -95.26
        sig33 = 190.5
    elif 1.0 < t <= 2.0:
        a = np.exp(12.33 * t)
        denom = 1.0 + 0.00001712 * a
        sig11 = (189.4 + 0.1704 * np.sqrt(a) - 0.003242 * a) / denom
        sig22 = -(76.87 + 1.443 * np.sqrt(a) - 0.001316 * a) / denom
        sig33 = (-112.5 + 1.272 * np.sqrt(a) + 0.001926 * a) / denom
    elif 2.0 < t:
        sig11 = 189.4
        sig22 = 76.87
        sig33 = 112.5
    else:
        raise Exception('Time was negative: t={0:.6e}'.format(t))

    return sig11 * 1.0e6, sig22 * 1.0e6, sig33 * 1.0e6, 0.0, 0.0, 0.0
