import os
import glob
import pytest
import random
import numpy as np

from matmodlab2 import *
from testing_utils import *


@pytest.mark.parametrize("realization", range(0, 10))
def test_stress_control(realization):

    jobid = "stress_control_{0:04d}".format(realization)

    failtol = 5.0e-7
    Nsteps = np.random.randint(20, 101)  # upper limit exclusive
    head = (
        "Time",
        "E.XX",
        "E.YY",
        "E.ZZ",
        "E.XY",
        "E.YZ",
        "E.XZ",
        "S.XX",
        "S.YY",
        "S.ZZ",
        "S.XY",
        "S.YZ",
        "S.XZ",
    )
    data = np.zeros((Nsteps, len(head)))

    #
    # Parameters
    #
    E = 10.0 ** np.random.uniform(0.0, 12.0)
    NU = np.random.uniform(-0.95, 0.45)
    K = E / 3.0 / (1.0 - 2.0 * NU)
    G = E / 2.0 / (1.0 + NU)
    params = {"E": E, "Nu": NU}

    #
    # Generate the path and analytical solution
    #
    eps = np.array(
        [2.0 * (np.random.rand() - 0.5) * np.random.randint(0, 2) for _ in range(6)]
    )
    eps_iso = (np.sum(eps[:3]) / 3.0) * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    eps_dev = eps - eps_iso
    sig = 3.0 * K * eps_iso + 2.0 * G * eps_dev

    for idx, t in enumerate(np.linspace(0.0, 1.0, Nsteps)):
        curr_eps = t * eps
        curr_sig = t * sig
        data[idx, 0] = t
        data[idx, 1:7] = curr_eps
        data[idx, 7:] = curr_sig

    #
    # Run the strain-controlled version
    #
    mps_eps = MaterialPointSimulator(jobid + "_eps")
    material = ElasticMaterial(**params)
    mps_eps.assign_material(material)
    for (i, row) in enumerate(data[1:]):
        increment = row[0] - data[i, 0]
        mps_eps.run_step("E", row[1:7], increment=increment)

    #
    # Run the stress-controlled version
    #
    mps_sig = MaterialPointSimulator(jobid + "_sig")
    material = ElasticMaterial(**params)
    mps_sig.assign_material(material)
    for (i, row) in enumerate(data[1:]):
        increment = row[0] - data[i, 0]
        mps_sig.run_step("S", row[7:], increment=increment)

    #
    # Analysis
    #
    data_eps = mps_eps.get2(*head)
    data_sig = mps_sig.get2(*head)
    assert data.shape == data_eps.shape and data.shape == data_sig.shape

    has_passed = True
    dt = data[1, 0] - data[0, 0]
    for idx, key in enumerate(head):
        gold = data[:, idx]
        eps = data_eps[:, idx]
        sig = data_sig[:, idx]

        basis = max(1.0, np.trapz(np.abs(gold), dx=dt))
        err_eps = np.trapz(np.abs(gold - eps), dx=dt) / basis
        err_sig = np.trapz(np.abs(gold - sig), dx=dt) / basis
        print("STRAIN ERROR: ", err_eps)
        print("STRESS ERROR: ", err_sig)
        if max(err_eps, err_sig) > failtol:
            has_passed = False

    assert has_passed
