from io import StringIO
from collections import OrderedDict
import numpy as np
import scipy.linalg as sl
import pandas as pd
import pytest
import matmodlab2 as mml


runid = 'simulation_output'
solution_filename = runid + "_solution.csv"
solution_string = """Time,E.XX,E.YY,E.ZZ,E.XY,E.YZ,E.XZ
0.00, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
0.07, 0.020,-0.005,-0.010, 0.010,-0.015, 0.004
0.14, 0.040,-0.010,-0.020, 0.020,-0.030, 0.008
0.21, 0.060,-0.015,-0.030, 0.030,-0.045, 0.012
0.28, 0.080,-0.020,-0.040, 0.040,-0.060, 0.016
0.35, 0.100,-0.025,-0.050, 0.050,-0.075, 0.020
0.48, 0.080,-0.030,-0.025, 0.025,-0.075, 0.018
0.61, 0.060,-0.035, 0.000, 0.000,-0.075, 0.016
0.74, 0.040,-0.040, 0.025,-0.025,-0.075, 0.014
0.87, 0.020,-0.045, 0.050,-0.050,-0.075, 0.012
1.00, 0.000,-0.050, 0.075,-0.075,-0.075, 0.010
"""

# kappa = 0 (logarithmic strain)
db = OrderedDict((
      ["Step 1", OrderedDict((
                  ("dt", 0.35),
                  ("frames", 5),
                  ("strain",    np.array([0.1, -0.025, -0.05, 0.05, -0.075, 0.020])),
                  ("straininc", np.array([0.1, -0.025, -0.05, 0.05, -0.075, 0.020]) / 0.35),
                 ))
      ],
      ["Step 2", OrderedDict((
                  ("dt", 0.65),
                  ("frames", 5),
                  ("strain",    np.array([ 0.0, -0.050, 0.075, -0.075, -0.075,  0.010])),
                  ("straininc", np.array([-0.1, -0.025, 0.125, -0.125,  0.000, -0.010]) / 0.65),
                 ))
      ],
     ))

solution_io = StringIO(solution_string)

with open(solution_filename, 'w') as fd:
    fd.write(solution_string)

solution_pd = pd.read_csv(solution_io)


def compare_dataframes(frame1, frame2, tol=1.0e-12):
    head1 = frame1.keys()
    head2 = frame2.keys()
    passed = True
    for key in set(list(head1)) & set(list(head2)):
        print(key)
        arr1 = frame1[key]
        arr2 = frame2[key]
        if not np.allclose(arr1, arr2, atol=tol, rtol=tol):
            passed = False
            print("Column {0} failed".format(key))
            print(arr1 - arr2)
            print(sum(arr1 - arr2))
    return passed

# Show that it can correctly run this for
# * mps.run_step(strain)
# * mps.run_step(strain increment)
# * mps.run_step(mixed strain and strain rate)
# * mps.run_from_data(filename)
# * mps.run_from_data(StringIO)
# * mps.run_from_data(?string?)
# * mps.run_from_data(?array?)

def test_strain_evolution_strain():

    # Initialize the simulator
    mps = mml.MaterialPointSimulator(runid)

    # Initialize the material
    mat = mml.ElasticMaterial(E=8.0, Nu=1.0 / 3.0)
    mps.assign_material(mat)

    # Run the steps
    for step_data in db.values():
        mps.run_step('EEEEEE', step_data["strain"],
                     frames=step_data["frames"],
                     increment=step_data["dt"])

    mps.df.to_csv("strain_evolution_strain.csv", index=False)
    assert compare_dataframes(solution_pd, mps.df)


def test_strain_evolution_strain_increment():

    # Initialize the simulator
    mps = mml.MaterialPointSimulator(runid)

    # Initialize the material
    mat = mml.ElasticMaterial(E=8.0, Nu=1.0 / 3.0)
    mps.assign_material(mat)

    # Run the steps
    for step_data in db.values():
        mps.run_step(['DE', 'DE', 'DE', 'DE', 'DE', 'DE'],
                     step_data["straininc"],
                     frames=step_data["frames"],
                     increment=step_data["dt"])

    

    mps.df.to_csv("strain_evolution_straininc.csv", index=False)
    assert compare_dataframes(solution_pd, mps.df)


@pytest.mark.parametrize('kappa', [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_strain_evolution_defgrad(kappa):

    # Initialize the simulator
    mps = mml.MaterialPointSimulator(runid)

    # Initialize the material
    mat = mml.ElasticMaterial(E=8.0, Nu=1.0 / 3.0)
    mps.assign_material(mat)


    for idx in range(1, len(solution_pd["Time"])):
        dt = solution_pd["Time"][idx] - solution_pd["Time"][idx-1]
        get = lambda x: solution_pd[x][idx]
        strain = np.array([[get("E.XX"), get("E.XY"), get("E.XZ")],
                           [get("E.XY"), get("E.YY"), get("E.YZ")],
                           [get("E.XZ"), get("E.YZ"), get("E.ZZ")]])
        if kappa == 0.0:
            defgrad = sl.expm(strain)
        else:
            defgrad = sl.fractional_matrix_power(kappa * strain + np.eye(3), 1.0 / kappa)

        mps.run_step('FFFFFFFFF',
                     defgrad.flatten(),
                     frames=1,
                     increment=dt,
                     kappa=kappa)

    mps.df.to_csv("strain_evolution_defgrad.csv", index=False)
    assert compare_dataframes(solution_pd, mps.df, tol=1.0e-12)

if __name__ == '__main__':
    test_strain_evolution_defgrad()
