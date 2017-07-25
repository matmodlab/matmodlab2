from io import StringIO
from collections import OrderedDict
import numpy as np
import scipy.linalg as sl
import pandas as pd
import pytest
import matmodlab2 as mml


runid = 'simulation_output'

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

@pytest.mark.skip()
@pytest.mark.parametrize('stretch', [1.5, 1.001, 0.999, 0.5])
@pytest.mark.parametrize('kappa', [2.0, 1.0, 0.0, -1.0, 2.0])
def test_defgrad_basic(kappa, stretch):

    stretch1 = (stretch - 1.0) / 2.0 + 1.0
    stretch2 = stretch
    if kappa == 0.0:
        strain1 = np.log(stretch1)
        strain2 = np.log(stretch2)
    else:
        strain1 = (stretch1 ** kappa - 1.0) / kappa
        strain2 = (stretch2 ** kappa - 1.0) / kappa

    sol_io = StringIO("""Time,E.XX,E.YY,E.ZZ,F.XX,F.YY,F.ZZ
0.0,0.0,0.0,0.0,1.0,1.0,1.0
0.5,{0:.14e},0.0,0.0,{1:.14e},1.0,1.0
1.0,{2:.14e},0.0,0.0,{3:.14e},1.0,1.0""".format(strain1, stretch1, strain2, stretch2))
    sol_df = pd.read_csv(sol_io)
    sol_df.to_csv("defgrad_basic_solution.csv")

    # Initialize the simulator
    mps = mml.MaterialPointSimulator(runid)

    # Initialize the material
    mat = mml.ElasticMaterial(E=8.0, Nu=1.0 / 3.0)
    mps.assign_material(mat)

    # Run the steps
    mps.run_step('FFFFFFFFF', [stretch, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                     frames=20,
                     increment=1.0, kappa=kappa)

    print(sol_df)
    mps.df.to_csv("defgrad_basic.csv", index=False)
    assert compare_dataframes(sol_df, mps.df)


if __name__ == '__main__':
    test_defgrad_basic(0.0, 1.5)
