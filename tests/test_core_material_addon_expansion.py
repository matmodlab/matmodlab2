import pytest

import numpy as np
from matmodlab2 import *
from testing_utils import *

@pytest.mark.material
@pytest.mark.expansion_model
@pytest.mark.parametrize('itemp', np.random.uniform(0., 100., 1))
@pytest.mark.parametrize('dtemp', np.random.uniform(10., 50., 1))
@pytest.mark.parametrize('alpha', np.random.uniform(1e-5, 1e-4, 1))
def test_expansion(itemp, dtemp, alpha):
    temp = (itemp, itemp+dtemp)
    E, Nu = 500, .45
    mps = MaterialPointSimulator('Job-1', initial_temp=temp[0])
    mps.material = ElasticMaterial(E=E, Nu=Nu, expansion_model=alpha)
    mps.run_step('ESS', (.01,0,0), temperature=temp[0], frames=20)
    mps.run_step(('DE','DE','DE'), (0,0,0), temperature=temp[1], frames=20)

    out = mps.get2('Temp', 'E.XX', 'S.XX', 'S.YY', 'S.ZZ', 'EM.XX')

    errors = []
    for (i, row) in enumerate(out[1:], start=1):
        dtemp = out[i,0] - temp[0]
        ee = out[i,1] - alpha * dtemp
        assert abs(ee-out[i,-1]) < 1e-6
        diff = E * ee + Nu * (out[i,3] + out[i,4]) - out[i,2]
        errors.append(abs(diff/out[i,2]))

    if any([abs(x) > 1e-10 for x in errors]):
        raise Exception('maximum error = {0}'.format(max(errors)))

    return

if __name__ == '__main__':
    test_expansion(75, 25, 1e-5)
