"""Test the Matmodlab2 umat interface. The interface to the fortran UMAT
procedure is identical to the Abaqus interface (albeit, written in free-form
fortran).
"""
import os
import sys
import glob
import pytest
import numpy as np
import testing_utils as tu
from subprocess import Popen, STDOUT
from matmodlab2.ext_helpers import build_extension_module_as_subprocess
from matmodlab2 import MaterialPointSimulator, UHyper

try:
    fc = os.getenv('FC', 'gfortran')
    with open(os.devnull, 'a') as fh:
        p = Popen([fc, '-v'], stdout=fh, stderr=STDOUT)
        p.wait()
    has_fortran = p.returncode == 0
except:
    has_fortran = False

def teardown_module():
    tu.teardown_module()
    for filename in glob.glob('_uhyper*.so'):
        os.remove(filename)

def build_extension_module(name, sources, user_ics=False):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    return build_extension_module_as_subprocess(name, sources,
                                                user_ics=user_ics,
                                                verbose=True, cwd=this_dir)

@pytest.mark.slow
@pytest.mark.uhyper
@pytest.mark.fortran
@pytest.mark.material
@pytest.mark.skipif(not has_fortran, reason='Fortran compiler not found')
def test_uhyper_neohooke():
    """Test building a umat"""

    name = 'uhyper'
    sources = ['../../matmodlab2/umat/umats/uhyper_neohooke.f90']
    assert os.path.isfile(sources[0])
    returncode = build_extension_module(name, sources)
    assert returncode == 0, 'uhyper not built'
    assert len(glob.glob('_uhyper*.so')), 'uhyper not built'
    if '_uhyper' in sys.modules:
        # Remove so it can be loaded below
        del sys.modules['_uhyper']
    try:
        import _uhyper
    except ImportError:
        raise Exception('_uhyper not imported')
    assert hasattr(_uhyper, 'sdvini')
    assert hasattr(_uhyper, 'umat')

    # Now do the actual material test
    E = 500
    Nu = .45
    C1 = E / (4. * (1. + Nu))
    D1 = 1. / (6. * (1. - 2. * Nu) / E)

    X = .1
    mps = MaterialPointSimulator('UHyper')
    mps.material = UHyper([C1, 1./D1])
    mps.run_step('ESS', (1,0,0), frames=10, scale=X)
    mps.run_step('ESS', (0,0,0), frames=10)

    V0 = ('E.XX', 'E.YY', 'E.ZZ',
          'S.XX', 'S.YY', 'S.ZZ',
          'F.XX', 'F.YY', 'F.ZZ')
    a = mps.get2(*V0)

    # make sure the strain table was interpoloated correctly
    i = np.argmax(a[:,0])
    assert np.allclose(a[i,0], X)

    # analytic solution for uniaxial stress

    J = np.prod(a[i, [6,7,8]])
    L = np.exp(a[i,0])
    S = 2. * C1 / (J ** (5. / 3.)) * (L ** 2 - J / L)
    assert np.allclose(a[i,3], S)

    # analytic solution for J
    f = lambda j: D1*j**(8./3.) - D1*j**(5./3.) + C1/(3.*L)*J - C1*L**2./3.
    df = lambda j: 8./3.*D1*j**(5./3.) - 5./3.*D1*j**(2./3.) + C1/(3.*L)
    j = tu.newton(1., f, df)
    assert np.allclose(J, j)
