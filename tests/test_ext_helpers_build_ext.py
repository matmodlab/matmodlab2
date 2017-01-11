import os
import sys
import glob
import pytest
import testing_utils as tu
from subprocess import Popen, STDOUT
from matmodlab.ext_helpers import build_extension_module_as_subprocess

#pytestmark = pytest.mark.skipif('linux' in sys.platform.lower(),
#                                reason='Does not pass on linux')

# matmodlab.ext_helpers.build_ext defines the actual function that builds the
# extension modules. It adds some Matmodlab specific fortran I/O files (and
# some Abaqus specific files for umats) to the list of source files and then
# distutils to compile the thing. Instead of using this function directly, the
# command line tool build-fext is used. This is done since distutils can only
# be initialized once and we want to run several different tests, each building
# a different library.

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
    for filename in glob.glob('_umat*.so'):
        os.remove(filename)

def build_extension_module(name, sources, user_ics=False):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    return build_extension_module_as_subprocess(name, sources,
                                                user_ics=user_ics,
                                                verbose=True, cwd=this_dir)

@pytest.mark.skipif(not has_fortran, reason='Fortran compiler not found')
def test_build_umat_ext_no_user_ics():
    """Test building a umat"""

    name = 'umat'
    sources = ['../matmodlab/umat/umats/umat_neohooke.f90']
    assert os.path.isfile(sources[0])
    returncode = build_extension_module(name, sources)
    assert returncode == 0, 'umat not built'
    assert len(glob.glob('_umat*.so')), 'umat not built'
    if '_umat' in sys.modules:
        # Remove so it can be loaded below
        del sys.modules['_umat']
    try:
        import _umat
    except ImportError:
        raise Exception('_umat not imported')
    assert hasattr(_umat, 'sdvini')
    assert hasattr(_umat, 'umat')

@pytest.mark.skipif(not has_fortran, reason='Fortran compiler not found')
def test_build_umat_ext_with_user_ics():
    """Test building a umat with user defined sdvini"""
    name = 'umat'
    sources = ['../matmodlab/umat/umats/umat_neohooke.f90',
               '../matmodlab/umat/aba_sdvini.f90']
    assert os.path.isfile(sources[0])
    assert os.path.isfile(sources[1])
    returncode = build_extension_module(name, sources, user_ics=True)
    assert returncode == 0, 'umat not built'
    assert len(glob.glob('_umat*.so')), 'umat not built'
    if '_umat' in sys.modules:
        # Remove so it can be loaded below
        del sys.modules['_umat']
    try:
        import _umat
    except ImportError:
        raise Exception('_umat not imported')
    assert hasattr(_umat, 'sdvini')
    assert hasattr(_umat, 'umat')
