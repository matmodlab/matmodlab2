import os
import sys
import glob
import pytest
import testing_utils as tu
from subprocess import Popen, STDOUT
from matmodlab2.ext_helpers import build_umat
from matmodlab2.umat.umats import neohooke_umat
from matmodlab2.core.misc import working_dir

# pytestmark = pytest.mark.skipif('linux' in sys.platform.lower(),
#                                reason='Does not pass on linux')

# matmodlab.ext_helpers.build_ext defines the actual function that builds the
# extension modules. It adds some Matmodlab specific fortran I/O files (and
# some Abaqus specific files for umats) to the list of source files and then
# distutils to compile the thing. Instead of using this function directly, the
# command line tool build-fext is used. This is done since distutils can only
# be initialized once and we want to run several different tests, each building
# a different library.

try:
    fc = os.getenv("FC", "gfortran")
    with open(os.devnull, "a") as fh:
        p = Popen([fc, "-v"], stdout=fh, stderr=STDOUT)
        p.wait()
    has_fortran = p.returncode == 0
except:
    has_fortran = False


@pytest.mark.skipif(not has_fortran, reason="Fortran compiler not found")
def test_build_umat_ext(tmpdir):
    """Test building a umat"""

    with working_dir(tmpdir):
        returncode = build_umat(neohooke_umat, cwd=".")
        if not len(glob.glob("_umat*.so")):
            assert 0, f"umat not found in {tmpdir}"
        if "_umat" in sys.modules:
            # Remove so it can be loaded below
            del sys.modules["_umat"]
        try:
            import _umat
        except ImportError:
            raise Exception("_umat not imported")
        assert hasattr(_umat, "sdvini")
        assert hasattr(_umat, "umat")
