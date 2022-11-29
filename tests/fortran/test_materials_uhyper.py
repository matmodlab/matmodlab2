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
from matmodlab2.ext_helpers import build_uhyper
from matmodlab2.umat.umats import neohooke_uhyper
from matmodlab2 import MaterialPointSimulator, UHyper
from matmodlab2.core.misc import working_dir

try:
    fc = os.getenv("FC", "gfortran")
    with open(os.devnull, "a") as fh:
        p = Popen([fc, "-v"], stdout=fh, stderr=STDOUT)
        p.wait()
    has_fortran = p.returncode == 0
except:
    has_fortran = False


@pytest.mark.skipif(not has_fortran, reason="Fortran compiler not found")
def test_uhyper_neohooke(tmpdir):
    """Test building a umat"""
    with working_dir(tmpdir):
        build_uhyper(neohooke_uhyper, verbose=True, cwd=".")
        if not len(glob.glob("_uhyper*.so")):
            assert 0, f"uhyper not found in {tmpdir}"
        if "_uhyper" in sys.modules:
            # Remove so it can be loaded below
            del sys.modules["_uhyper"]
        try:
            import _uhyper
        except ImportError:
            raise Exception("_uhyper not imported")
        assert hasattr(_uhyper, "sdvini")
        assert hasattr(_uhyper, "umat")

        # Now do the actual material test
        E = 500
        Nu = 0.45
        C1 = E / (4.0 * (1.0 + Nu))
        D1 = 1.0 / (6.0 * (1.0 - 2.0 * Nu) / E)

        X = 0.1
        mps = MaterialPointSimulator("UHyper")
        mps.material = UHyper([C1, 1.0 / D1])
        mps.run_step("ESS", (1, 0, 0), frames=10, scale=X)
        mps.run_step("ESS", (0, 0, 0), frames=10)

        V0 = ("E.XX", "E.YY", "E.ZZ", "S.XX", "S.YY", "S.ZZ", "F.XX", "F.YY", "F.ZZ")
        a = mps.get2(*V0)

        # make sure the strain table was interpoloated correctly
        i = np.argmax(a[:, 0])
        assert np.allclose(a[i, 0], X)

        # analytic solution for uniaxial stress

        J = np.prod(a[i, [6, 7, 8]])
        L = np.exp(a[i, 0])
        S = 2.0 * C1 / (J ** (5.0 / 3.0)) * (L**2 - J / L)
        assert np.allclose(a[i, 3], S)

        # analytic solution for J
        f = (
            lambda j: D1 * j ** (8.0 / 3.0)
            - D1 * j ** (5.0 / 3.0)
            + C1 / (3.0 * L) * J
            - C1 * L**2.0 / 3.0
        )
        df = (
            lambda j: 8.0 / 3.0 * D1 * j ** (5.0 / 3.0)
            - 5.0 / 3.0 * D1 * j ** (2.0 / 3.0)
            + C1 / (3.0 * L)
        )
        j = tu.newton(1.0, f, df)
        assert np.allclose(J, j)
