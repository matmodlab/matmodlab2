import os
import glob
import pytest
import matmodlab2
from matmodlab2.ext_helpers import build_mml_matrix_functions

def pytest_configure():
    # Setup code, executed ahead of first test
    # Remove all fortran files
    matmodlab2_dir = os.path.dirname(matmodlab2.__file__)
    for d in ('core', 'ext_helpers'):
        dirname = os.path.join(matmodlab2_dir, d)
        assert os.path.isdir(dirname)
        for filename in glob.glob(os.path.join(dirname, '*.o')):
            os.remove(filename)
        for filename in glob.glob(os.path.join(dirname, '*.so')):
            os.remove(filename)
    returncode = build_mml_matrix_functions()
    assert returncode == 0, 'matfuncs not built'
