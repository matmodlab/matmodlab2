# -*- coding: utf-8 -*-
"""
This file contains tests that focus on general set-up
of the python environment. They also check things that
probably don't need any checking.
"""

import sys
import pathlib


# Ensure that 'matmodlab' is imported from parent directory.
sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent))

try:
    import matmodlab2
except ImportError:
    matmodlab2 = None


def test_absolute_truth():
    """Ensure that the testing library is working."""
    # Setup

    # Test
    assert True

    # Teardown


def test_require_python3():
    """The module 'matmodlab' and these tests require  at least Python 3.0."""
    # Setup

    # Test
    assert sys.version_info > (3, 0) or sys.version_info > (2, 6)

    # Teardown


def test_import():
    """Ensure that 'matmodlab' is imported."""
    # Setup

    # Test
    assert matmodlab2 is not None

    # Teardown


def test_initialize():
    """Do something simple with 'matmodlab'."""
    # Setup

    # Test
    assert 'MaterialPointSimulator' in matmodlab2.__all__

    # Teardown

if __name__ == '__main__':
    test_import()
