# -*- coding: utf-8 -*-
"""
This file contains tests for misc.py
"""
import sys
import pytest
import numpy as np

try:
    import matmodlab2
except ImportError:
    matmodlab = None

import matmodlab2.core.misc as misc

def test_is_listlike():
    """Is item list like?"""
    assert not misc.is_listlike('aaa')
    assert misc.is_listlike([0,1,2])
    assert misc.is_listlike((0,1,2))
    assert not misc.is_listlike(None)

def test_is_stringlike():
    """Is item string like?"""
    assert misc.is_stringlike('aaa')
    assert not misc.is_stringlike([0,1,2])
    assert not misc.is_stringlike((0,1,2))

def test_is_scalarlike():
    """Is item scalar like?"""
    assert misc.is_scalarlike(5)
    assert misc.is_scalarlike(5.)
    assert misc.is_scalarlike(np.array(5.))
    assert misc.is_scalarlike(np.array(5))
    assert not misc.is_scalarlike([1,2])
    assert not misc.is_scalarlike(np.array([1,2]))

if __name__ == '__main__':
    test_import()
