import pytest
import numpy as np
from numpy.testing import assert_allclose
from sbfl.base import SBFL
from sbfl.diagnosability import *

def X_y_sample_1():
    X = np.array([
        [1,0,1],
        [0,0,1],
        [1,1,0]
    ], dtype=bool)
    y = np.array([1,0,1], dtype=bool)
    return X, y

def test_diversity():
    X, y = X_y_sample_1()
    assert diversity(X) == 1.0

def test_uniqueness():
    X, y = X_y_sample_1()
    assert uniqueness(X) == 1.0

def test_density():
    X, y = X_y_sample_1()
    assert_allclose(density(X), 5/9)