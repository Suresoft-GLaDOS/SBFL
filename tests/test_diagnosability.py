import pytest
import numpy as np
from numpy.testing import assert_allclose
from sbfl.base import SBFL
from sbfl.diagnosability import *

@pytest.fixture
def X_y_sample_1():
    X = np.array([
        [1,0,1],
        [0,0,1],
        [1,1,0]
    ], dtype=bool)
    y = np.array([1,0,1], dtype=bool)
    return X, y

def test_diversity(X_y_sample_1):
    X, y = X_y_sample_1
    assert diversity(X) == 1.0

def test_uniqueness(X_y_sample_1):
    X, y = X_y_sample_1
    assert uniqueness(X) == 1.0

def test_density(X_y_sample_1):
    X, y = X_y_sample_1
    assert_allclose(density(X), 5/9)

def test_normalized_density(X_y_sample_1):
    X, y = X_y_sample_1
    assert_allclose(norm_density(X), 8/9)

def test_compare_densities():
    X = np.array([
        [1,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        [1,1,0,0]
    ], dtype=int)

    assert_allclose(density(X), 0.5)
    assert_allclose(norm_density(X), 1.0)

    X = np.array([
        [1,1,0,0],
        [0,0,1,1],
        [1,1,1,0],
        [0,0,0,1]
    ], dtype=int)

    assert_allclose(density(X), 0.5)
    assert_allclose(norm_density(X), 1.0)

    X = np.array([
        [1,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]
    ], dtype=int)

    assert_allclose(density(X), 1/16)
    assert_allclose(norm_density(X), 1/8)

    X = np.array([
        [0,1,1,1],
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1]
    ], dtype=int)

    assert_allclose(density(X), 15/16)
    assert_allclose(norm_density(X), 1/8)
