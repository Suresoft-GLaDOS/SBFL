import pytest
import numpy as np
from numpy.testing import assert_allclose
from sbfl.base import SBFL
from sbfl.diagnosability import *

@pytest.fixture
def X_sample_1():
    X = np.array([
        [1,0,1],
        [0,0,1],
        [1,1,0]
    ], dtype=bool)
    return X

@pytest.fixture
def X_sample_2():
    X = np.array([
        [1,0,1],
    ], dtype=bool)
    return X

@pytest.fixture
def X_sample_3():
    X = np.array([
        [1,0,1],
        [0,0,1],
        [1,1,0],
        [1,1,0],
        [1,1,0]
    ], dtype=bool)
    return X

@pytest.fixture
def X_sample_4():
    X = np.array([
        [1,1,0],
        [1,1,0],
        [1,1,0]
    ], dtype=bool)
    return X

def test_diversity(X_sample_1):
    assert diversity(X_sample_1) == 1.0

def test_diversity_with_one_row_coverage_matrix(X_sample_2):
    assert diversity(X_sample_2) == 1.0

def test_diversity_with_less_diverse_coverage_matrix(X_sample_3):
    assert diversity(X_sample_3) == 0.7

def test_diversity_with_redundant_tests(X_sample_4):
    assert diversity(X_sample_4) == 0.0

def test_uniqueness(X_sample_1):
    assert uniqueness(X_sample_1) == 1.0

def test_uniqueness_with_one_row_coverage_matrix(X_sample_2):
    assert_allclose(uniqueness(X_sample_2), 2/3)

def test_uniqueness_with_less_diverse_coverage_matrix(X_sample_3):
    assert uniqueness(X_sample_3) == 1.0

def test_density(X_sample_1):
    assert_allclose(density(X_sample_1), 5/9)

def test_density_2(X_sample_2):
    assert_allclose(density(X_sample_2), 2/3)

def test_density_3(X_sample_3):
    assert_allclose(density(X_sample_3), 9/15)

def test_normalized_density(X_sample_1):
    assert_allclose(norm_density(X_sample_1), 8/9)

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

def test_DDU(X_sample_1):
    assert_allclose(DDU(X_sample_1), 8/9)

def test_DDU_2(X_sample_2):
    assert_allclose(DDU(X_sample_2), 2/3 * 2/3)

def test_DDU_3(X_sample_3):
    assert_allclose(DDU(X_sample_3), 0.7 * 0.8)

def test_DDU_4(X_sample_4):
    assert_allclose(DDU(X_sample_4), 0)