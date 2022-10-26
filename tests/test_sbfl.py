import pytest
import numpy as np
from sbfl.base import SBFL

@pytest.fixture
def X_y_sample_1():
    X = np.array([
        [1,0,1],
        [0,0,1],
        [1,1,0]
    ], dtype=bool)
    y = np.array([1,0,1], dtype=bool)
    # spectrum
    # - e_p = [2, 1, 1]
    # - n_p = [0, 1, 1]
    # - e_f = [0, 0, 1]
    # - n_f = [1, 1, 0]
    return X, y

"""
Test `get_spectrum`
"""
def test_get_spectrum(X_y_sample_1):
    X, y = X_y_sample_1

    loc = SBFL()
    e_p, n_p, e_f, n_f = loc.get_spectrum(X, y)

    num_passings = y.sum()
    num_failings = np.invert(y).sum()

    assert e_p[0] == 2 and e_p[1] == 1 and e_p[2] == 1
    assert e_f[0] == 0 and e_f[1] == 0 and e_f[2] == 1
    assert np.all(e_p + n_p == num_passings)
    assert np.all(e_f + n_f == num_failings)

"""
Test `ranks` with different tie breakers
"""
def test_minrank(X_y_sample_1):
    X, y = X_y_sample_1

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    min_ranks = ochiai.ranks(method='min')
    assert min_ranks[0] == 2
    assert min_ranks[1] == 2
    assert min_ranks[2] == 1

def test_avgrank(X_y_sample_1):
    X, y = X_y_sample_1

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    avg_ranks = ochiai.ranks(method='average')
    assert avg_ranks[0] == 2.5
    assert avg_ranks[1] == 2.5
    assert avg_ranks[2] == 1

def test_maxrank(X_y_sample_1):
    X, y = X_y_sample_1

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    max_ranks = ochiai.ranks(method='max')
    assert max_ranks[0] == 3
    assert max_ranks[1] == 3
    assert max_ranks[2] == 1

def test_denserank(X_y_sample_1):
    X, y = X_y_sample_1

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    dense_ranks = ochiai.ranks(method='dense')
    assert dense_ranks[0] == 2
    assert dense_ranks[1] == 2
    assert dense_ranks[2] == 1

def test_ordinalrank(X_y_sample_1):
    X, y = X_y_sample_1

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    ord_ranks = ochiai.ranks(method='ordinal')
    assert ord_ranks[0] == 2
    assert ord_ranks[1] == 3
    assert ord_ranks[2] == 1

"""
Test `to_frame`
"""
def test_to_frame_without_elements(X_y_sample_1):
    X, y = X_y_sample_1

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    df = ochiai.to_frame()
    scores = ochiai.scores_

    assert df.shape[0] == 3
    assert "score" in df.columns
    for i in range(df.shape[0]):
        assert df.index[i] == i
        assert df.values[i] == scores[i]

def test_to_frame_with_tuple_elements(X_y_sample_1):
    X, y = X_y_sample_1
    elements = [
        ('file1.py', 'method1'),
        ('file2.py', 'method2'),
        ('file2.py', 'method3')
    ]

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    df = ochiai.to_frame(elements=elements)
    scores = ochiai.scores_

    assert df.shape[0] == 3
    assert "score" in df.columns
    for i in range(df.shape[0]):
        assert df.index[i] == elements[i]
        assert df.values[i] == scores[i]

def test_to_frame_with_tuple_elements_and_names(X_y_sample_1):
    X, y = X_y_sample_1
    names = ['file', 'method']
    elements = [
        ('file1.py', 'method1'),
        ('file2.py', 'method2'),
        ('file2.py', 'method3')
    ]

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    df = ochiai.to_frame(elements=elements, names=names)
    scores = ochiai.scores_

    assert df.shape[0] == 3
    assert "score" in df.columns
    assert df.index.names[0] == names[0]
    assert df.index.names[1] == names[1]
    for i in range(df.shape[0]):
        assert df.index[i] == elements[i]
        assert df.values[i] == scores[i]

def test_to_frame_shape_error(X_y_sample_1):
    X, y = X_y_sample_1
    elements = [
        ('file1.py', 'method1'),
        ('file2.py', 'method2'),
    ]

    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)

    with pytest.raises(ValueError):
        ochiai.to_frame(elements=elements)