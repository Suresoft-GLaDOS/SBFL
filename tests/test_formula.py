import numpy as np
from sbfl.formula import *

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

def test_get_spectrum():
    X, y = X_y_sample_1()

    ochiai = Ochiai()
    e_p, n_p, e_f, n_f = ochiai.get_spectrum(X, y)

    num_passings = y.sum()
    num_failings = np.invert(y).sum()

    assert e_p[0] == 2 and e_p[1] == 1 and e_p[2] == 1
    assert e_f[0] == 0 and e_f[1] == 0 and e_f[2] == 1
    assert np.all(e_p + n_p == num_passings)
    assert np.all(e_f + n_f == num_failings)

def test_ochai():
    X, y = X_y_sample_1()
    ochiai = Ochiai()
    ochiai.fit(X, y)
    scores = ochiai.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1/np.sqrt(2)

def test_tarantula():
    X, y = X_y_sample_1()
    tarantula = Tarantula()
    tarantula.fit(X, y)
    scores = tarantula.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 2/3

def test_jaccard():
    X, y = X_y_sample_1()
    jaccard = Jaccard()
    jaccard.fit(X, y)
    scores = jaccard.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1/2

def test_russellrao():
    X, y = X_y_sample_1()
    rr = RussellRao()
    rr.fit(X, y)
    scores = rr.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1/3

def test_hamann():
    X, y = X_y_sample_1()
    hamann = Hamann()
    hamann.fit(X, y)
    scores = hamann.scores_
    assert scores[0] == -1
    assert scores[1] == -1/3
    assert scores[2] == 1/3