import numpy as np
from sbfl.formula import Ochiai

def X_y_sample_1():
    X = np.array([
        [1,0,1],
        [0,0,1],
        [1,1,0]
    ], dtype=bool)
    y = np.array([1,0,1], dtype=bool)
    return X, y

def test_ochai():
    X, y = X_y_sample_1()

    ochiai = Ochiai()
    ochiai.fit(X, y)
    scores = ochiai.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert np.round(scores[2], 3) == 0.707