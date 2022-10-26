import pytest
import numpy as np
from sbfl.base import SBFL


def test_reproduce_inf_in_ochiai2():
    # np.var(0) / np.var(0) is inf
    X = np.array([
        [0,0,1],
        [0,0,1],
        [0,1,0]
    ], dtype=bool)
    y = np.array([1,1,0], dtype=bool)

    with pytest.deprecated_call():
        sbfl = SBFL(formula='Ochiai2')
        scores = sbfl.fit_predict(X, y)

    assert not np.isinf(scores[0])
    assert not np.isinf(scores[1])
    assert not np.isinf(scores[2])


def test_reprodue_nan_in_ochiai():
    # non-zero-np-var / np.var(0) is nan
    X = np.array([
        [0,0,1],
        [0,0,1],
        [0,1,0]
    ], dtype=bool)
    y = np.array([1,1,0], dtype=bool)

    sbfl = SBFL(formula='Ochiai')

    scores = sbfl.fit_predict(X, y)

    assert not np.isnan(scores[0])
    assert not np.isnan(scores[1])
    assert not np.isnan(scores[2])
