import numpy as np
import pytest
from numpy.testing import assert_allclose
from sbfl.base import SBFL

def assert_equal(s1, s2, error=1e-05):
    return pytest.approx(s1, error) == s2

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
Test each formula
"""
def test_ochiai():
    X, y = X_y_sample_1()
    ochiai = SBFL(formula='Ochiai')
    ochiai.fit(X, y)
    scores = ochiai.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert_allclose(scores[2], 1/np.sqrt(2))

def test_tarantula():
    X, y = X_y_sample_1()
    tarantula = SBFL(formula='Tarantula')
    tarantula.fit(X, y)
    scores = tarantula.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert_allclose(scores[2], 2/3)

def test_jaccard():
    X, y = X_y_sample_1()
    jaccard = SBFL(formula='Jaccard')
    jaccard.fit(X, y)
    scores = jaccard.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert_allclose(scores[2], 1/2)

def test_russellrao():
    X, y = X_y_sample_1()
    rr = SBFL(formula='RussellRao')
    rr.fit(X, y)
    scores = rr.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert_allclose(scores[2], 1/3)

def test_hamann():
    X, y = X_y_sample_1()
    hamann = SBFL(formula='Hamann')
    hamann.fit(X, y)
    scores = hamann.scores_
    assert scores[0] == -1
    assert_allclose(scores[1], -1/3)
    assert_allclose(scores[2], 1/3)

def test_sorensondice():
    X, y = X_y_sample_1()
    sd = SBFL(formula='SorensonDice')
    sd.fit(X, y)
    scores = sd.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert_allclose(scores[2], 2/3)

def test_dice():
    X, y = X_y_sample_1()
    dice = SBFL(formula='Dice')
    dice.fit(X, y)
    scores = dice.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1

def test_kulczynski1():
    X, y = X_y_sample_1()
    k1 = SBFL(formula='Kulczynski1')
    k1.fit(X, y)
    scores = k1.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1

def test_kulczynski2():
    X, y = X_y_sample_1()
    k2 = SBFL(formula='Kulczynski2')
    k2.fit(X, y)
    scores = k2.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert_allclose(scores[2], 3/4)

def test_simplematching():
    X, y = X_y_sample_1()
    sm = SBFL(formula='SimpleMatching')
    sm.fit(X, y)
    scores = sm.scores_
    assert scores[0] == 0
    assert_allclose(scores[1], 1/3)
    assert_allclose(scores[2], 2/3)

def test_sokal():
    X, y = X_y_sample_1()
    sokal = SBFL(formula='Sokal')
    sokal.fit(X, y)
    scores = sokal.scores_
    assert scores[0] == 0
    assert_allclose(scores[1], 1/2)
    assert_allclose(scores[2], 4/5)

def test_m1():
    X, y = X_y_sample_1()
    m1 = SBFL(formula='M1')
    m1.fit(X, y)
    scores = m1.scores_
    assert scores[0] == 0
    assert_allclose(scores[1], 1/2)
    assert scores[2] == 2

def test_m2():
    X, y = X_y_sample_1()
    m2 = SBFL(formula='M2')
    m2.fit(X, y)
    scores = m2.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert_allclose(scores[2], 1/4)

def test_op1():
    X, y = X_y_sample_1()
    op1 = SBFL(formula="Op1")
    op1.fit(X, y)
    scores = op1.scores_
    assert scores[0] == -1
    assert scores[1] == -1
    assert scores[2] == 1

def test_op2():
    X, y = X_y_sample_1()
    op2 = SBFL(formula="Op2")
    op2.fit(X, y)
    scores = op2.scores_
    assert_allclose(scores[0], -2/3)
    assert_allclose(scores[1], -1/3)
    assert_allclose(scores[2], 2/3)

def test_wong1():
    X, y = X_y_sample_1()
    wong1 = SBFL(formula="Wong1")
    wong1.fit(X, y)
    scores = wong1.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1

def test_wong2():
    X, y = X_y_sample_1()
    wong2 = SBFL(formula="Wong2")
    wong2.fit(X, y)
    scores = wong2.scores_
    assert scores[0] == -2
    assert scores[1] == -1
    assert scores[2] == 0

def test_wong3():
    X, y = X_y_sample_1()
    wong3 = SBFL(formula="Wong3")
    wong3.fit(X, y)
    scores = wong3.scores_
    assert scores[0] == -2
    assert scores[1] == -1
    assert scores[2] == 0

def test_ample():
    X, y = X_y_sample_1()
    ample = SBFL(formula="Ample")
    ample.fit(X, y)
    scores = ample.scores_
    assert scores[0] == 1.0
    assert scores[1] == 0.5
    assert scores[2] == 0.5  

def test_dstar2():
    X, y = X_y_sample_1()
    dstar2 = SBFL(formula="Dstar2")
    dstar2.fit(X, y)
    scores = dstar2.scores_
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 1.0

def test_gp02():
    X, y = X_y_sample_1()
    gp02 = SBFL(formula="GP02")
    gp02.fit(X, y)
    scores = gp02.scores_
    assert_allclose(scores[0], 4.242641)
    assert_allclose(scores[1], 3.828427)
    assert_allclose(scores[2], 5.828427)

def test_gp03():
    X, y = X_y_sample_1()
    formula = SBFL(formula="GP03")
    formula.fit(X, y)
    scores = formula.scores_
    assert_allclose(scores[0], 1.189207)
    assert_allclose(scores[1], 1.0)
    assert_allclose(scores[2], 0.0)

def test_gp13():
    X, y = X_y_sample_1()
    formula = SBFL(formula="GP13")
    formula.fit(X, y)
    scores = formula.scores_
    assert_allclose(scores[0], 0.0)
    assert_allclose(scores[1], 0.0)
    assert_allclose(scores[2], 4 / 3)

def test_gp19():
    X, y = X_y_sample_1()
    formula = SBFL(formula="GP19")
    formula.fit(X, y)
    scores = formula.scores_
    assert_allclose(scores[0], 0.0)
    assert_allclose(scores[1], 0.0)
    assert_allclose(scores[2], 1.0)

def test_er5c():
    X, y = X_y_sample_1()
    formula = SBFL(formula="ER5C")
    formula.fit(X, y)
    scores = formula.scores_
    assert_allclose(scores[0], 0.0)
    assert_allclose(scores[1], 0.0)
    assert_allclose(scores[2], 1.0)

