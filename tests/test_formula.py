import numpy as np
from sbfl.base import SBFL

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
    assert scores[2] == 1/np.sqrt(2)

def test_tarantula():
    X, y = X_y_sample_1()
    tarantula = SBFL(formula='Tarantula')
    tarantula.fit(X, y)
    scores = tarantula.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 2/3

def test_jaccard():
    X, y = X_y_sample_1()
    jaccard = SBFL(formula='Jaccard')
    jaccard.fit(X, y)
    scores = jaccard.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1/2

def test_russellrao():
    X, y = X_y_sample_1()
    rr = SBFL(formula='RussellRao')
    rr.fit(X, y)
    scores = rr.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 1/3

def test_hamann():
    X, y = X_y_sample_1()
    hamann = SBFL(formula='Hamann')
    hamann.fit(X, y)
    scores = hamann.scores_
    assert scores[0] == -1
    assert scores[1] == -1/3
    assert scores[2] == 1/3

def test_sorensondice():
    X, y = X_y_sample_1()
    sd = SBFL(formula='SorensonDice')
    sd.fit(X, y)
    scores = sd.scores_
    assert scores[0] == 0
    assert scores[1] == 0
    assert scores[2] == 2/3

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
    assert scores[2] == 3/4

def test_simplematching():
    X, y = X_y_sample_1()
    sm = SBFL(formula='SimpleMatching')
    sm.fit(X, y)
    scores = sm.scores_
    assert scores[0] == 0
    assert scores[1] == 1/3
    assert scores[2] == 2/3

def test_sokal():
    X, y = X_y_sample_1()
    sokal = SBFL(formula='Sokal')
    sokal.fit(X, y)
    scores = sokal.scores_
    assert scores[0] == 0
    assert scores[1] == 1/2
    assert scores[2] == 4/5
