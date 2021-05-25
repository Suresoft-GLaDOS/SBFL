import numpy as np

def Ochiai(e_p, n_p, e_f, n_f):
    return e_f/np.sqrt(((e_f + n_f) * (e_f + e_p)))

def Tarantula(e_p, n_p, e_f, n_f):
    r_f = e_f/(e_f + n_f)
    r_p = e_p/(e_p + n_p)
    return r_f/(r_f + r_p)

def Jaccard(e_p, n_p, e_f, n_f):
    return e_f/(e_f + n_f + e_p)

def RussellRao(e_p, n_p, e_f, n_f):
    return e_f/(e_f + n_f + e_p + n_p)

def Hamann(e_p, n_p, e_f, n_f):
    return (e_f + n_p - e_p - n_f)/(e_f + n_f + e_p + n_p)

def SorensonDice(e_p, n_p, e_f, n_f):
    return (2 * e_f)/(2 * e_f + e_p + n_f)

def Dice(e_p, n_p, e_f, n_f):
    return (2 * e_f)/(e_f + n_f + e_p)

def Kulczynski1(e_p, n_p, e_f, n_f):
    return e_f/(n_f + e_p)

def Kulczynski2(e_p, n_p, e_f, n_f):
    return e_f * (1/(e_f + n_f) + 1/(e_f + e_p)) / 2

def SimpleMatching(e_p, n_p, e_f, n_f):
    return (e_f + n_p)/(e_f + n_f + e_p + n_p)

def Sokal(e_p, n_p, e_f, n_f):
    return (2 * e_f + 2 * n_p)/(2 * e_f + 2 * n_p + n_f + e_p)

def M1(e_p, n_p, e_f, n_f):
    return (e_f + n_p)/(n_f + e_p)

def M2(e_p, n_p, e_f, n_f):
    return e_f/(e_f + n_p + 2 * n_f + 2 * e_p)

def Op2(e_p, n_p, e_f, n_f):
    return e_f - e_p / (e_p + n_p + 1.0)

def Op1(e_p, n_p, e_f, n_f):
    _scores = np.array([-1.0] * len(e_p))
    _condition = n_f <= 0
    _scores[_condition] = n_p[_condition]
    return _scores

def Wong1(e_p, n_p, e_f, n_f):
    return e_f

def Wong2(e_p, n_p, e_f, n_f):
    return e_f - e_p

def Wong3(e_p, n_p, e_f, n_f):
    _scores = e_f - e_p
    _condition = 2 < e_p <= 10
    _scores[_condition] = e_f[_condition] - (2 + 0.1 * (e_p[_condition] - 2))
    _condition = e_p > 10
    _scores[_condition] = e_f[_condition] - (2.8 + 0.001 * (e_p[_condition] - 10))
    return _scores

def Ample(e_p, n_p, e_f, n_f):
    return abs(e_f / (e_f + n_f) - e_p / (e_p + n_p))