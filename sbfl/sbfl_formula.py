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
    scores = n_p
    scores[n_f > 0] = -1
    return scores

def Wong1(e_p, n_p, e_f, n_f):
    return e_f

def Wong2(e_p, n_p, e_f, n_f):
    return e_f - e_p

def Wong3(e_p, n_p, e_f, n_f):
    cond1 = (e_p > 2) & (e_p <= 10)
    cond2 = e_p > 10

    h = e_p
    h[cond1] = 2 + 0.1 * (e_p[cond1] - 2)
    h[cond2] = 2.8 + 0.001 * (e_p[cond2] - 10)

    return e_f - h

def Ample(e_p, n_p, e_f, n_f):
    return np.absolute(e_f / (e_f + n_f) - e_p / (e_p + n_p))

def Dstar2(e_p, n_p, e_f, n_f):
    return np.absolute(e_f ** 2) / (np.absolute(e_p) + np.absolute(n_f))

def GP02(e_p, n_p, e_f, n_f):
    print(e_p, n_p, e_f, n_f)
    return 2 * (e_f + np.sqrt(n_p)) + np.sqrt(e_p)

def GP03(e_p, n_p, e_f, n_f):
    return np.sqrt(np.absolute(e_f ** 2 - np.sqrt(e_p)))

def GP13(e_p, n_p, e_f, n_f):
    return e_f * (1 + 1 / (2 * e_p + e_f))

def GP19(e_p, n_p, e_f, n_f):
    return e_f * np.sqrt(np.absolute(e_p - e_f + n_f - n_p))

def ER5C(e_p, n_p, e_f, n_f):
    scores = np.zeros(len(e_p))
    scores[n_f == 0] = 1.0
    return scores

def SBI(e_p, n_p, e_f, n_f):
    return e_f / (e_f + e_p)

def Goodman(e_p, n_p, e_f, n_f):
    return (2 * e_f - n_f - e_p) / (2 * e_f + n_f + e_p)

def Zoltar(e_p, n_p, e_f, n_f):
    scores = e_f
    nz = e_f != 0
    scores[nz] = e_f[nz] / (e_f[nz] + n_f[nz] + e_p[nz] + 10000 * n_f[nz] * e_p[nz] / e_f[nz])
    return scores

def Ochiai2(e_p, n_p, e_f, n_f):
    scores = np.zeros(len(e_f))
    cond = (e_f != 0) & (n_p != 0)
    scores[cond] = e_f[cond] * n_p[cond] / np.sqrt((e_f[cond] + e_p[cond]) * (n_f[cond] + n_p[cond]) * (e_f[cond] * n_p[cond]) * (n_f[cond] + e_p[cond]))
    return scores

def Anderberg(e_p, n_p, e_f, n_f):
    return e_f / (e_f + 2 * n_f + 2 * e_p)

def Hamming(e_p, n_p, e_f, n_f):
    return e_f + n_p

def RogerTanimoto(e_p, n_p, e_f, n_f):
    return (e_f + n_p) / (e_f + n_p + 2 * n_f + 2 * e_p)

def Euclid(e_p, n_p, e_f, n_f):
    return np.sqrt(e_f + n_p)

def Overlap(e_p, n_p, e_f, n_f):
    return e_f / np.minimum(e_f, np.minimum(e_p, n_f))