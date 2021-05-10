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