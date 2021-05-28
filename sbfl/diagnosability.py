import numpy as np

def diversity(X):
    N, _ = X.shape
    _, counts = np.unique(X, axis=0, return_counts=True)
    if N > 1:
        value = sum([ n * (n - 1) for n in counts ])
        value /= N * (N - 1)
        value = 1 - value
    else:
        value = 1.
    assert 0 <= value <= 1
    return value

def density(X):
    N, M = X.shape
    value = np.sum(X) / (N * M)
    assert 0 <= value <= 1
    return value

def uniqueness(X):
    _, M = X.shape
    unique_elems = np.unique(X, axis=1)
    value = unique_elems.shape[1] / M
    assert 0 <= value <= 1
    return value

def DDU(X):
    return diversity(X) * density(X) * uniqueness(X)