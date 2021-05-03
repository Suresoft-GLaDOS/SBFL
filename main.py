import numpy as np
from sbfl.formula import *

if __name__ == "__main__":
    X = np.array([
        [1,0,1], # coverage of test t0
        [0,0,1], # coverage of test t1
        [1,1,0]  # coverage of test t2
    ], dtype=bool)

    y = np.array([1,0,1], dtype=bool)

    ochiai = Ochiai()
    ochiai.fit(X, y)
    print(ochiai.scores_)
