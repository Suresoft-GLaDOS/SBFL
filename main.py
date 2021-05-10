import numpy as np
from sbfl.base import SBFL

if __name__ == "__main__":
    X = np.array([
        [1,0,1], # coverage of test t0
        [0,0,1], # coverage of test t1
        [1,1,0]  # coverage of test t2
    ], dtype=bool)

    y = np.array([
        1, # t0: PASS
        0, # t1: FAIL
        1  # t2: PASS
    ], dtype=bool)

    ochiai = SBFL(formula='Ochiai')

    ochiai.fit(X, y)
    print(ochiai.scores_)
    print("Ranks with min tiebreaker:", ochiai.ranks(method='min'))
    print("Ranks with avg tiebreaker:", ochiai.ranks(method='average'))
    print("Ranks with max tiebreaker:", ochiai.ranks(method='max'))
    print("Ranks with dense tiebreaker:", ochiai.ranks(method='dense'))
    print("Ranks with ordinal tiebreaker:", ochiai.ranks(method='ordinal'))