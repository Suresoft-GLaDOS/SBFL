import numpy as np
import pandas as pd
from sbfl.base import SBFL
from sbfl.diagnosability import DDU

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

    print(f"DDU: {DDU(X)}")

    """
    Calculate the suspiciousness scores
    """
    sbfl = SBFL(formula='Ochiai')
    sbfl.fit(X, y)
    print(sbfl.ranks(method='max'))


    """
    file-level score aggregation (max)
    """
    names = ['file', 'method']
    elements = [
        ('file1.py', 'method1'),
        ('file2.py', 'method2'),
        ('file2.py', 'method3')
    ]
    df = sbfl.to_frame(elements=elements, names=names)

    print(df)
    print(df.max(level='file'))