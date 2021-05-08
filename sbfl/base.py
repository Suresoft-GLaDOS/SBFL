import numpy as np
from sklearn.utils import check_X_y
from . import formula

class NoFailingTestException(Exception):
    pass

class SBFL:
    def __init__(self, formula='Ochiai'):
        self.formula = formula
        self.scores_ = None

    def check_X_y(self, X, y):
        """Validate Input"""
        X, y = check_X_y(X, y, accept_sparse=False, dtype=bool,
            ensure_2d=True, y_numeric=True, multi_output=False)
        if np.invert(y).sum() == 0:
            raise NoFailingTestException
        return X, y

    def get_spectrum(self, X, y):
        """
        Convert coverage data (X) and test results (y) to program execution spectrum
 
        Return: (e_p, n_p, e_f, n_f)
            - e_p: the number of passing tests that cover each elements
            - n_p: the number of passing tests that do not cover each elements
            - e_f: the number of failing tests that cover each elements
            - n_f: the number of failing tests that do not cover each elements
        """
        X, y = self.check_X_y(X, y)

        is_passing = y
        is_failing = np.invert(y)

        e_p = X[is_passing].sum(axis=0)
        e_f = X[is_failing].sum(axis=0)   
        n_p = np.sum(is_passing) - e_p
        n_f = np.sum(is_failing) - e_f

        return e_p, n_p, e_f, n_f

    def fit(self, X, y):
        e_p, e_f, n_p, n_f = self.get_spectrum(X, y)
        """Compute suspiciousness scores and store it in self.scores_"""
        fml = getattr(formula, self.formula)
        self.scores_ = fml(e_p, e_f, n_p, n_f)
        self.e_p, self.e_f, self.n_p, self.n_f = e_p, e_f, n_p, n_f

    def fit_predict(self, X, y):
        """Compute and return suspiciousness scores"""
        self.fit(X, y)
        return self.scores_