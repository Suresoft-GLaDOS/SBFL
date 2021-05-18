import numpy as np
import pandas as pd
from inspect import getmembers, isfunction
from sklearn.utils import check_X_y
from scipy.stats import rankdata
from . import sbfl_formula

class NoFailingTestError(Exception):
    """Raised when there is no failing test (0 not in y)"""
    pass

class NotSupportedFormulaError(Exception):
    """Raised when the formula is not supported"""
    pass

class SBFL:
    def __init__(self, formula='Ochiai'):
        supported_formulae = dict(getmembers(sbfl_formula, isfunction))
        if formula not in supported_formulae:
            raise NotSupportedFormulaError(f"Supported formulae: {supported_formulae}")

        self.formula = formula
        self.formula_func = supported_formulae[formula]

    def check_X_y(self, X, y):
        """Validate Input"""
        X, y = check_X_y(X, y, accept_sparse=False, dtype=bool,
            ensure_2d=True, y_numeric=True, multi_output=False)
        if np.invert(y).sum() == 0:
            raise NoFailingTestError
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
        """Compute suspiciousness scores and store it in self.scores_"""
        e_p, e_f, n_p, n_f = self.get_spectrum(X, y)
        self.scores_ = self.formula_func(e_p, e_f, n_p, n_f)
        self.e_p, self.e_f, self.n_p, self.n_f = e_p, e_f, n_p, n_f

    def fit_predict(self, X, y):
        """Compute and return suspiciousness scores"""
        self.fit(X, y)
        return self.scores_

    def ranks(self, method='average'):
        """
        An array of size equl to the size of scores_, containing ranks 

        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
        for the details of ranking methods (tie-breakers).
        """
        return rankdata(-self.scores_, method=method)

    def to_frame(self, elements=None):
        """
        Convert self.scores_ to a Pandas DataFrame object `df`

        When `elements` is not None,
        - if `elements` is a list of tuple, set the index of `df` to a MultiIndex made from the tuples
        - otherwise, set the index of `df` to `elements`.
        """
        if elements is None:
            index = None
        else:
            if all([isinstance(e, tuple) for e in elements]):
                index = pd.MultiIndex.from_tuples(elements)
            else:
                index = pd.Index(elements)

        df = pd.DataFrame({'score': self.scores_}, index=index)

        return df