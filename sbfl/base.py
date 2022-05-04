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
        if callable(formula):
            self.formula_func = formula
        else:
            supported_formulae = dict(getmembers(sbfl_formula, isfunction))
            if formula not in supported_formulae:
                raise NotSupportedFormulaError(f"Supported formulae: {set(supported_formulae.keys())}")
            self.formula_func = supported_formulae[formula]
        self.formula = formula

    @staticmethod
    def validate_input(X, y):
        X = np.asanyarray(X)
        y = np.asanyarray(y).astype(bool)
        X, y = check_X_y(X, y, accept_sparse=False, dtype=bool,
            ensure_2d=True, y_numeric=True, multi_output=False)
        if (y==0).sum() == 0:
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
        X, y = self.validate_input(X, y)

        is_passing = y
        is_failing = np.invert(y)

        e_p = X[is_passing].sum(axis=0)
        e_f = X[is_failing].sum(axis=0)
        n_p = np.sum(is_passing) - e_p
        n_f = np.sum(is_failing) - e_f

        return e_p, n_p, e_f, n_f

    def fit(self, X, y):
        """Compute suspiciousness scores and store to self.scores_"""
        e_p, e_f, n_p, n_f = self.get_spectrum(X, y)
        self.scores_ = self.formula_func(e_p, e_f, n_p, n_f)
        self.e_p, self.e_f, self.n_p, self.n_f = e_p, e_f, n_p, n_f

    def fit_predict(self, X, y):
        """Compute and return suspiciousness scores"""
        self.fit(X, y)
        return self.scores_

    def ranks(self, method='average'):
        """
        An array of size equals to the size of scores_, containing ranks

        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
        for the details of ranking methods (tie-breakers).
        """
        return rankdata(-self.scores_, method=method)

    def to_frame(self, elements=None, names=None, index=None):
        """
        Convert self.scores_ to a Pandas DataFrame object `df`

        When `elements` is not None, it should be a list of tuples, and
        the index of `df` is set to a MultiIndex made from the tuples.
        """
        if index is None and elements is not None:
            index = pd.MultiIndex.from_tuples(elements, names=names)
        return pd.DataFrame({'score': self.scores_}, index=index)

