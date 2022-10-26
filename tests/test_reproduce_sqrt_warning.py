import pytest
import json
import numpy as np
from pathlib import Path
from sbfl.base import SBFL

RESOURCES_PATH = Path(__file__).parent / 'resources'

@pytest.fixture
def get_X_y():
    def _get_X_y(file_path):
        with open(file_path) as f:
            test_data = json.load(f)
        coverages, results = [], []
        for test_case in test_data:
            coverages.append([int(bit) for bit in test_case['bitmap']])
            results.append(test_case['result'])
        X = np.array(coverages, dtype=bool)
        y = np.array(results, dtype=bool)
        return X, y
    return _get_X_y

def test_reproduce_sqrt_warning_in_ochiai2(get_X_y, capsys):
    # halt when np.seterr(invalid='raise')
    # np.seterr(invalid='raise')
    X, y = get_X_y(RESOURCES_PATH / 'input.json')
    with pytest.deprecated_call():
        sbfl = SBFL(formula='Ochiai2')
        scores = sbfl.fit_predict(X, y)
    assert not any(np.isnan(score) for score in scores)
