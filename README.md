# Spectrum-based Fault Localization

[![test](https://github.com/Suresoft-GLaDOS/SBFL/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Suresoft-GLaDOS/SBFL/actions/workflows/run_tests.yml)

⚠️ This engine is under construction.👷‍♀️

## Environment
- Developed & tested under Python 3.9.1
- Installing dependencies:
  ```bash
  python -m pip install -r requirements.txt
  ```

## Installation
```bash
git clone https://github.com/Suresoft-GLaDOS/SBFL
cd SBFL
pip install setuptools
python setup.py install # or pip install -e .
```

## Getting Started
```python
import numpy as np
from sbfl.base import SBFL

if __name__ == "__main__":
    """
    X: coverage data
    y: test results
    """
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

    """
    Calculate the suspiciousness scores
    """
    sbfl = SBFL(formula='Ochiai')
    sbfl.fit(X, y)
    print(sbfl.ranks(method='max'))
```

See the full example usage of this engine in [./main.ipynb](./main.ipynb).

## Running tests
- If you implement new functionality, please add the test cases for it.
- After any code change, make sure that the entire test suite passes.

```bash
# without measuring coverage
python -m pytest

# with measuring coverage
python -m pip install coverage
python -m coverage run --source=sbfl -m pytest
python -m coverage report
```
