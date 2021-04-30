# Spectrum-based Fault Localization

[![test](https://github.com/Suresoft-GLaDOS/SBFL/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Suresoft-GLaDOS/SBFL/actions/workflows/run_tests.yml)

## Environment
- Tested under Python 3.9.1
- Install dependencies
  ```bash
  python -m pip install -r requirements.txt
  ```

## Run tests
```bash
# without measuring coverage
python -m pytest

# with measuring coverage
python -m pip install coverage
python -m coverage run --source=sbfl -m pytest
python -m coverage report
```
