import pytest
import sys
from pathlib import Path
from unittest.mock import patch
from sbfl.__main__ import TestInformation, main, _argparse
from sbfl.base import SBFL

RESOURCES_PATH = Path(__file__).parent.parent / 'resources'


def test_help_message(capsys):
    with patch.object(sys, 'argv', ['sbfl', '-h']):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        captured = capsys.readouterr()
        assert pytest_wrapped_e.type == SystemExit
        assert captured.out.startswith('usage: sbfl')


def test_argument_formula_is_required(capsys):
    with patch.object(sys, 'argv', ['sbfl', str((RESOURCES_PATH / 'yara-buggy#3-100'))]):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            _argparse()
        captured = capsys.readouterr()
        assert pytest_wrapped_e.type == SystemExit
        assert captured.err.startswith('usage:') and captured.err.endswith('-f/--formula\n')


@pytest.mark.parametrize('gcov_dirs,expect_error', [
    ([], True),
    ([RESOURCES_PATH / 'yara-buggy#3-100' / '100.output'], True),
    ([RESOURCES_PATH / 'yara-buggy#3-100'], False),
    ([d for d in RESOURCES_PATH.glob('yara-buggy#3-1*')], False)
])
def test_check_gcov_dirs(gcov_dirs, expect_error):
    sbfl = SBFL(formula='Ochiai')
    if expect_error:
        with pytest.raises(ValueError):
            TestInformation(sbfl, gcov_dirs)
    else:
        TestInformation(sbfl, gcov_dirs)
