import pytest
import sys
from pathlib import Path
from unittest.mock import patch
from sbfl.__main__ import main, _check_gcov_dirs

RESOURCES_PATH = Path(__file__).parent.parent / 'resources'


def test_help_message(capsys):
    with patch.object(sys, 'argv', ['sbfl', '-h']):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            main()
        captured = capsys.readouterr()
        assert pytest_wrapped_e.type == SystemExit
        assert captured.out.startswith('usage: sbfl')


def test_check_gcov_dirs():
    assert not _check_gcov_dirs(None)
    assert not _check_gcov_dirs([])
    assert not _check_gcov_dirs([RESOURCES_PATH / 'yara-buggy#3-100' / '100.output'])
    assert _check_gcov_dirs([RESOURCES_PATH / 'yara-buggy#3-100'])
    assert _check_gcov_dirs([d for d in RESOURCES_PATH.glob('yara-buggy#3-1*')])
