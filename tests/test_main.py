import pytest
import sys
import json
import math
import numpy
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


def test_outputs(capsys, tmp_path):
    numpy.seterr(all='raise')
    with patch.object(sys, 'argv', ['sbfl', '-f', 'Ochiai', '--sbfl-out', (tmp_path / 'sbfl.json').as_posix(),
                                    '--info-out', (tmp_path / 'test_info.json').as_posix(),
                                    str(RESOURCES_PATH / 'yara-buggy#3*')]):
        main()
    captured = capsys.readouterr()
    sbfl_output = json.loads((tmp_path / 'sbfl.json').read_text())
    assert sbfl_output[0][0] == "/usr/include/openssl/x509.h"
    assert sbfl_output[0][1] == 99
    assert math.isnan(sbfl_output[0][2])
    assert len(sbfl_output) == 13696
    info_output = json.loads((tmp_path / 'test_info.json').read_text())
    assert info_output['formula'] == 'Ochiai'
    assert any([('yara-buggy#3-100' in gcov_dir) for gcov_dir in info_output['test']['passing']])
    assert 'yara-buggy#3-102' in info_output['test']['failing'][0]
    assert 'libyara.c.gcov' in info_output['sources']
    assert info_output['coverage'] == '5.68'


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
