import pytest
from sbfl.utils import *

def test_is_function_summary():
    assert "chooseCandidate" == is_function_summary(
        "function chooseCandidate called 9 returned 100% blocks executed 73%")
    assert None == is_function_summary(
        "    #####:   95:        if (!strcmp(str, kb_type_str[i]))")
    assert None == is_function_summary(
        "branch  0 never executed")


def test_line_coverage():
    assert not is_line_coverage(
        "function chooseCandidate called 9 returned 100% blocks executed 73%")
    assert is_line_coverage(
        "    #####:   95:        if (!strcmp(str, kb_type_str[i]))")
    assert not is_line_coverage("branch  0 never executed")

def test_parse_gcov_line():
    hits, lineno, content = parse_gcov_line(
        "    #####:   95:        if (!strcmp(str, kb_type_str[i]))")
    assert hits == "#####"
    assert lineno == 95
    assert content == "        if (!strcmp(str, kb_type_str[i]))"