import pytest
from distutils import dir_util
from sbfl.utils import *

@pytest.fixture
def datadir(tmpdir):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    test_dir, _ = os.path.splitext(os.path.abspath(__file__))
    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir

@pytest.fixture
def d4cpp_dir():
    test_dir, _ = os.path.splitext(os.path.abspath(__file__))
    return os.path.join(test_dir, "d4cpp_output")

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

def test_parse_gcov_line_with_contents_containing_colon():
    hits, lineno, content = parse_gcov_line(
        "    #####:   95:        (age >= 18) ? printf(\"Can Vote\") : printf(\"Cannot Vote\");")
    assert hits == "#####"
    assert lineno == 95
    assert content == "        (age >= 18) ? printf(\"Can Vote\") : printf(\"Cannot Vote\");"


def test_read_gcov(datadir):
    source, graph, coverage = read_gcov(datadir.join("threading.c.gcov"))
    assert source == "threading.c"
    assert graph == "/home/workspace/libyara/threading.gcno"
    assert all([coverage[key] != -1 for key in coverage])
    assert len(coverage) == 32

def test_read_gcov_uncoverable_lines(datadir):
    source, graph, coverage = read_gcov(datadir.join("threading.c.gcov"),
        only_coverable=False)
    assert source == "threading.c"
    assert graph == "/home/workspace/libyara/threading.gcno"
    not_coverable_lines = 0
    for key in coverage:
        if coverage[key] == -1:
            not_coverable_lines += 1
        elif coverage[key] > 0:
            assert coverage[key] in [1, 8, 12]
    assert not_coverable_lines == 209 - 32
    assert len(coverage) == 209

def test_read_gcov_with_damaged_gcov_file(datadir):
    with pytest.raises(Exception):
        source, graph, coverage = read_gcov(
            datadir.join("damaged_threading.c.gcov"))

def test_read_gcov_with_non_existing_file(datadir):
    with pytest.raises(Exception):
        source, graph, coverage = read_gcov(
            datadir.join("w_threading.c.gcov"))

def test_gcov_files_to_frame(d4cpp_dir):
    gcov_dir = {}
    for tc in os.listdir(d4cpp_dir):
        if os.path.isdir(os.path.join(d4cpp_dir, tc)):
            gcov_dir[tc] = os.path.join(d4cpp_dir, tc)
    gcov_files = {test:[] for test in gcov_dir}
    for test in gcov_dir:
        for path in os.listdir(gcov_dir[test]):
            if path.endswith('.gcov'):
                gcov_files[test].append(os.path.join(gcov_dir[test], path))

    cov_df = gcov_files_to_frame(gcov_files, only_covered=True)
    assert cov_df.shape[1] == len(gcov_files)
    assert (cov_df.sum(axis=1) == 0).sum() == 0
    assert cov_df.loc[
        "/home/workspace/libyara/threading.gcno//threading.c"].shape[0] > 0
    assert cov_df.index.get_level_values("function").isna().all()
    assert list(cov_df.index.names) == ["file", "function", "line"]

def test_get_sbfl_scores_from_frame(d4cpp_dir):
    gcov_dir = {}
    for tc in os.listdir(d4cpp_dir):
        if os.path.isdir(os.path.join(d4cpp_dir, tc)):
            gcov_dir[tc] = os.path.join(d4cpp_dir, tc)
    gcov_files = {test:[] for test in gcov_dir}
    for test in gcov_dir:
        for path in os.listdir(gcov_dir[test]):
            if path.endswith('.gcov'):
                gcov_files[test].append(os.path.join(gcov_dir[test], path))

    cov_df = gcov_files_to_frame(gcov_files, only_covered=True)
    score_df = get_sbfl_scores_from_frame(cov_df,
        failing_tests=['yara-buggy#3-102'])
    assert (cov_df.loc[score_df.score == 0, 'yara-buggy#3-102'] == 0).all()

def test_read_dfcpp_test_results(d4cpp_dir):
    cov_df = read_dfcpp_coverage(d4cpp_dir)
    assert cov_df.shape[1] == 5
    assert (cov_df.sum(axis=1) == 0).sum() == 0
    assert cov_df.loc[
        "/home/workspace/libyara/threading.gcno//threading.c"].shape[0] > 0
    assert cov_df.index.get_level_values("function").isna().all()
    assert list(cov_df.index.names) == ["file", "function", "line"]

def test_read_dfcpp_test_results(d4cpp_dir):
    test_results = read_dfcpp_test_results(d4cpp_dir)
    assert len(test_results) == 5
    for tc in test_results:
        assert test_results[tc] in ["failed", "passed"]