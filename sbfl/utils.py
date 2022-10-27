import os
import re
import pandas as pd
from tqdm import tqdm
from . import base

def is_function_summary(l: str) -> str or None:
    m = re.match(
        r"^function (\S+) called \d+ returned \d+% blocks executed \d+%", l
    )
    if m:
        return m.group(1)
    return None

def is_line_coverage(l: str) -> bool:
    m = re.match(r"^\s+\S+:\s*\d+:", l)
    return m is not None

def parse_gcov_line(l: str) -> tuple:
    """Parses each line in gcov file

    Parameter
    ----------
    l : str

    Returns
    -------
    tuple (str, int, str)
    """
    l_split = l.split(':')

    # parse the line
    hits = l_split[0].strip()
    lineno = int(l_split[1].strip())
    content =  ':'.join(l_split[2:]).rstrip()

    return hits, lineno, content

def read_gcov(path_to_file, only_coverable=True, encoding='utf-8') -> dict:
    """ Parses a gcov file

    Parameters
    ----------
    path_to_file : str or path-like object pointing to a file
    only_coverable : bool, optional

    Returns
    -------
    tuple (str, dict)
        a tuple of source file name and dict-type line coverage data
        line coverage data: dict(function, lineno: hits)
            -   -1: not coverable (hits == '-')
            -    0: coverable, but not covered (hits == '#####' or '=====')
            -  > 0: coverable and covered (hits == <number>)
    """
    source = None
    graph = None
    coverage = {}
    function = None
    try:
        with open(path_to_file, 'r', encoding=encoding) as gcov_file:
            for l in gcov_file:
                if is_function_summary(l) is not None:
                    function = is_function_summary(l)
                    continue

                if not is_line_coverage(l):
                    continue

                hits, lineno, content = parse_gcov_line(l)

                if lineno == 0:
                    # read metadata
                    if content.startswith('Source'):
                        source = content.split(':')[1]
                    if content.startswith('Graph'):
                        graph = content.split(':')[1]
                        if not graph.strip():
                            graph = None
                    continue

                dict_key = (function, lineno)
                if dict_key in coverage and coverage[dict_key] > 0:
                    raise Exception("Duplicated", path_to_file, dict_key)

                if hits == "-":
                    if only_coverable:
                        continue
                    else:
                        coverage[dict_key] = -1
                elif hits == "#####" or hits == "=====":
                    coverage[dict_key] = 0
                else:
                    if hits.endswith("*"):
                        hits = hits[:-1]
                    coverage[dict_key] = int(hits)
    except Exception as e:
        raise Exception(f"Error while reading {path_to_file}: {e}")

    if source is None:
        raise Exception(f"Unable to read soure file from {path_to_file}")

    return source, graph, coverage

def gcov_files_to_frame(gcov_files: dict, only_coverable=True,
    only_covered=False, verbose=False, **kwargs):
    """ Converts gcov files to a coverage matrix

    If verbose is set to True, a progress bar will be printed.

    Parameters
    ----------
    gcov_files : dict
        the mapping from a test name to a list of gcov files
    only_coverable : bool, optional
    only_covered : bool, optional
    verbose : bool, optional

    Returns
    -------
    pd.Dataframe
        a pandas dataframe representing the coverage matrix
        whose index is two-level(source, line number)
        and column is test case name

    Q. What's Multi-index?: https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.html
    """

    # coverage: source -> line -> test -> hits
    coverage = {}
    for test in tqdm(gcov_files) if verbose else gcov_files:
        for path_to_file in gcov_files[test]:
            src, grp, line_coverage = read_gcov(
                path_to_file, only_coverable=only_coverable, **kwargs)
            if grp is None:
                source = src
            else:
                source = grp + "//" + src
            if source not in coverage:
                coverage[source] = {}

            for dict_key in line_coverage:
                function, lineno = dict_key
                hits = line_coverage[dict_key]

                if dict_key not in coverage[source]:
                    coverage[source][dict_key] = {}
                if test in coverage[source][dict_key]:
                     raise Exception(f"{test} is already in coverage[{source}][{dict_key}]")
                coverage[source][dict_key][test] = hits

    data = [] # data
    index = [] # two-level index
    columns = list(gcov_files) # test case name

    for source in coverage:
        for dict_key in coverage[source]:
            function, lineno = dict_key
            index.append((source, function, lineno))
            data.append([coverage[source][dict_key].get(test, 0) for test in columns])

    # create dataframe
    df = pd.DataFrame(
        data, index=pd.MultiIndex.from_tuples(index,
                names=['file', 'function', 'line']), columns=columns)

    if only_covered:
        covered = df.values.sum(axis=1) > 0
        return df.iloc[covered]

    return df

def get_sbfl_scores_from_frame(cov_df, failing_tests, sbfl=None):
    """
    Calculates sbfl scores from the coverage-matrix dataframe `cov_df` and `failing_tests`

    Parameters
    ----------
    cov_df : pd.Dataframe
        a pandas DataFrame format coverage matrix
        index: source, line number (two-level)
        column: test case name
    failing_tests: Iterable (set or list)
        a list/set of failing test names
    sbfl: SBFL, optional
        SBFL-type instance

    Returns
    -------
    pd.Dataframe
        a pandas dataframe representing the SBFL scores
        that has only one column, score
    """
    assert all([t in cov_df.columns for t in failing_tests])
    X, y = cov_df.values.T > 0, ~cov_df.columns.isin(failing_tests)
    if sbfl is None:
        sbfl = base.SBFL()
    sbfl.fit(X, y)
    return sbfl.to_frame(index=cov_df.index)

def read_dfcpp_coverage(d4cpp_output_dir, **kwargs):
    """
    Returns coverage data
    """
    coverage_dirs = {}

    for dir_name in os.listdir(d4cpp_output_dir):
        path_to_dir = os.path.join(d4cpp_output_dir, dir_name)
        if not os.path.isdir(path_to_dir):
            continue
        case = dir_name.split('-')[-1]
        coverage_dirs[case] = path_to_dir

    coverage_files = {
        test: [
            os.path.join(coverage_dirs[test], fn)
            for fn in os.listdir(coverage_dirs[test]) if fn.endswith('gcov')
        ]
        for test in coverage_dirs
    }

    return gcov_files_to_frame(coverage_files, **kwargs)

def read_dfcpp_test_results(d4cpp_output_dir):
    """
    Read d4cpp test results

    Parameters
    ----------
    d4cpp_output_dir : Path or st
        path to the d4cpp output directory containing the coverage data

    Returns
    -------
    dict
        test results (key: test case name, value: "passed" or "failed")
    """
    test_results = {}
    for dir_name in os.listdir(d4cpp_output_dir):
        path_to_dir = os.path.join(d4cpp_output_dir, dir_name)
        if not os.path.isdir(path_to_dir):
            continue
        case = dir_name.split('-')[-1]
        result_path = os.path.join(path_to_dir, f"{case}.test")
        with open(result_path, 'r') as f:
            test_results[case] = f.read().strip()

    return test_results
