import os
import pandas as pd
from . import base

def parse_gcov_line(l):
    """
    Parse each line in gcov file
    """
    l_split = l.split(':')

    # parse the line
    hits = l_split[0].strip()
    lineno = int(l_split[1].strip())
    content =  ':'.join(l_split[2:]).rstrip()

    return hits, lineno, content

def read_gcov(path_to_file, only_coverable=True):
    """
    Return a tuple of source file name and line coverage data
    - line coverage data: dict(lineno: hits)
        -   -1: not coverable (hits == '-')
        -    0: coverable, but not covered (hits == '#####')
        -  > 0: coverable and covered (hits == <number>)
    """
    source = None
    coverage = {}
    with open(path_to_file, 'r') as gcov_file:
        for l in gcov_file:
            hits, lineno, content = parse_gcov_line(l)

            if lineno == 0:
                # read metadata
                if content.startswith('Source'):
                    source = content.split(':')[1]
                continue
            
            if hits == "-":
                if only_coverable:
                    continue
                else:
                    coverage[lineno] = -1
            elif hits == "#####":
                coverage[lineno] = 0
            else:
                coverage[lineno] = int(hits)

    assert source is not None
    return source, coverage
        
def gcov_files_to_frame(gcov_files, only_coverable=True, only_covered=False):
    """
    Convert test cases' coverage data (list of .gcov files)
    to a pandas DataFrame format coverage matrix
    - index: source, line number (two-level)
    - column: test case name
    
    Q. What's Multi-index?: https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.html
    """

    # coverage: source -> line -> test -> hits
    coverage = {}
    for test in gcov_files:
        for path_to_file in gcov_files[test]:
            source, line_coverage = read_gcov(path_to_file, only_coverable=only_coverable)

            if source not in coverage:
                coverage[source] = {}

            source_coverage = coverage[source]

            for line in line_coverage:
                hits = line_coverage[line]

                if line not in source_coverage:
                    source_coverage[line] = {}
                
                assert test not in source_coverage[line]
                source_coverage[line][test] = hits


    data = [] # data
    index = [] # two-level index
    columns = list(gcov_files) # test case name

    for source in coverage:
        for line in coverage[source]:
            index.append((source, line))
            data.append([coverage[source][line].get(test, 0) for test in columns])

    # create dataframe
    df = pd.DataFrame(
        data, index=pd.MultiIndex.from_tuples(index, names=['source', 'line']), columns=columns)
    
    if only_covered:
        covered = df.values.sum(axis=1) > 0
        return df.iloc[covered]

    return df

def get_sbfl_scores_from_frame(cov_df, failing_tests, sbfl=None):
    """
    Calculate sbfl scores from the coverage-matrix dataframe `cov_df` and `failing_tests`

    - cov_df: a pandas DataFrame format coverage matrix
        - index: source, line number (two-level)
        - column: test case name
    - failing_tests: Iterable
    """
    assert all([t in cov_df.columns for t in failing_tests])
    X, y = cov_df.values.T > 0, cov_df.columns.isin(failing_tests)

    if sbfl is None:
        sbfl = base.SBFL()
    sbfl.fit(X, y)
    return sbfl.to_frame(index=cov_df.index)