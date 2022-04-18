import argparse
import pandas
import json
import sys
import glob
from pathlib import Path
from collections import defaultdict
from sbfl.base import SBFL
from failure_clustering.failure_clustering.base import FailureDistance
from failure_clustering.failure_clustering.clustering import Agglomerative
from sbfl.utils import gcov_files_to_frame, get_sbfl_scores_from_frame, sbfl_formula_list, \
    get_coverage_info_from_frame, get_X_y

def _argparse():
    parser = argparse.ArgumentParser(prog='sbfl')
    parser.add_argument('-f',
                        '--formula',
                        dest='formula',
                        nargs=1,
                        type=str,
                        choices=sbfl_formula_list(),
                        required=True,
                        help='sbfl formula')
    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        help='verbose output')
    parser.add_argument('dirs',
                        nargs='+',
                        help='gcov data directories')
    parser.add_argument('-s',
                        '--sbfl-out',
                        dest='sbfl_out',
                        nargs=1,
                        type=str,
                        help='sbfl data in json format')
    parser.add_argument('-i',
                        '--info-out',
                        dest='info_out',
                        nargs=1,
                        type=str,
                        help='information about test data')
    parser.add_argument('-c',
                        '--cluster-out',
                        dest='cluster_out',
                        nargs=1,
                        type=str,
                        help='cluster data in json format')
    return parser.parse_args()


def _write_sbfl_output(sbfl_score, sbfl_out_path):
    sbfl_score.reset_index()
    sbfl_list = [[index[0], index[1], row['score']] for index, row in sbfl_score.iterrows()]
    with open(sbfl_out_path, 'w') as f:
        json.dump(sbfl_list, f, indent=4)


def _write_info_out(test_info, info_out_path):
    with open(info_out_path, 'w') as f:
        json.dump(test_info, f, indent=4)


def main():
    args = _argparse()

    sbfl = SBFL(formula=args.formula[0])

    gcov_dirs = []
    for d in args.dirs:
        gcov_dirs.extend(glob.glob(d))

    test_info = TestInformation(sbfl, gcov_dirs, verbose=args.verbose)
    sbfl_score = get_sbfl_scores_from_frame(test_info.cov_df,
                                            sbfl=test_info.sbfl,
                                            failing_tests=test_info.failing_tests)

    if args.cluster_out is not None:
        X, y = get_X_y(test_info.cov_df, failing_tests=test_info.failing_tests)
        fd = FailureDistance(measure='hdist')
        distance_matrix, failure_indices = fd.get_distance_matrix(
            X, y, weights=sbfl.fit_predict(X, y), return_index=True
        )
        aggl = Agglomerative(linkage='complete')
        clustering = aggl.run(distance_matrix,
                              stopping_criterion='min_intercluster_distance_elbow')
        # for i, cluster in zip(failure_indices, clustering):
        #     print(f"Cluster of {gcov_dirs[i]}: {cluster}")
        cluster_map = defaultdict(list)
        for i, cluster in zip(failure_indices, clustering):
            cluster_map[str(cluster)].append(gcov_dirs[i])
        with open(args.cluster_out[0], 'w') as f:
            json.dump(dict(cluster_map), f, indent=4)

    if args.info_out is not None:
        _write_info_out(test_info.to_dict(), args.info_out[0])

    if args.sbfl_out is not None:
        _write_sbfl_output(sbfl_score, args.sbfl_out[0])
    else:
        pandas.set_option('display.max_rows', sbfl_score.shape[0] + 1)
        print(sbfl_score)


class TestInformation:
    __test__ = False

    def __init__(self, sbfl, gcov_dirs, verbose=False):
        self.sbfl = sbfl
        self.gcov_dirs = gcov_dirs
        if not self._check_gcov_dirs():
            raise ValueError("Invalid gcov dirs.")
        self.gcov_files = {Path(d): list(Path(d).glob('*.gcov')) for d in gcov_dirs}
        if len(self.gcov_files) == 0:
            raise ValueError("No gcov files found.")
        self.all_sources_set = self._all_sources_set()
        self.passing_tests, self.failing_tests = self._classify_tests()
        self.cov_df = gcov_files_to_frame(self.gcov_files, only_coverable=True, verbose=verbose)
        self.covered_lines, self.total_lines = get_coverage_info_from_frame(self.cov_df)

    def to_dict(self):
        return {
            'formula': self.sbfl.formula,
            'test': {
                'passing': [str(path) for path in self.passing_tests],
                'failing': [str(path) for path in self.failing_tests],
            },
            'sources': list(self.all_sources_set),
            'coverage': f'{(self.covered_lines / self.total_lines)*100:.2f}'
        }

    def _all_sources_set(self):
        ret = set()
        for gcov_files in self.gcov_files.values():
            for gcov_file in gcov_files:
                ret.add(gcov_file.name)
        return ret

    def _check_gcov_dirs(self):
        ret = True
        if self.gcov_dirs is None or self.gcov_dirs == []:
            print(f'No gcov dirs are matched.', file=sys.stderr)
            return False
        for d in self.gcov_dirs:
            if not Path(d).exists():
                print(f'{d} not exists.')
                ret = False
                continue
            if not Path(d).is_dir():
                print(f'{d} is not a directory.', file=sys.stderr)
                ret = False
                continue
            glob_test_files = list(Path(d).glob('*.test'))
            if len(glob_test_files) != 1:
                print(f'{d} should have one and only test result file.', file=sys.stderr)
                ret = False
                continue
        return ret

    def _classify_tests(self):
        passing_tests, failing_tests = [], []
        for d in self.gcov_dirs:
            test_file = next(Path(d).glob('*.test'), None)
            assert test_file is not None, f'{d} has no *.test file.'
            with open(test_file, 'r') as f:
                result = f.read().rstrip()
                if result.lower() in ['failing', 'failed']:
                    failing_tests.append(Path(d))
                elif result.lower() in ['passing', 'passed']:
                    passing_tests.append(Path(d))
                else:
                    print(f'warning: {d} has no failing or passing test result.\n'
                          'Implicitly treat it as passed test.')
                    passing_tests.append(Path(d))
        return passing_tests, failing_tests


if __name__ == '__main__':
    main()
