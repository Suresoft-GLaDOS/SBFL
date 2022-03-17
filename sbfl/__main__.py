import argparse
import pandas
import json
import sys
from pathlib import Path
from sbfl.base import SBFL
from sbfl.utils import gcov_files_to_frame, get_sbfl_scores_from_frame


def _sbfl_formla():
    from inspect import getmembers, isfunction
    from sbfl import sbfl_formula
    return [x[0] for x in getmembers(sbfl_formula, isfunction)]


def _argparse():
    parser = argparse.ArgumentParser(prog='sbfl')
    parser.add_argument('-f',
                        '--formula',
                        dest='formula',
                        nargs=1,
                        type=str,
                        choices=_sbfl_formla(),
                        required=True,
                        help='sbfl formula')
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
                        '--information-out',
                        dest='info_out',
                        nargs=1,
                        type=str,
                        help='information about test data')
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
        gcov_dirs.extend(Path('.').glob(d))

    test_info = TestInformation(sbfl, gcov_dirs)
    sbfl_score = get_sbfl_scores_from_frame(test_info.cov_df,
                                            sbfl=test_info.sbfl,
                                            failing_tests=test_info.failing_tests)

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
        self.cov_df = gcov_files_to_frame(self.gcov_files, only_covered=True)

    def to_dict(self):
        return {
            'formula': self.sbfl.formula,
            'test': {
                'passing': [str(path) for path in self.passing_tests],
                'failing': [str(path) for path in self.failing_tests],
            },
            'sources': list(self.all_sources_set),
            'coverage': 50.32
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
