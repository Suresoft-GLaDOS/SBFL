import argparse
import pandas
from pathlib import Path
from sbfl.base import SBFL
from sbfl.utils import gcov_files_to_frame, get_sbfl_scores_from_frame


def _get_choices():
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
                        choices=_get_choices(),
                        help='sbfl formula')
    parser.add_argument('dirs', nargs='+', help='directory to each search for gcov files')
    return parser.parse_args()


def _check_gcov_dirs(gcov_dirs):
    ret = True
    for d in gcov_dirs:
        if not Path(d).exists():
            print(f'{d} not exists.')
            ret = False
            continue
        if not Path(d).is_dir():
            print(f'{d} is not a directory.')
            ret = False
            continue
        glob_test_files = list(Path(d).glob('*.test'))
        if len(glob_test_files) != 1:
            print(f'{d} should have one and only test result file.')
            ret = False
            continue
    return ret


def _get_failing_tests(gcov_dirs):
    ret = []
    for d in gcov_dirs:
        test_file = next(Path(d).glob('*.test'), None)
        assert(test_file is not None)
        with open(test_file, 'r') as f:
            if f.read().rstrip() == 'failed':
                ret.append(Path(d))
    return ret


def main():
    args = _argparse()

    sbfl = SBFL(formula=args.formula[0])

    gcov_dirs = []
    for d in args.dirs:
        gcov_dirs.extend(Path('.').glob(d))
    _check_gcov_dirs(gcov_dirs) or exit(1)
    failing_tests = _get_failing_tests(gcov_dirs)

    gcov_files = {Path(d): list(Path(d).glob('*.gcov')) for d in gcov_dirs}
    if len(gcov_files) == 0:
        print('Gcov file not found.')
        exit(1)
    cov_df = gcov_files_to_frame(gcov_files, only_covered=True)

    sbfl_score = get_sbfl_scores_from_frame(cov_df, sbfl=sbfl, failing_tests=failing_tests)
    pandas.set_option('display.max_rows', sbfl_score.shape[0]+1)

    print(sbfl_score)


if __name__ == '__main__':
    main()

