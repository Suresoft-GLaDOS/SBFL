import argparse
import pandas
import json
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
                        required=True,
                        help='sbfl formula')
    parser.add_argument('dirs',
                        nargs='+',
                        help='gcov data directories')
    parser.add_argument('-j',
                        '--json-out',
                        dest='json_out',
                        nargs=1,
                        type=str,
                        help='json output')
    return parser.parse_args()


def _check_gcov_dirs(gcov_dirs):
    ret = True
    if gcov_dirs is None or gcov_dirs == []:
        print(f'No gcov dirs are matched.')
        return False
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


def _write_json_output(sbfl_score, json_path):
    sbfl_score.reset_index()
    sbfl_list = [[index[0], index[1], row['score']] for index, row in sbfl_score.iterrows()]
    with open(json_path, 'w') as f:
        json.dump(sbfl_list, f, indent=4)
    sbfl_json_list = json.dumps(sbfl_list,  indent=4)


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

    if args.json_out is not None:
        _write_json_output(sbfl_score, args.json_out[0])
    else:
        pandas.set_option('display.max_rows', sbfl_score.shape[0] + 1)
        print(sbfl_score)


if __name__ == '__main__':
    main()
