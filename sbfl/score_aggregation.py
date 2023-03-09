import pandas as pd
from scipy.stats import rankdata

def _check_aggregation_input(score_df, level, column):
    if not isinstance(level, (list, tuple)):
        level = [level]
    for l in level:
        if l not in score_df.index.names:
            raise Exception(f"Level '{l}' is not a valid index level. (level should be one of {score_df.index.names})")
    if column not in score_df.columns:
        raise Exception(f"Column '{column}' is not in the input dataframe")
    return level

def max_aggregation(score_df, level, column='score'):
    level = _check_aggregation_input(score_df, level, column)
    return score_df.groupby(level=level).max()[[column]]

def mean_aggregation(score_df, level, column='score'):
    level = _check_aggregation_input(score_df, level, column)
    return score_df.groupby(level=level).mean()[[column]]

def max_rank_based_voting(score_df, level, column='score'):
    """
    The amount of vote each program component
    (corresponding to a row in `score_df`) cast is defined by
    the inverse of the max rank of its suspiciousness score,
    which is stored in the column `column` of `score_df`.
    Then, the total votes are aggregated for the program components
    at the given `level`.

    - See Section V.A.1 in https://coinse.kaist.ac.kr/publications/pdfs/Sohn2021ea.pdf for more details about this voting scheme
    """
    level = _check_aggregation_input(score_df, level, column)
    vote_df = pd.DataFrame(
        data=1/rankdata(-score_df[column], axis=0, method='max'),
        index=score_df.index,
        columns=[column]
    )
    return vote_df.groupby(level=level).sum()

def dense_rank_based_voting(score_df, level, column='score'):
    """
    The amount of vote each program component
    (corresponding to a row in `score_df`) cast is defined by
    the inverse of the dense rank of its suspiciousness score,
    which is stored in the column `column` of `score_df`.
    Then, the total votes are aggregated for the program components
    at the given `level`.

    - See Section V.A.1 in https://coinse.kaist.ac.kr/publications/pdfs/Sohn2021ea.pdf for more details about this voting scheme
    """
    level = _check_aggregation_input(score_df, level, column)
    vote_df = pd.DataFrame(
        data=1/rankdata(-score_df[column], axis=0, method='dense'),
        index=score_df.index,
        columns=[column]
    )
    return vote_df.groupby(level=level).sum()

def dense_rank_based_tie_aware_voting(score_df, level,
    column='score'):
    """
    The amount of vote each program component
    (corresponding to a row in `score_df`) cast is defined by
    the inverse of the dense rank of the score multiplied by
    the number of components tied with the program component.
    Then, the total votes are aggregated for the program components
    at the given `level`.

    - See Section V.A.2 in https://coinse.kaist.ac.kr/publications/pdfs/Sohn2021ea.pdf for more details about this voting scheme
    """
    level = _check_aggregation_input(score_df, level, column)
    vote_df = pd.DataFrame(
        data=rankdata(-score_df[column], axis=0, method='dense'),
        index=score_df.index,
        columns=[column]
    )
    ties = vote_df[column].value_counts()
    vote_df[column] = vote_df[column].apply(lambda rank: 1/(rank * ties[rank]))
    return vote_df.groupby(level=level).sum()

def min_rank_based_voting(score_df, level, column='score'):
    """
    The amount of vote each program component
    (corresponding to a row in `score_df`) cast is defined by
    the inverse of the minimum rank of its suspiciousness score,
    which is stored in the column `column` of `score_df`.
    Then, the total votes are aggregated for the program components
    at the given `level`.

    - See Section V.A.3 in https://coinse.kaist.ac.kr/publications/pdfs/Sohn2021ea.pdf for more details about this voting scheme
    """
    level = _check_aggregation_input(score_df, level, column)
    vote_df = pd.DataFrame(
        data=1/rankdata(-score_df[column], axis=0, method='min'),
        index=score_df.index,
        columns=[column]
    )
    return vote_df.groupby(level=level).sum()

def dense_rank_based_suspiciousness_aware_voting(score_df, level,
    column='score'):
    """
    The amount of vote each program component
    (corresponding to a row in `score_df`) cast is defined by
    its suspiciousness score (in `column`) over the dense rank of the score.
    Then, the total votes are aggregated for the program components
    at the given `level`.

    - See Section V.A.4 in https://coinse.kaist.ac.kr/publications/pdfs/Sohn2021ea.pdf for more details about this voting scheme
    """
    level = _check_aggregation_input(score_df, level, column)
    ranks = rankdata(-score_df[column], axis=0, method='dense')
    vote_df = pd.DataFrame(
        data=score_df[column]/ranks,
        index=score_df.index,
        columns=[column]
    )
    return vote_df.groupby(by=level).sum()

