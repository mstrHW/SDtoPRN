import pandas as pd
import numpy as np

from definitions import *
from module.data_loader import read_data


def get_unique_symbols(data_frame: pd.DataFrame, column: str) -> Dict[str, str]:
    _uniq = data_frame.loc[:, column].unique()
    uniq = _uniq.tolist()
    symbols = dict()
    start_i = 0
    if np.nan in uniq:
        symbols[np.nan] = 0
        uniq.remove(np.nan)
        start_i = 1
    [symbols.update({word: (i+start_i)}) for i, word in enumerate(uniq)]
    return symbols


def try_to_convert_column(data_frame: pd.DataFrame, column: str):
    print(data_frame[column].describe())
    auok = data_frame[column]
    for line, i in zip(auok, range(auok.shape[0])):
        try:
            float(line)
        except:
            print('column {} : line {} : {}'.format(column, i, line))


def try_to_convert_frame(data_frame: pd.DataFrame, target_columns: List[str] = None):

    error_columns = []

    if not target_columns:
        target_columns = data_frame.columns

    for column in target_columns:
        try:
            data_frame[column].astype(float)
        except:
            error_columns.append(column)
            pass

    print('errors : {}'.format(len(error_columns)))

    for column in error_columns:
        current_column = column

        try_to_convert_column(data_frame, current_column)
        uniq = get_unique_symbols(data_frame, current_column)

        uniq_len = len(uniq)
        print(current_column)
        print(uniq_len)

        if uniq_len < 15:
            print(uniq)


def get_nans_count(data_frame: pd.DataFrame, columns: List[str] = None, by_columns: bool = False) -> int:
    nans_summ = 0
    target_columns = data_frame.columns.tolist()
    if columns:
        target_columns = columns

    if by_columns:
        nans_summ += data_frame[target_columns].isna().sum()
    else:
        nans_summ += data_frame[target_columns].isna().sum().sum()
    return nans_summ


def get_nans_percent(df: pd.DataFrame, columns: List[str] = None) -> float:
    target_columns = df.columns.tolist()
    if isinstance(columns, list):
        target_columns = columns
    if isinstance(columns, str):
        target_columns = [columns]
    nans_count = get_nans_count(df, target_columns)
    df_count = 0
    for column in target_columns:
        df_count += df[column].shape[0]
    return nans_count/df_count


def is_nan(df: pd.DataFrame) -> pd.DataFrame:
    return pd.isna(df)


def get_percentile(column: pd.DataFrame, percents) -> [float, float, int]:
    check = is_nan(column)
    _column = column[~check]
    lower, upper = np.percentile(_column, percents)
    return lower, upper, _column.count()


def get_range(column: pd.DataFrame) -> [float, float, int]:
    check = is_nan(column)
    _column = column[~check]
    lower, upper = _column.min(), _column.max()
    return lower, upper, _column.count()


if __name__ == '__main__':
    filled_file = path_join(FINAL_DIR, 'filled.csv')
    data_frame = read_data(filled_file)
    print(get_nans_count(data_frame, by_columns=True))
