import re
import numpy as np

import pandas as pd
from definitions import *
from module.almazov_dataset_processing.data_analysis import get_unique_symbols
from module.almazov_dataset_processing.data_files_processing import dates_processing


def time_to_my_format(time):
    try:
        # 5 symbols = 2 for hours + 1 for ':' + 2 for minutes
        while len(time) < 5:
            time = '0' + time
        # seconds will be ignored
        return time[:5]
    except:
        return np.nan


def code_to_my_format(code):
    # 7 symbols for patient + 3 for episode
    return code[:10]


def delete_nan_ids(data_frame: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
    data_frame = data_frame.dropna(subset=id_columns, how='any')
    return data_frame


def prepare_float_columns(data_frame: pd.DataFrame, float_columns: List[str]) -> pd.DataFrame:
    float_pattern = r'[-+]?\d*\.\d+|\d+'
    for column in float_columns:
        column_data = data_frame[column]
        func = lambda r: re.findall(float_pattern, str(r))
        column_data = column_data.apply(lambda r: func(r)[0] if len(func(r)) > 0 else np.nan)
        data_frame[column] = column_data

    return data_frame


def prepare_categorical_columns(data_frame: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    for column in categorical_columns:
        unique_symbols = get_unique_symbols(data_frame, column)
        data_frame[column] = data_frame[column].replace(unique_symbols)

    return data_frame


def prepare_date_columns(data_frame: pd.DataFrame, date_columns: List[str], date_format: str) -> pd.DataFrame:
    for column in date_columns:
        data_frame[column] = data_frame[column].map(lambda x: __get_my_format_date(x, date_format))
        data_frame = data_frame.loc[data_frame[column] != dates_processing.NULL_DATE]

    return data_frame


def __get_my_format_date(date: str, date_format:str):
    try:
        value = __prepare_date(date, date_format)
    except:
        value = dates_processing.NULL_DATETIME
    value = dates_processing.datetime_to_string(value, dates_processing.MY_DATE_FORMAT)
    return value


def __prepare_date(date_string, date_format):
    date_string = date_string.replace('.', '')
    date_string = date_string.replace(':', '')
    date_string = date_string.replace(' ', '')

    while len(date_string) < 12:
        date_string += '0'

    date_string = date_string[:12]
    date_time = dates_processing.datetime_from_string(date_string, date_format)

    return date_time


def delete_unused_columns(data_frame: pd.DataFrame, unused_columns: List[str]) -> pd.DataFrame:
    for column in unused_columns:
        if column in data_frame.columns:
            data_frame = data_frame.drop(column, axis=1)

    return data_frame


def prepare_columns(data_frame: pd.DataFrame, columns: Dict[str, List[str]], date_format: str) -> pd.DataFrame:
    print('### === Fields Preprocessing === ###')

    data_frame = prepare_float_columns(data_frame, columns['float_columns'])
    data_frame[columns['float_columns']] = data_frame[columns['float_columns']].astype(float)
    print('after prepare_float_columns {}'.format(data_frame.shape))

    data_frame = prepare_categorical_columns(data_frame, columns['categorical_columns'])
    data_frame[columns['categorical_columns']] = data_frame[columns['categorical_columns']].astype(float)
    print('after prepare_categorical_columns {}'.format(data_frame.shape))

    data_frame = prepare_date_columns(data_frame, columns['date_columns'], date_format)
    print('after prepare_date_columns {}'.format(data_frame.shape))

    data_frame = delete_unused_columns(data_frame, columns['unused_columns'])
    print('after delete_unused_columns {}'.format(data_frame.shape))

    return data_frame
