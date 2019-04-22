import pandas as pd

from definitions import *
from files_description import holesterin_description
from module.almazov_dataset_processing.data_files_processing.columns_processing import prepare_columns, delete_nan_ids, time_to_my_format
from module.data_loader import read_data, write_data


def __first_read_holesterin(file_name: str, id_columns: List[str], DEMO: bool = False) -> pd.DataFrame:
    data_frame = read_data(file_name, delimiter=';')

    if DEMO:
        data_frame = data_frame.head(1000)
        write_data(data_frame, path_join(BASE_FILES, 'events_sample.csv'))

    data_frame = delete_nan_ids(data_frame, id_columns)
    data_frame['timev -'] = data_frame['timev -'].apply(time_to_my_format)
    data_frame['datep -'] += data_frame['timev -']
    data_frame['исход  -'][(data_frame['исход  -'] != 'Выписан') & (data_frame['исход  -'] != 'Умер')] = 'Выписан'
    # data_frame[(data_frame['sex'] == 'Выписан')]['sex'] = np.nan
    data_frame = data_frame.drop('timev -', axis=1)
    return data_frame


def __is_word_in_row(row, word):
    if ~pd.isna(row):
        return word in str(row)
    else:
        return False


def __prepare_holesterin_diagnosis(data):
    column = 'diagstac'
    data[column] = data[column].apply(lambda r: 'окс' if __is_word_in_row(r, 'I20') else r)
    data[column] = data[column].apply(lambda r: 'окс2' if __is_word_in_row(r, 'I21') else r)
    data[column] = data[column].apply(lambda r: 'окс3' if __is_word_in_row(r, 'I22') else r)
    data.loc[:, column] = data[(data[column] == 'окс') | (data[column] == 'окс2') | (data[column] == 'окс3')][column]
    data = __diag_fact(data)
    return data


def __diag_fact(data_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = data_frame.groupby('pacient_id -').apply(__add_diag_columns)
    return grouped


def __add_diag_columns(group: pd.DataFrame) -> pd.DataFrame:
    diag = group['diagstac'].values[0]

    group['I20'] = [0] * group['diagstac'].shape[0]
    group['I21'] = [0] * group['diagstac'].shape[0]
    group['I22'] = [0] * group['diagstac'].shape[0]
    if diag == 'окс':
        group.loc[:, 'I20'] = 1
    if diag == 'окс2':
        group.loc[:, 'I21'] = 1
    if diag == 'окс3':
        group.loc[:, 'I22'] = 1
    return group


def main(input_file: str, output_file: str) -> None:
    data = __first_read_holesterin(input_file, holesterin_description['id_columns'])
    data = __prepare_holesterin_diagnosis(data)

    data = prepare_columns(data, holesterin_description, '%d%m%Y%H%M')
    write_data(data, output_file)
    print('Holesterin was processed')


if __name__ == '__main__':
    __DEMO = False
    output_file_name = 'holesterin.csv'
    if __DEMO:
        output_file_name = 'holesterin_sample.csv'

    main(path_join(BASE_FILES, FILE_NAMES[1]), path_join(CLEARED_DIR, output_file_name))
