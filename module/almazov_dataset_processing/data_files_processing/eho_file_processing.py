import numpy as np
import pandas as pd

from definitions import *
from files_description import eho_description
from module.almazov_dataset_processing.data_files_processing.columns_processing import prepare_columns, delete_nan_ids, time_to_my_format, code_to_my_format
from module.data_loader import read_data, write_to_csv


def __read_raw_eho_file(file_name: str, id_columns: List[str], DEMO: bool = False) -> pd.DataFrame:
    data_frame = read_data(file_name, delimiter='\t')

    if DEMO:
        sample = data_frame.head(20)
        data_frame = sample
        write_to_csv(sample, path_join(BASE_FILES, 'eho_sample.csv'))

    print('before preprocessing {}'.format(data_frame.shape))
    data_frame = delete_nan_ids(data_frame, id_columns)
    print('after drop nan ids {}'.format(data_frame.shape))

    data_frame['time'] = data_frame['time'].apply(time_to_my_format)
    data_frame['date'] += data_frame['time']
    data_frame['id'] = data_frame['id'].apply(code_to_my_format)
    data_frame = data_frame.drop('time', axis=1)
    return data_frame


def __is_word_in_row(row, word):
    if ~pd.isna(row):
        return word in str(row)
    else:
        return False


def __prepare_eho_categorical_columns(data: pd.DataFrame) -> pd.DataFrame:
    column = 'Mkin Кинетика '
    data[column] = data[column].apply(lambda r: 'нарушена' if __is_word_in_row(r, 'нарушен') else r)
    data[(data[column] != 'нарушена') & (data[column] != 'не_изменена')
         & (data[column] != 'диффузная_гипокинезия')].loc[:, column] = np.nan

    column = 'Mrit ритм '
    data[column] = data[column].apply(lambda r: 'синусовый' if __is_word_in_row(r, 'синусов') else r)
    data[(data[column] != 'синусовый') & (data[column] != 'фибрилляция предсердий')
         & (data[column] != 'ЭКС')].loc[:, column] = np.nan

    column = 'Mst состояние аорты '
    data[column] = data[column].apply(lambda r: 'уплотнены' if __is_word_in_row(r, 'уплотнен') else r)
    data[(data[column] != 'не_изменены') & (data[column] != 'уплотнены')].loc[:, column] = np.nan

    column = 'MLpo МПП '
    data[column] = data[column].apply(lambda r: 'аневризма' if __is_word_in_row(r, 'аневризма') else r)
    data[(data[column] != 'не_изменена') & (data[column] != 'аневризма')].loc[:, column] = np.nan

    column = 'Makstv Аортальный клапан створки '
    data[column] = data[column].apply(lambda r: 'кальциноз' if __is_word_in_row(r, 'кальцинир') else r)
    data[column] = data[column].apply(lambda r: 'уплотнены' if __is_word_in_row(r, 'уплотнен') else r)
    data[column] = data[column].apply(lambda r: 'протез' if __is_word_in_row(r, 'протез') else r)
    data[(data[column] != 'не_изменены') & (data[column] != 'кальциноз')
         & (data[column] != 'протез') & (data[column] != 'уплотнены')].loc[:, column] = np.nan

    column = 'Makregur Аортальный клапан регургитация '
    data[column] = data[column].apply(lambda r: '1_степени' if __is_word_in_row(r, '0-1') else r)
    data[column] = data[column].apply(lambda r: '2_степени' if __is_word_in_row(r, '1-2') else r)
    data[column] = data[column].apply(lambda r: '3_степени' if __is_word_in_row(r, '2-3') else r)
    data[column] = data[column].apply(lambda r: '4_степени' if __is_word_in_row(r, '3-4') else r)
    data[(data[column] != 'отсутствует') & (data[column] != '1_степени') &
         (data[column] != '2_степени') & (data[column] != '3_степени') &
         (data[column] != '4_степени') & (data[column] != 'приклапанная')].loc[:, column] = np.nan

    column = 'mkregur  митральный клапан  регургитация '
    data[column] = data[column].apply(lambda r: '1_степени' if __is_word_in_row(r, '0-1') else r)
    data[column] = data[column].apply(lambda r: '2_степени' if __is_word_in_row(r, '1-2') else r)
    data[column] = data[column].apply(lambda r: '3_степени' if __is_word_in_row(r, '2-3') else r)
    data[column] = data[column].apply(lambda r: '4_степени' if __is_word_in_row(r, '3-4') else r)
    data[(data[column] != 'отсутствует') & (data[column] != '1_степени') &
         (data[column] != '2_степени') & (data[column] != '3_степени') &
         (data[column] != '4_степени') & (data[column] != 'приклапанная')].loc[:, column] = np.nan

    column = 'mkstv митральный клапан створки'
    data[column] = data[column].apply(lambda r: 'кальциноз' if __is_word_in_row(r, 'кальцинир') else r)
    data[column] = data[column].apply(lambda r: 'уплотнены' if __is_word_in_row(r, 'уплотнен') else r)
    data[column] = data[column].apply(lambda r: 'протез' if __is_word_in_row(r, 'протез') else r)
    data[(data[column] != 'не_изменены') & (data[column] != 'кальциноз')
         & (data[column] != 'протез') & (data[column] != 'уплотнены')].loc[:, column] = np.nan

    column = 'Mtkregur трикуспидальный  клапан  регургитация '
    data[column] = data[column].apply(lambda r: '1_степени' if __is_word_in_row(r, '0-1') else r)
    data[column] = data[column].apply(lambda r: '2_степени' if __is_word_in_row(r, '1-2') else r)
    data[column] = data[column].apply(lambda r: '3_степени' if __is_word_in_row(r, '2-3') else r)
    data[column] = data[column].apply(lambda r: '4_степени' if __is_word_in_row(r, '3-4') else r)
    data[(data[column] != 'отсутствует') & (data[column] != '1_степени') &
         (data[column] != '2_степени') & (data[column] != '3_степени') &
         (data[column] != '4_степени') & (data[column] != 'приклапанная')].loc[:, column] = np.nan

    column = 'Mtkstv  трикуспидальный  клапан створки'
    data[column] = data[column].apply(lambda r: 'пролапс' if __is_word_in_row(r, 'пролапс') else r)
    data[column] = data[column].apply(lambda r: 'уплотнены' if __is_word_in_row(r, 'уплотнен') else r)
    data[column] = data[column].apply(lambda r: 'протез' if __is_word_in_row(r, 'протез') else r)
    data[(data[column] != 'не_изменены') & (data[column] != 'пролапс') &
         (data[column] != 'уплотнены')].loc[:, column] = np.nan

    column = 'Mpkreg пульмональный  клапан  регургитация '
    data[column] = data[column].apply(lambda r: '1_степени' if __is_word_in_row(r, '0-1') else r)
    data[column] = data[column].apply(lambda r: '2_степени' if __is_word_in_row(r, '1-2') else r)
    data[column] = data[column].apply(lambda r: '3_степени' if __is_word_in_row(r, '2-3') else r)
    data[column] = data[column].apply(lambda r: '4_степени' if __is_word_in_row(r, '3-4') else r)
    data[(data[column] != 'отсутствует') & (data[column] != '1_степени') &
         (data[column] != '2_степени') & (data[column] != '3_степени') &
         (data[column] != '4_степени') & (data[column] != 'приклапанная')].loc[:, column] = np.nan

    return data


def main(input_file: str, output_file: str) -> None:
    logging.info('Start processing events file')
    data = __read_raw_eho_file(input_file, eho_description['id_columns'])
    data = __prepare_eho_categorical_columns(data)

    data = prepare_columns(data, eho_description, '%Y%m%d%H%M')
    write_to_csv(data, output_file)
    logging.info('Eho was processed')


if __name__ == '__main__':
    __DEMO = False
    output_file_name = 'eho.csv'
    if __DEMO:
        output_file_name = 'eho_sample.csv'

    main(path_join(BASE_FILES, FILE_NAMES[4]), path_join(CLEARED_DIR, output_file_name))
