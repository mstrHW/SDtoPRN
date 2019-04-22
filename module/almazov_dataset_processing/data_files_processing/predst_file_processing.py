import numpy as np
import pandas as pd

from definitions import *
from files_description import predst_description
from module.almazov_dataset_processing.data_files_processing.columns_processing import prepare_columns, delete_nan_ids, time_to_my_format, code_to_my_format
from module.data_loader import write_data


def __first_read_predst(file_name: str, id_columns: List[str], DEMO: bool = False) -> pd.DataFrame:
    data_frame = __first_read_data_predst(file_name, encoding='UTF-8', delimiter='\t')

    if DEMO:
        data_frame = data_frame.head(1000)
        write_data(data_frame, path_join(BASE_FILES, 'predst_sample.csv'))

    data_frame = delete_nan_ids(data_frame, id_columns)
    data_frame['time'] = data_frame['time'].apply(time_to_my_format)
    data_frame['date'] += data_frame['time']
    data_frame['id'] = data_frame['id'].apply(code_to_my_format)
    data_frame = data_frame.drop('time', axis=1)
    return data_frame


def __first_read_data_predst(file_name: str, encoding: str = 'CP1251', delimiter: str = ';') -> pd.DataFrame:
    data = []
    in_file = open(file_name, 'r', encoding=encoding)
    columns = in_file.readline()
    columns = columns.replace('\n', '').split(delimiter)
    print(columns)
    columns[0] = columns[0][1:]
    for row in in_file:
        row = row.replace('\n', '')
        csv_row = row.split(delimiter)
        data.append(csv_row)
    in_file.close()
    df = pd.DataFrame(data=data, columns=columns)
    df = df.replace(r'', 'NA', regex=False)
    df = df.replace('NA', np.nan)
    return df


def __prepare_predst_conditions(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame.fillna(value=np.nan, inplace=True) # For None type
    uniq = data_frame.loc[:, 'condition'].unique().tolist()

    if np.nan in uniq:
        uniq.remove(np.nan)

    thefile = open('uniq_conditions.txt', 'w')
    for item in uniq:
        thefile.write("%s\n" % item)
    thefile.close()

    condition_map = dict()
    for i, value in enumerate(uniq):
        try:
            condition_map[value] = __standardize_condition_predst(value)
            # print('{} : {}'.format(i, value))
        except:
            print('Exception at ' + value)
    data_frame['condition'] = data_frame['condition'].map(condition_map)
    # print('End of prepare conditions method')
    return data_frame


def __standardize_condition_predst(key: str) -> str:
    value = np.nan
    if 'средней' in key or ('отн' in key and 'удовлетв' in key) or 'компенсированное' in key:
        value = 'относительно_удовлетворительное'
    elif (('крайне' in key or 'очень' in key) and 'тяжелое' in key) or 'агональное' in key or 'терминальное' in key or \
                    'с_отрицательной_динамикой' in key or 'критическое' in key:
        value = 'критическое'
    elif 'тяжелое' in key or ' нестабильное' in key:
        value = 'тяжелое'
    elif 'удовлетвор' in key or 'соответствует' in key or 'стабильное' in key or 'с_положительной_динамикой' in key or 'лежит на полу' in key:
        value = 'удовлетворительное'

    return value


def main(input_file: str, output_file: str) -> None:
    data = __first_read_predst(input_file, predst_description['id_columns'])
    data = __prepare_predst_conditions(data)
    data = prepare_columns(data, predst_description, '%Y%m%d%H%M')
    write_data(data, output_file)
    print('Predst was processed')


if __name__ == '__main__':
    __DEMO = False
    output_file_name = 'predst.csv'
    if __DEMO:
        output_file_name = 'predst_sample.csv'

    main(path_join(BASE_FILES, FILE_NAMES[2]), path_join(CLEARED_DIR, output_file_name))
