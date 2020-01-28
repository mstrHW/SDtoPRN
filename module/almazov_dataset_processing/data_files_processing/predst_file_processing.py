import numpy as np
import pandas as pd

from definitions import *
from files_description import predst_description
from module.almazov_dataset_processing.data_files_processing.columns_processing import prepare_columns, delete_nan_ids, time_to_my_format, code_to_my_format
from module.data_loader import write_to_csv


def __read_raw_predst_file(file_name: str, DEMO: bool = False) -> pd.DataFrame:
    encoding = 'UTF-8'
    delimiter = '\t'

    data = []
    in_file = open(file_name, 'r', encoding=encoding)
    columns = in_file.readline()
    columns = columns.replace('\n', '').split(delimiter)

    logging.info('Predst columns: {}'.format(columns))

    columns[0] = columns[0][1:]
    for row in in_file:
        row = row.replace('\n', '')
        csv_row = row.split(delimiter)
        data.append(csv_row)
    in_file.close()
    df = pd.DataFrame(data=data, columns=columns)
    df = df.replace(r'', 'NA', regex=False)
    df = df.replace('NA', np.nan)

    if DEMO:
        data_frame = df.head(1000)
        write_to_csv(data_frame, path_join(BASE_FILES, 'predst_sample.csv'))

    return df


def __prepare_predst_conditions(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame.fillna(value=np.nan, inplace=True) # For None type
    uniq = data_frame.loc[:, 'condition'].unique().tolist()

    if np.nan in uniq:
        uniq.remove(np.nan)

    # thefile = open('uniq_conditions.txt', 'w')
    # for item in uniq:
    #     thefile.write("%s\n" % item)
    # thefile.close()

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
    logging.info('Start processing predst file')
    data = __read_raw_predst_file(input_file)

    data = delete_nan_ids(data, predst_description['id_columns'])

    data['time'] = data['time'].apply(time_to_my_format)
    data['date'] += data['time']
    data['id'] = data['id'].apply(code_to_my_format)

    data = data.drop('time', axis=1)

    data = __prepare_predst_conditions(data)
    data = prepare_columns(data, predst_description, '%Y%m%d%H%M')
    write_to_csv(data, output_file)

    logging.info('Predst was processed')


if __name__ == '__main__':
    __DEMO = False
    output_file_name = 'predst.csv'
    if __DEMO:
        output_file_name = 'predst_sample.csv'

    main(path_join(BASE_FILES, FILE_NAMES[2]), path_join(CLEARED_DIR, output_file_name))
