import pandas as pd

from definitions import *
from files_description import labresult_description
from module.almazov_dataset_processing.data_files_processing.columns_processing import prepare_columns, delete_nan_ids, code_to_my_format
from module.data_loader import read_data, write_data


def __first_read_labresult(file_name: str, id_columns: List[str], DEMO: bool = False) -> pd.DataFrame:
    data_frame = read_data(file_name, delimiter='\t')

    if DEMO:
        data_frame = data_frame.head(1000)
        write_data(data_frame, path_join(BASE_FILES, 'labresult_sample.csv'))

    data_frame = delete_nan_ids(data_frame, id_columns)
    data_frame['cod2'] = data_frame['cod2'].apply(code_to_my_format)
    return data_frame


def main(input_file: str, output_file: str, DEMO: bool = False) -> None:
    data = __first_read_labresult(input_file, labresult_description['id_columns'], DEMO)
    # print(get_nans_count(data, by_columns=True))
    data = prepare_columns(data, labresult_description, '%Y%m%d%H%M')
    # print(get_nans_count(data, by_columns=True))
    write_data(data, output_file)
    print('Labresult was processed')


if __name__ == '__main__':
    __DEMO = False
    output_file_name = 'labresult.csv'
    if __DEMO:
        output_file_name = 'labresult_sample.csv'

    main(path_join(BASE_FILES, FILE_NAMES[0]), path_join(CLEARED_DIR, output_file_name), __DEMO)
