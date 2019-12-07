import pandas as pd

from definitions import *
from files_description import labresult_description
from module.almazov_dataset_processing.data_files_processing.columns_processing import prepare_columns, delete_nan_ids, code_to_my_format
from module.data_loader import read_data, write_to_csv


def __read_raw_labresult_file(file_name: str, DEMO: bool = False) -> pd.DataFrame:
    data_frame = read_data(file_name, delimiter='\t')

    if DEMO:
        data_frame = data_frame.head(1000)

        output_file = path_join(BASE_FILES, 'labresult_sample.csv')
        write_to_csv(data_frame, output_file)

        logging.info('Data was saved at {}'.format(output_file))

    return data_frame


def main(input_file: str, output_file: str, DEMO: bool = False) -> None:
    logging.info('Start processing labresult file')

    data = __read_raw_labresult_file(input_file, DEMO)

    # logging.debug('Nans count before preparing {}'.format(get_nans_count(data, by_columns=True)))
    data = delete_nan_ids(data, labresult_description['id_columns'])
    # logging.debug('Nans count after preparing {}'.format(get_nans_count(data, by_columns=True)))

    data['cod2'] = data['cod2'].apply(code_to_my_format)
    data = prepare_columns(data, labresult_description, '%Y%m%d%H%M')

    write_to_csv(data, output_file)
    logging.info('Labresult was processed')


if __name__ == '__main__':
    __DEMO = False
    output_file_name = 'labresult.csv'
    if __DEMO:
        output_file_name = 'labresult_sample.csv'

    main(path_join(BASE_FILES, FILE_NAMES[0]), path_join(CLEARED_DIR, output_file_name), __DEMO)
