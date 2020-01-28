import numpy as np

from definitions import *
from module.data_loader import read_data, write_to_csv
import module.almazov_dataset_processing.data_files_processing as cf
from module.almazov_dataset_processing import convert_to_periodic_data, merge_files, prepare_dataset, filling_methods, generate_data_for_rnn


def __clean_files() -> None:
    input_dir = BASE_FILES
    files = FILE_NAMES
    output_dir = CLEARED_DIR

    # cf.labresult_file_processing.main(path_join(input_dir, files[0]), path_join(output_dir, 'labresult.csv'))
    # cf.holesterin_file_processing.main(path_join(input_dir, files[1]), path_join(output_dir, 'holesterin.csv'))
    cf.predst_file_processing.main(path_join(input_dir, files[2]), path_join(output_dir, 'predst.csv'))
    cf.events_file_processing.main(path_join(input_dir, files[3]), path_join(output_dir, 'events.csv'))
    cf.eho_file_processing.main(path_join(input_dir, files[4]), path_join(output_dir, 'eho.csv'))


def __fill_missing_data(target_columns: Dict[str, List]) -> None:
    input_file = path_join(FINAL_DIR, 'dataset.csv')
    output_file = path_join(FINAL_DIR, 'filled.csv')

    data_frame = read_data(input_file)
    data_frame[target_columns['continuous']] = data_frame[target_columns['continuous']].astype(float)
    data_frame[target_columns['categorical']] = data_frame[target_columns['categorical']].astype(float)
    result_frame = filling_methods.fill_by_groups(data_frame, target_columns['continuous'], target_columns['categorical'])
    write_to_csv(result_frame, output_file)


def __generate_periodic_data(using_columns: List[str], period: int = 6, method: str = 'nearest') -> None:
    input_file = path_join(FINAL_DIR, 'filled.csv')
    output_file = path_join(FINAL_DIR, 'periodic_{}H_{}.csv'.format(period, method))

    data_frame = read_data(input_file)

    result_frame = convert_to_periodic_data.__remove_patients_without_tracks(data_frame)
    result_frame = convert_to_periodic_data.generate_periodic_data(result_frame, using_columns, period, method)

    write_to_csv(result_frame, output_file)


def __create_dataset(using_columns: List[str], period: int = 12, method: str = 'time', need_scale: bool = False) -> None:
    periodic_file = path_join(FINAL_DIR, 'periodic_{}H_{}.csv'.format(_period, _method))
    data_frame = read_data(periodic_file)
    X, Y, X_grouped, Y_grouped = generate_data_for_rnn.create_dataset(data_frame, using_columns, need_scale)

    mask = str(period) + method

    directory = os.path.join(DATASETS_DIR, mask, 'Base')
    if need_scale:
        directory = os.path.join(DATASETS_DIR, mask, 'Scalled')

    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(path_join(directory, 'names.npy'), using_columns)
    np.save(path_join(directory, 'X.npy'), X.values)
    np.save(path_join(directory, 'Y.npy'), Y.values)

    np.save(path_join(directory, 'X_grouped.npy'), X_grouped)
    np.save(path_join(directory, 'Y_grouped.npy'), Y_grouped)


if __name__ == '__main__':
    # _demo = False
    #
    # _period = 12
    # _method = 'time'
    # _need_scale = False
    #
    # __clean_files()
    # merge_files.main()
    #
    continuous_columns = [
        'Troponin',
        'RBC',
        'WBC',
        'HGB',
        'HCT',
        'PLT',
        'AST',
        'ALT',
        'Kreatinin',
        'Glucose',
        'Holesterin',
        'pressure',
    ]

    categorical_columns = [
        'condition',
    ]

    using_columns = [
        'patient_id',
        'epizod_id',
        'start_date',
        'event_date',
        'end_date',
        'data',
        'sex',
        'age',
        'sd',
        'activ',
        'result',
        ] + continuous_columns + categorical_columns

    prepare_dataset.main(using_columns)
    __fill_missing_data(filling_methods.get_target_columns(continuous_columns, categorical_columns))

    added_columns = [
        'event_timedelta',
        'Bleeding',
        'Contrast-induced nephropathy',
        'Stress-induced hyperglycemia',
        'Systemic inflammatory response',
        'duration',
        'timedelta',
    ]

    using_columns = continuous_columns + categorical_columns + added_columns

    __generate_periodic_data(using_columns, _period, _method)

    using_columns = [
        'patient_id',
        'activ',
    ] + continuous_columns + categorical_columns

    __create_dataset(using_columns, _period, _method, _need_scale)

    pass
