import errno
import pandas as pd

from files_description import *
from definitions import *

from module.data_loader import read_data, write_data
from module.almazov_dataset_processing.data_files_processing.dates_processing import NULL_DATE


def add_columns(dfs: List[pd.DataFrame], id_columns: List[List[str]], columns: List[str]):
    df1_columns = [column for column in dfs[0].columns.tolist() if column not in columns]
    _df1 = dfs[0][df1_columns]
    _df2 = dfs[1]
    result_columns = _df1.columns.tolist() + columns
    result_frame = pd.merge(_df1, _df2, left_on=id_columns[0], right_on=id_columns[1], how='left')
    result_frame = result_frame[result_columns]
    return result_frame


def __add_columns_from_predst(df1: pd.DataFrame, df2: pd.DataFrame, columns: List[str]) -> pd.DataFrame:   # TODO: rename method

    df1['cod2'] = df1['cod2'].apply(lambda r: r[:10])
    df2['id'] = df2['id'].apply(lambda r: r[:10])

    df3 = df1[df1.columns]
    id1_col = 'tmp_event_date'
    id2_col = 'tmp_date'
    df3[id1_col] = df1['event_date'].apply(lambda r: r[:8])
    df2[id2_col] = df2['date'].apply(lambda r: r[:8])

    dfs = list()
    dfs.append(df3)
    dfs.append(df2)

    ids = []
    df1_id = ['cod2', id1_col]
    df2_id = ['id', id2_col]
    ids.append(df1_id)
    ids.append(df2_id)

    columns.append(id2_col)
    new_df = add_columns(dfs, ids, columns)
    new_df = new_df.drop([id1_col], axis=1)

    return new_df


def add_columns_from_holesterin(data_frame: pd.DataFrame, input_dir: str) -> pd.DataFrame:

    df1_id = labresult_description['id_columns'][:2]
    df2_id = holesterin_description['id_columns']

    holesterin_file = os.path.join(input_dir, 'holesterin.csv')
    holesterin_data = read_data(holesterin_file)

    columns = holesterin_description['id_columns'] + holesterin_description['float_columns'] + holesterin_description['categorical_columns']
    # + holesterin_desc['id_columns']
    result_frame = add_columns([data_frame, holesterin_data], [df1_id, df2_id], columns)
    _columns = holesterin_description['float_columns'] + holesterin_description['categorical_columns']

    result_frame[_columns] = result_frame[_columns].astype(float)
    print('after merge holesterin {}'.format(result_frame.shape))

    return result_frame


def add_columns_from_predst(data_frame: pd.DataFrame, input_dir: str) -> pd.DataFrame:
    predst_file = os.path.join(input_dir, 'predst.csv')
    predst_data = read_data(predst_file)

    columns = predst_description['id_columns'] + predst_description['float_columns'] + predst_description['categorical_columns']
    result_frame = __add_columns_from_predst(data_frame, predst_data, columns)
    _columns = predst_description['float_columns'] + predst_description['categorical_columns']
    result_frame[_columns] = result_frame[_columns].astype(float)

    print('after merge predst {}'.format(result_frame.shape))
    return result_frame


def add_columns_from_events(data_frame: pd.DataFrame, input_dir: str) -> pd.DataFrame:
    events_file = os.path.join(input_dir, 'events.csv')
    events_data = read_data(events_file)

    df1_id = labresult_description['id_columns'][:2]
    df2_id = events_description['id_columns']

    result_frame = add_columns([data_frame, events_data], [df1_id, df2_id], ['activ', 'data'])
    result_frame['activ'] = result_frame['activ'].fillna(0)
    result_frame['data'] = result_frame['data'].fillna(NULL_DATE)

    print('after merge events {}'.format(result_frame.shape))
    return result_frame


def add_columns_from_eho(data_frame: pd.DataFrame, input_dir: str) -> pd.DataFrame:
    eho_file = os.path.join(input_dir, 'eho.csv')
    eho_data = read_data(eho_file)

    columns = eho_description['id_columns'] + eho_description['float_columns'] + eho_description['categorical_columns']
    result_frame = __add_columns_from_predst(data_frame, eho_data, columns)
    _columns = eho_description['float_columns'] + eho_description['categorical_columns']
    result_frame[_columns] = result_frame[_columns].astype(float)

    print('after merge eho {}'.format(result_frame.shape))
    return result_frame


def check_for_required_files(input_dir: str, required_files: List[str]) -> None:
    for file_name in required_files:
        absolute_path = path_join(input_dir, file_name)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), absolute_path)


def main(input_dir=CLEARED_DIR, output_dir=MERGED_DIR):
    required_files = [
        'labresult.csv',
        'holesterin.csv',
        'predst.csv',
        'events.csv',
        'eho.csv',
    ]
    check_for_required_files(input_dir, required_files)

    labresult_file = path_join(input_dir, 'labresult.csv')
    labresult_data = read_data(labresult_file)
    _columns = labresult_description['float_columns'] + labresult_description['categorical_columns']

    labresult_data[_columns] = labresult_data[_columns].astype(float)

    print('### === Add columns from another files === ###')
    result_frame = add_columns_from_holesterin(labresult_data, input_dir)
    result_frame = add_columns_from_predst(result_frame, input_dir)
    result_frame = add_columns_from_events(result_frame, input_dir)
    result_frame = add_columns_from_eho(result_frame, input_dir)

    write_data(result_frame, os.path.join(output_dir, 'merged.csv'))


if __name__ == '__main__':
    main()
    pass
