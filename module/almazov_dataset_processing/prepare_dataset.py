from files_description import labresult_description
from module.data_loader import write_data

from module.almazov_dataset_processing.data_analysis import *
from module.almazov_dataset_processing.data_files_processing.dates_processing import *


def add_additional_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    result_frame = __add_event_timedelta(data_frame)
    result_frame = __add_intervention_timedelta(result_frame)
    result_frame = __add_duration_column(result_frame)
    result_frame = __add_additional_indicators(result_frame)
    result_frame['patient_id'] = result_frame['patient_id'].astype(int)
    result_frame = result_frame.sort_values(by=['patient_id', 'epizod_id', 'timedelta'])

    assert(data_frame.shape[0] == result_frame.shape[0])

    print('after add_new_columns : {}'.format(data_frame.shape))

    return result_frame


def __add_event_timedelta(data_frame: pd.DataFrame) -> pd.DataFrame:
    date_columns = ['start_date', 'event_date']
    timedelta_func = lambda x: (x[date_columns[1]] - x[date_columns[0]])

    result_frame = __add_timedelta_column(data_frame, date_columns, timedelta_func, 'timedelta')
    return result_frame


def __add_timedelta_column(dataset: pd.DataFrame, date_columns: List[str], timedelta_func, new_column_name: str) -> pd.DataFrame:
    new_frame = dataset.loc[:, date_columns]

    for column in date_columns:
        new_frame[column] = new_frame[column].map(lambda r: datetime_from_string(r, MY_DATE_FORMAT))

    timedelta_column = new_frame.apply(timedelta_func, axis=1)
    timedelta_column = timedelta_column.map(timedelta_to_hours)
    timedelta_column = pd.DataFrame(data=timedelta_column, columns=[new_column_name])

    print(timedelta_column.describe())

    deltas_df = timedelta_column.reset_index(drop=True)
    new_df = dataset.reset_index(drop=True)
    result = new_df.join(deltas_df)

    return result


def __add_intervention_timedelta(data_frame: pd.DataFrame) -> pd.DataFrame:
    date_columns = ['data', 'event_date']
    timedelta_func = lambda x: (x[date_columns[1]] - x[date_columns[0]])

    result_frame = __add_timedelta_column(data_frame, date_columns, timedelta_func, 'event_timedelta')
    result_frame = __preprocess_negative_numbers(result_frame)
    print(result_frame['event_timedelta'].describe())
    return result_frame


def __preprocess_negative_numbers(data_frame: pd.DataFrame) -> pd.DataFrame:
    print('### === Start method : fill_event_timedelta === ###')

    data_frame['event_timedelta'] = data_frame['event_timedelta'].astype(float)

    _mask = data_frame['data'] == NULL_DATE
    data_frame['activ'][_mask] = 0
    data_frame['event_timedelta'][_mask] = 0

    _mask = data_frame['event_timedelta'] < 0
    data_frame['activ'][_mask] = 0
    data_frame['event_timedelta'][_mask] = 0

    print('### === End method : fill_event_timedelta === ###')
    print('after fill_event_timedelta : {}'.format(data_frame.shape))
    return data_frame


def __add_duration_column(data_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = data_frame.groupby('patient_id')
    return grouped.apply(__add_duration_for_patient)


def __add_duration_for_patient(patient_frame):
    start_date = datetime_from_string(patient_frame['start_date'].values[0], MY_DATE_FORMAT)
    end_date = datetime_from_string(patient_frame['event_date'].values[-1], MY_DATE_FORMAT)
    duration = timedelta_to_hours(end_date - start_date)
    rows_count = patient_frame['event_date'].shape[0]
    durations = [duration] * rows_count
    patient_frame['duration'] = durations
    return patient_frame


def __add_additional_indicators(data_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = data_frame.groupby('patient_id')
    return grouped.apply(__add_indicators_for_patient)


def __add_indicators_for_patient(patient_frame: pd.DataFrame) -> pd.DataFrame:
    patient_frame = __calculate_indicator(patient_frame, 'HGB', 'Bleeding', '-0.1')
    patient_frame = __calculate_indicator(patient_frame, 'Kreatinin', 'Contrast-induced nephropathy', '0.2')
    patient_frame = __calculate_indicator(patient_frame, 'Glucose', 'Stress-induced hyperglycemia', '0.1')
    # group = get_indicator(group, 'CRP', 'systemic inflammatory response', '')     #TODO: Question about CRP
    patient_frame = __calculate_indicator(patient_frame, 'WBC', 'Systemic inflammatory response', '0.1')
    return patient_frame


def __calculate_indicator(group: pd.DataFrame, column: str, new_column: str, condition: str, period='all'):
    group[column] = group[column].astype(float)
    growth_condition = float(condition)
    RISE = growth_condition > 0
    _min, _max = __get_min_max(group, column)
    indicator = 0

    if not pd.isna(_min):
        min_index, max_index = __get_min_max_index(group, column, _min, _max)
        if RISE:
            if _max > (_min * (1 + growth_condition)) and (max_index > min_index):
                indicator = 1
        else:
            if _min < (_max * (1 + growth_condition)) and (min_index > max_index):
                indicator = 1

    rows_count = group[column].shape[0]
    indicators = [indicator] * rows_count
    group[new_column] = indicators
    return group


def __get_min_max(group, column):
    _mask = pd.isna(group[column])
    _min = group[column][~_mask].min()
    _max = group[column][~_mask].max()
    return _min, _max


def __get_min_max_index(group, column, min_value, max_value):
    _min_index = group[column].values.tolist().index(min_value)
    _max_index = group[column].values.tolist().index(max_value)
    return _min_index, _max_index


def remove_missing_values(data_frame: pd.DataFrame) -> pd.DataFrame:
    needed_columns = ['sex', 'age', 'sd', 'timedelta']
    target_columns = ['Troponin', 'RBC', 'WBC', 'HGB', 'HCT', 'PLT', 'AST', 'ALT', 'Kreatinin', 'Glucose', 'Holesterin',
                      'pressure', 'condition']  # TODO: Check if contains all columns

    data_frame[needed_columns] = data_frame[needed_columns].astype(float)
    data_frame[target_columns] = data_frame[target_columns].astype(float)

    data_frame[target_columns] = data_frame[target_columns].replace(0.0, np.nan)
    data_frame[target_columns] = data_frame[target_columns].replace(0, np.nan)

    data_frame = data_frame.dropna(subset=needed_columns)
    data_frame = data_frame.dropna(subset=target_columns, how='all')

    print('after remove_missing_values : {}'.format(data_frame.shape))
    return data_frame


def prepare_labresult_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    columns = ['Troponin', 'RBC', 'WBC', 'HGB', 'HCT', 'PLT', 'AST', 'ALT', 'Kreatinin', 'Glucose', 'Holesterin',
               'pressure', 'condition']

    print('before prepare thresholds : {}'.format(df.shape))
    df[columns] = df[columns].astype(float)

    df = __prepare_by_thresholds(df, 'timedelta', -0.001, 720.01, 'delete')
    df = __prepare_by_thresholds(df, 'age', 24.99, 95.01, 'delete')

    print('after td and age : {}'.format(df.shape))

    df = __prepare_by_thresholds(df, 'Troponin', 0.01, 50)    # TODO: config file for thresholds
    df = __prepare_by_thresholds(df, 'ALT', 0.2, 3000)
    df = __prepare_by_thresholds(df, 'AST', 1.4, 3000)
    df = __prepare_by_thresholds(df, 'HGB', 40, 214)
    df = __prepare_by_thresholds(df, 'RBC', 1, 9.8)
    df = __prepare_by_thresholds(df, 'PLT', 2, 700)
    df = __prepare_by_thresholds(df, 'Kreatinin', 18, 800)
    df = __prepare_by_thresholds(df, 'Glucose', 2, 30)
    print('after base columns : {}'.format(df.shape))

    target_columns = columns
    delete_columns = ['Troponin', 'ALT', 'AST', 'HGB', 'RBC', 'PLT', 'Kreatinin', 'Glucose']
    for column in delete_columns:
        target_columns.remove(column)

    for column in target_columns:
        lower, upper, count = get_percentile(df[column], [0, 97.5])
        df = __prepare_by_thresholds(df, column, lower, upper)

    print('after prepare_labresult_thresholds : {}'.format(df.shape))
    return df


def __prepare_by_thresholds(df: pd.DataFrame, column: str, min_value: float, max_value: float, method: str = 'setnan') -> pd.DataFrame:
    mask = (df[column] >= min_value) & (df[column] <= max_value) | (pd.isna(df[column]))
    if method == 'setnan':
        df.loc[:, column] = df[column].where(mask, other=np.nan)
    if method == 'delete':
        df = df.drop(df[~mask].index)
    return df


def clean_episode_dates(data_frame: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    start_date_field = date_columns[0]
    end_date_field = date_columns[1]
    event_date_field = date_columns[2]

    date_columns_frame = data_frame[date_columns]
    for column in date_columns:
        date_columns_frame.loc[:, column] = date_columns_frame[column].apply(lambda x: datetime_from_string(x, MY_DATE_FORMAT))

    # day = timedelta(days=1)
    # dates_frame.loc[:, start_date_field] = dates_frame[start_date_field].map(lambda x: x - day)
    # dates_frame.loc[:, end_date_field] = dates_frame[end_date_field].map(lambda x: x + day)

    in_range_start_end = lambda x: (x[event_date_field] >= x[start_date_field]) & (x[event_date_field] <= x[end_date_field])
    _mask = in_range_start_end(date_columns_frame)
    date_columns_frame = date_columns_frame[_mask]

    not_too_long_from_start = lambda x: (x[event_date_field] - x[start_date_field]) < timedelta(days=16)
    _mask = not_too_long_from_start(date_columns_frame)
    date_columns_frame = date_columns_frame[_mask]

    events = date_columns_frame[event_date_field].apply(lambda x: datetime_to_string(x, MY_DATE_FORMAT))

    result_frame = data_frame[data_frame[event_date_field].isin(events)]
    result_frame = result_frame.drop_duplicates(subset=['patient_id', event_date_field])
    result_frame['patient_id'] = result_frame['patient_id'].astype(int)
    result_frame = result_frame.sort_values(by=['patient_id', event_date_field])

    return result_frame


def main(using_columns, input_dir=MERGED_DIR, output_dir=FINAL_DIR):
    data_frame = read_data(os.path.join(input_dir, 'merged.csv'))
    data_frame = data_frame[using_columns]

    result_frame = add_additional_columns(data_frame)
    result_frame = remove_missing_values(result_frame)
    result_frame = prepare_labresult_thresholds(result_frame)
    result_frame = clean_episode_dates(result_frame, labresult_description['date_columns'])

    write_data(result_frame, os.path.join(output_dir, 'dataset.csv'))


if __name__ == '__main__':
    main()
    pass
