import pandas as pd
import numpy as np

from definitions import *
from module.almazov_dataset_processing.data_files_processing import dates_processing


def __remove_patients_without_tracks(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.groupby(['patient_id', 'epizod_id']).filter(lambda group: group['patient_id'].count() > 1)


def generate_periodic_data(data_frame: pd.DataFrame, target_columns: List[str], period: int, method: str) -> pd.DataFrame:
    print('### === Start method : clean_dates.prepare_data_by_period === ###')
    np.random.seed(1234)

    data_frame[target_columns] = data_frame[target_columns].astype(float)

    new_frame = data_frame.groupby(['patient_id', 'epizod_id']).progress_apply(lambda x: prepare_dates(x, target_columns, period, method))

    print(new_frame.shape, data_frame.shape)

    print('### === End method : clean_dates.prepare_data_by_period === ###')
    return new_frame


def prepare_dates(group_frame: pd.DataFrame, target_columns: List[str], period: int, method: str = 'nearest') -> pd.DataFrame:
    values = group_frame.loc[:, group_frame.columns != 'event_date']
    timestamps = pd.to_datetime(group_frame['event_date'], format=dates_processing.MY_DATE_FORMAT)

    values.index = timestamps
    values[target_columns] = values[target_columns].astype(float)

    ts = interpolate_pd_series(values, period, method)
    ts = ts.apply(lambda x: x.fillna(x.iloc[0]))

    return ts


def interpolate_pd_series(series: pd.Series, period: int, method: str = 'nearest') -> pd.Series:

    idx = pd.date_range(start=series.index[0], end=series.index[-1], freq='{}H'.format(period))

    if method == 'last':
        result_series = series.reindex(idx, method='ffill', fill_value=np.nan)
    else:
        series2 = series.reindex(idx, fill_value=np.nan)
        result_series = series.combine_first(series2)
        result_series = result_series.interpolate(method)
        result_series = result_series.reindex(idx)

    result_series = result_series.round(2)
    return result_series
