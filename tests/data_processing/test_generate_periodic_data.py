from unittest import TestCase
import pandas as pd
from tqdm import tqdm

from module.almazov_dataset_processing.convert_to_periodic_data import prepare_dates


tqdm.pandas()


class TestGeneratePeriodicData(TestCase):
    def test_prepare_dates(self):
        patient_id = ['1', '1', '2', '2', '3']
        event_date = ['201201010930', '201201022315', '201201010930', '201201011000', '201201010930']
        timedelta = ['0', '37.75', '0', '0.5', '0']
        continuous = ['1.0', '35.5', '0', '15', '35']
        categorical = ['1', '1', '0', '0', '1']

        data = {'patient_id': patient_id, 'event_date': event_date, 'timedelta': timedelta, 'continuous': continuous, 'categorical': categorical}
        input_data = pd.DataFrame(data=data)
        input_data[['timedelta', 'continuous', 'categorical']] = input_data[['timedelta', 'continuous', 'categorical']].astype(float)

        actual = input_data.groupby(['patient_id']).progress_apply(
            lambda x: prepare_dates(x, ['timedelta', 'continuous', 'categorical'], 6, 'linear'))

        print(actual)
