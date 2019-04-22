from unittest import TestCase
from module.almazov_dataset_processing.data_analysis import *


class TestAnalysis(TestCase):

    def test_get_unique_symbols(self):
        data = {'col1': [1, 2], 'col2': [3, 4]}
        data_frame = pd.DataFrame(data=data)
        actual = get_unique_symbols(data_frame, 'col2')
        expected = {3: 0, 4: 1}
        self.assertEqual(actual, expected)

    def test_get_nans_count(self):
        data = {'col1': ['Na', 'Nan', 'N/A'], 'col2': ['', 'NaN', np.nan]}
        data_frame = pd.DataFrame(data=data, )
        actual = get_nans_count(data_frame, 'col1')
        self.assertEqual(actual, 0)
        actual = get_nans_count(data_frame, 'col2')
        self.assertEqual(actual, 1)
        actual = get_nans_count(data_frame)
        self.assertEqual(actual, 1)

    def test_get_nans_percent(self):
        data = {'col1': ['Na', 'Nan', 'N/A'], 'col2': ['', 'NaN', np.nan]}
        data_frame = pd.DataFrame(data=data)

        actual = get_nans_percent(data_frame, 'col1')
        self.assertEqual(actual, 0)

        actual = get_nans_percent(data_frame, 'col2')
        self.assertAlmostEqual(actual, 0.333, places=3)

        actual = get_nans_percent(data_frame)
        self.assertAlmostEqual(actual, 0.166, places=2)

        actual = get_nans_percent(data_frame, ['col1', 'col2'])
        self.assertAlmostEqual(actual, 0.166, places=2)
