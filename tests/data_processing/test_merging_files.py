from unittest import TestCase

from module.almazov_dataset_processing.merge_files import *


class TestMergingFiles(TestCase):

    def test_add_columns_same_ids(self):
        data1 = {'col1': ['1', '2'], 'col2': ['3', '4'], 'col3': [1, 2]}
        data1 = pd.DataFrame(data=data1)
        data2 = {'col1': ['1', '2'], 'col2': ['3', '4'], 'col4': [3, 4]}
        data2 = pd.DataFrame(data=data2)

        dfs = [data1, data2]
        ids = [['col1', 'col2'], ['col1', 'col2']]
        actual = add_columns(dfs, ids, ['col4'])

        expected = {'col1': ['1', '2'], 'col2': ['3', '4'],
                    'col3': [1, 2], 'col4': [3, 4]}
        expected = pd.DataFrame(data=expected)
        self.assertTrue(expected.equals(actual))

    def test_add_columns_different_ids(self):
        data1 = {'col1': ['1', '2'], 'col2': ['3', '4'], 'col3': [1, 2]}
        data1 = pd.DataFrame(data=data1)
        data2 = {'_col1': ['1', '2'], '_col2': ['3', '4'], 'col4': [3, 4]}
        data2 = pd.DataFrame(data=data2)

        dfs = [data1, data2]
        ids = [['col1', 'col2'], ['_col1', '_col2']]
        actual = add_columns(dfs, ids, ['col4'])

        expected = {'col1': ['1', '2'], 'col2': ['3', '4'],
                    'col3': [1, 2], 'col4': [3, 4]}
        expected = pd.DataFrame(data=expected)
        self.assertTrue(expected.equals(actual))

    def test_add_columns_same_column(self):
        data1 = {'col1': ['1', '2'], 'col2': ['3', '4'], 'col3': [1, 2]}
        data1 = pd.DataFrame(data=data1)
        data2 = {'col1': ['1', '2'], 'col2': ['3', '4'], 'col3': [3, 4]}
        data2 = pd.DataFrame(data=data2)

        dfs = [data1, data2]
        ids = [['col1', 'col2'], ['col1', 'col2']]
        actual = add_columns(dfs, ids, ['col3'])

        expected = {'col1': ['1', '2'], 'col2': ['3', '4'],
                    'col3': [3, 4]}
        expected = pd.DataFrame(data=expected)
        self.assertTrue(expected.equals(actual))
