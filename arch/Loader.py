import pandas as pd
import csv
import numpy as np
import re
import sys
import matplotlib.pyplot as plt

class Loader(object):

    data_path = 'D:/Datasets/RW_2/'
    holesterin_file = 'holesterin.csv'
    labresult_file = 'labresult.csv'
    predst_file ='predst-o-bolnom-stp7.txt'
    nan_value = 'NA'




    def prepare_file(self, in_file, out_file):
        data = self.read_data(in_file)
        self.write_data(data, out_file)


    def read_data_predst(self, file_name, encoding='CP1251', delimiter='\t'):
        data = []
        in_file = open(file_name, 'r', encoding=encoding)
        columns = in_file.readline()
        columns = columns.replace('\n', '').split(delimiter)
        # columns = columns.split(delimiter)
        print(columns)
        for row in in_file:
            csv_row = row.split(delimiter)
            len_row = len(csv_row)
            len_columns = len(columns)
            if len_row > len_columns:
                # print(csv_row)
                pass
            data.append(csv_row[:len(columns)])
        in_file.close()
        df = pd.DataFrame(data=data, columns=columns)
        df = df.replace(r'', self.nan_value, regex=False)
        df = df.replace(self.nan_value, np.nan)

        return df

    def read_data(self, file_name, encoding='CP1251', delimiter='\t'):
        data = []
        with open(file_name, newline='', encoding=encoding) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
            columns = next(csv_reader)
            print(columns)
            for row in csv_reader:
                len_row = len(row)
                len_columns = len(columns)
                if len_row > len_columns:
                    # print(row)
                    pass
                data.append(row[:len(columns)])

        df = pd.DataFrame(data=data, columns=columns)
        df = df.replace(r'', self.nan_value, regex=False)
        df = df.replace(self.nan_value, np.nan)

        return df


    def write_data(self, data, file_name, encoding='CP1251'):
        # , float_format='%g'
        data.to_csv(file_name, sep='\t',  encoding=encoding, index=False, na_rep=self.nan_value)

    def read_df(self, file_name, encoding='CP1251'):
        if '.txt' in file_name:
            df = pd.read_csv(self, file_name, encoding=encoding, low_memory=False, index_col=False, delimiter='\t', na_values=[self.nan_value])
        else:
            df = pd.read_csv(self, file_name, encoding=encoding, low_memory=False, index_col=False, na_values=[self.nan_value])
        return df

    def txt_to_csv(self, file_name):
        data = self.read_data_predst(file_name)
        self.write_data(data, file_name[:-3]+'csv')
        return file_name[:-3]+'csv'

    def nan_prepare(self, data_frame, column_name=None):
        print('method nan_prepare()')
        if column_name:
            result = data_frame[column_name].fillna(data_frame[column_name])
        else:
            result = data_frame.fillna(0)
        return result

    def df_preproc_for_rnn(self, data_frame):

        columns = data_frame.columns
        columns = columns.drop(['patient_id', 'epizod_id', 'result', 'Department'])

        grouped_by_id = data_frame.groupby('patient_id')[columns]

        np_grouped = []

        groups_count = len(grouped_by_id)
        print('Groups count : '.format(groups_count))

        for patient_id, i in zip(grouped_by_id, range(groups_count)):
            new_frame = np.array(patient_id[1][columns].as_matrix())
            np_grouped.append(new_frame)
            if i % int(groups_count * 0.01) == 0:
                print('{}% of {} groups was prepared'.format(int(i / groups_count * 100), groups_count))

        np_grouped = np.array(np_grouped)

        X, Y = self.np_preproc_for_rnn3d(np_grouped)

        return [columns, X, Y]

    def __np_preproc_for_rnn3d(self, numpy_array):
        X = []
        Y = []

        for i in range(numpy_array.shape[0]):
            batch = numpy_array[i]
            batch_X = batch[:-1]
            batch_Y = batch[1:]
            if len(batch_X) > 0 and len(batch_Y) > 0:
                X.append(batch_X)
                Y.append(batch_Y)

        X = np.array(X)
        Y = np.array(Y)

        return [X, Y]



loader = Loader()

def split_on_seq_size(data, seq_size):
    new_shape = data.shape[0]-seq_size
    if new_shape > 0:
        data_for_rnn = [data[i:i + seq_size] for i in range(new_shape+1)]
    else:
        data_for_rnn = data
    return np.array(data_for_rnn)



def normalize_data(X):
    from sklearn import preprocessing
    new_X = np.array(X[0])
    for x in X[1:]:
        new_X = np.append(new_X, x, axis=0)

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(new_X)
    return x_scaled


if __name__ == '__main__':
    loader = Loader()