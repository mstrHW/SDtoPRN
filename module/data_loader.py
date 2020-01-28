import pandas as pd
import csv
import numpy as np
import re


def read_data(file_name, encoding='CP1251', delimiter='\t'):
    data = []
    with open(file_name, newline='', encoding=encoding) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        columns = next(csv_reader)
        for row in csv_reader:
            data.append(row[:len(columns)])
    df = pd.DataFrame(data=data, columns=columns)
    df[df == 'NaN'] = np.nan
    return df


def write_to_csv(data, file_name, encoding='CP1251'):
    data.to_csv(file_name, sep='\t',  encoding=encoding, index=False, na_rep='NaN')


def np_preproc_for_rnn2d(numpy_array):
    X = numpy_array[:-1]
    Y = numpy_array[1:]
    return [X, Y]


def np_preproc_for_rnn3d(numpy_array):
    Xs = np.concatenate([group[1].values[:-1] for group in numpy_array], axis=0)
    ys = np.concatenate([group[1].values[1:] for group in numpy_array], axis=0)

    return Xs, ys
