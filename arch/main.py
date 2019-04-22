import numpy as np

from module.data_loader import split_on_seq_size
from arch.SeriosPredictor import SeriesPredictor


class dataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

def read_file(file_name):
    import csv

    data = []
    with open(file_name, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        csvfile.readline()
        for row in csv_reader:
            data.append(row)

    return data

def file_preproc(file_name, layers, WITH_DELIMITER=False):

    num_features = layers[0]
    num_targets = layers[-1]

    data = read_file(file_name)
    data_count = data.shape[0]
    # print(data_count)
    X = data[:, :num_features]
    Y = [int(y) for y in data[:, num_features]]
    tmpY = np.zeros((data_count, num_targets))
    tmpY[np.arange(data_count), Y] = 1
    Y = tmpY
    X = data_preprocessing(X)[0]

    if WITH_DELIMITER:
        delimiter = int(0.9 * data_count)

        trainX = X[:delimiter]
        trainY = Y[:delimiter]
        train = dataset(trainX, trainY)

        testX = X[delimiter:]
        testY = Y[delimiter:]
        test = dataset(testX, testY)
        return train, test
    else:
        return dataset(X, Y)

def data_preproc_for_rnn(data, seq_size):

    X = data[:, :data.shape[1]-1]
    Y = data[:, data.shape[1]-1]
    X_for_rnn = split_on_seq_size(X, seq_size)
    Y_for_rnn = split_on_seq_size(Y, seq_size)

    X_for_rnn = np.array(X_for_rnn)
    Y_for_rnn = np.array(Y_for_rnn)

    return [X_for_rnn, Y_for_rnn]

def demo():
    part_name = 'new_labresult'
    full_name = 'data/{}.csv'.format(part_name)
    seq_size = 4

    data = read_file(full_name)
    data = np.array(data)

    X, Y = data_preproc_for_rnn(data, seq_size)

    input_dim = X.shape[2]
    print(X.shape)

    predictor = SeriesPredictor(input_dim=input_dim, seq_size=seq_size, hidden_dim=10)
    predictor.train(X, Y)
    predictor.test(X)

if __name__ == '__main__':
    demo()