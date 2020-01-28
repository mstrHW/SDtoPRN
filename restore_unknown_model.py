from module.fd_model.fd_rnn_converter import FDRNNConverter
from module.fd_model.vensim_fd_converter import get_fd, KNOWN_MODEL, UNKNOWN_MODEL
from module.data_loader import np_preproc_for_rnn2d
from module.print_results.stats import biplot, plot_graphs

import numpy as np
import pandas as pd
import pysd
import logging
import argparse
from sklearn import preprocessing
import pickle

from definitions import path_join, make_directory, VENSIM_MODELS_DIR, EXPERIMENTS_DIR, DATA_DIR
from arch.base_nn import BaseNN


def generate_train_data(fields, data):
    dataset = data[fields].as_matrix()
    # dataset = dataset/abs(dataset).max()
    # dataset = preprocessing.normalize(dataset)
    # dataset = preprocessing.scale(dataset)
    return dataset, np_preproc_for_rnn2d(dataset)


def get_sd_components(data):
    fields = data.columns
    general_stopwords = ['INITIAL TIME', 'FINAL TIME', 'SAVEPER', 'TIME STEP', 'Time', 'TIME']
    stopwords = ['predator births', 'predator deaths', 'prey births', 'prey deaths', 'Heat Loss to Room']
    fields = [key for key in fields if key not in general_stopwords]
    fields = [key for key in fields if key not in stopwords]
    return fields


def get_weights(rnn_model, rnn_model_file, FDRNN_converter):
    weights = rnn_model.get_weights(model_file=rnn_model_file)
    w = weights['W']
    edges_list = FDRNN_converter.print_w(w)

    return w, edges_list


def run_simulation(rnn_model, rnn_model_file, initial_value, iterations_count):
    output = rnn_model.get_simulation(initial_value=initial_value, iterations_count=iterations_count, model_file=rnn_model_file)

    return output


def calculate_error(required_columns_data, output):
    output = np.array(output)
    error = sum(abs((output-required_columns_data)/required_columns_data))/required_columns_data.shape[0]

    return error


def main(args):
    model_name = args.model_name
    need_train = bool(args.need_retrain)
    mode = UNKNOWN_MODEL

    dataset_file_name = args.dataset_file_name
    dataset_dir = path_join(DATA_DIR, model_name)
    dataset_path = path_join(dataset_dir, dataset_file_name)

    experiment_name = args.experiment_name
    experiment_dir = path_join(EXPERIMENTS_DIR, experiment_name)
    make_directory(experiment_dir)

    tf_model_dir = path_join(experiment_dir, 'tf_model')
    make_directory(tf_model_dir)

    images_dir = path_join(experiment_dir, 'images')
    make_directory(images_dir)

    log_path = path_join(experiment_dir, 'log.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)

    prn_model_file = path_join(tf_model_dir, '{}_case{}.ckpt'.format(model_name, mode))
    nn_model_dir = path_join(tf_model_dir, 'nn_model')
    make_directory(nn_model_dir)
    nn_model_file = path_join(nn_model_dir, 'my_checkpoint')

    general_params = \
        {
            'phi_h': lambda x: x,
            'phi_o': lambda x: x,
        }

    train_params = \
        {
            'learning_rate': args.learning_rate,
            'epochs_before_decay': args.epochs_before_decay,
            'epochs_count': args.epochs_count,
            'learning_rate_decay': args.learning_rate_decay,
        }

    ### === Generate data === ###
    # data = generate_sd_output(vensim_model_file)
    # fields = get_sd_components(data)

    data = pd.read_csv(dataset_path)
    dt = data['TIME'].values[0] - data['TIME'].values[1]
    fields = [column for column in data.columns if column != 'TIME']

    FD = get_fd(fields, mode=mode)
    FD.dT = dt

    print('dt: {}'.format(dt))
    fields = [level for level in FD.names_units_map.keys()]

    simulation_data, (X, Y) = generate_train_data(fields, data)

    delimiter = int(0.5 * X.shape[0])
    train_X = X[:delimiter]
    train_Y = Y[:delimiter]
    test_X = X[delimiter:]
    test_Y = Y[delimiter:]

    logging.info('###=== Fields ===###')
    logging.info(fields)
    logging.info('###=== train_X ===###')

    ### === FD model to RNN === ###
    FDRNN_converter = FDRNNConverter(general_params['phi_h'], general_params['phi_o'])
    rnn_model = FDRNN_converter.fd_to_rnn(FD)
    levels = FD.levels

    levels_file = path_join(experiment_dir, 'levels')
    with open(levels_file, 'wb') as f:
        pickle.dump(levels, f)

    logging.info('###=== Levels ===###')
    logging.info(levels)

    rnn_model.calculate_trainable_parameters()
    if need_train:
        rnn_model.train(train_X, train_Y, train_params=train_params, model_file=prn_model_file)

    weight, edges_list = get_weights(rnn_model, prn_model_file, FDRNN_converter)
    edges_file = path_join(experiment_dir, 'edges')

    with open(edges_file, 'wb') as f:
        pickle.dump(edges_list, f)

    logging.info('###=== Weights ===###')
    logging.info(weight)

    initial_value = np.reshape(test_X[0], [1, test_X.shape[1]])

    iterations_count = args.iterations_count
    if iterations_count == 0:
        iterations_count = test_X.shape[0]-1

    # test_Y = test_Y[:iterations_count + 1]
    simulation_data = test_Y.copy()
    # simulation_data = np.concatenate((initial_value, simulation_data), axis=0)

    prn_output = run_simulation(rnn_model, prn_model_file, initial_value, iterations_count)

    simulation_data_file = path_join(experiment_dir, 'simulation_data')
    with open(simulation_data_file, 'wb') as f:
        pickle.dump(simulation_data, f)

    prn_output_file = path_join(experiment_dir, 'prn_output')
    with open(prn_output_file, 'wb') as f:
        pickle.dump(prn_output, f)

    logging.info('###=== Stimulation output ===###')
    logging.info(prn_output)
    logging.info(initial_value)

    error = calculate_error(test_Y, prn_output)

    predictor = BaseNN(train_X.shape[1], train_X.shape[1])
    predictor.calculate_trainable_parameters()

    if need_train:
        predictor.train(train_X, train_Y, train_params, nn_model_file)
    nn_output = predictor.test(test_X, nn_model_file)

    nn_output_file = path_join(experiment_dir, 'nn_output')
    with open(nn_output_file, 'wb') as f:
        pickle.dump(nn_output, f)

    logging.info('###=== Simulation table ===###')
    logging.info(prn_output)
    logging.info('###=== Error ===###')
    logging.info(error)
    error2 = calculate_error(test_Y, nn_output)
    logging.info(error2)

    for level in levels:
        i = fields.index(level)
        level_output = prn_output[:, i]
        level_nn_output = nn_output[:, i]
        level_y = simulation_data[:, i]

        graphs = (level_output, level_nn_output, level_y)
        labels = ('prn_y', 'nn_y', 'true_y')

        biplot_name1 = 'biplot {} nn and sd simulation'.format(level)
        biplot_name2 = 'biplot {} prn and sd simulation'.format(level)
        graph_name = 'graph {} sd and prn graphs'.format(level)

        biplot(level_nn_output, level_y, biplot_name1, images_dir)
        biplot(level_output, level_y, biplot_name2, images_dir)
        plot_graphs(graphs, labels, '{} ({})'.format(model_name, level), images_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        help="",
    )

    parser.add_argument(
        "--dataset_file_name",
        type=str,
        help="",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="",
    )

    parser.add_argument(
        "--need_retrain",
        type=bool,
        help="",
    )

    parser.add_argument(
        "--iterations_count",
        type=int,
        default=0,
        help=""
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-1,
        help="",
    )

    parser.add_argument(
        "--epochs_before_decay",
        type=float,
        default=0.1,
        help="",
    )

    parser.add_argument(
        "--epochs_count",
        type=float,
        default=5e3,
        help="",
    )

    parser.add_argument(
        "--learning_rate_decay",
        type=float,
        default=1/3,
        help="",
    )

    args = parser.parse_args()
    main(args)
