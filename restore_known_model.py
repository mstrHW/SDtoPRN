from module.fd_model.fd_rnn_converter import FDRNNConverter
from module.fd_model.vensim_fd_converter import get_fd, KNOWN_MODEL, UNKNOWN_MODEL
from module.data_loader import np_preproc_for_rnn2d
from module.print_results.stats import biplot, plot_two_graphs

import numpy as np
import pysd
import logging
import argparse
from sklearn import preprocessing

from definitions import path_join, make_directory, VENSIM_MODELS_DIR, EXPERIMENTS_DIR


def case2():
    # the model and coefficients are unknown

    general_params =\
        {
            'mode': UNKNOWN_MODEL,
            'seq_size': 200,
            'phi_h': lambda x: x,
            'phi_o': lambda x: x,
            'dt': 0.125
            # 'dt': 0.03125
        }

    train_params =\
        {
            'learning_rate': 1e-2,
            'epochs_before_decay': 0.1,
            'epochs_count': 2e4,
            'learning_rate_decay': 1/3
        }

    return general_params, train_params


def generate_sd_output(vensim_model_file):
    model = pysd.read_vensim(vensim_model_file)
    data = model.run()
    return data


def get_sd_components(data):
    fields = data.columns
    general_stopwords = ['INITIAL TIME', 'FINAL TIME', 'SAVEPER', 'TIME STEP', 'Time', 'TIME']
    stopwords = ['predator births', 'predator deaths', 'prey births', 'prey deaths', 'Heat Loss to Room']
    fields = [key for key in fields if key not in general_stopwords]
    fields = [key for key in fields if key not in stopwords]
    return fields


def generate_train_data(fields, data):
    dataset = data[fields].as_matrix()
    dataset = dataset/abs(dataset).max()
    # dataset = preprocessing.normalize(dataset)
    # dataset = preprocessing.scale(dataset)
    return dataset, np_preproc_for_rnn2d(dataset)


def get_weights(rnn_model, rnn_model_file, FDRNN_converter):
    weights = rnn_model.get_weights(model_file=rnn_model_file)
    w = weights['W']
    FDRNN_converter.print_w(w)

    return w


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
    mode = KNOWN_MODEL

    experiment_name = args.experiment_name
    experiment_dir = path_join(EXPERIMENTS_DIR, experiment_name)
    make_directory(experiment_dir)

    tf_model_dir = path_join(experiment_dir, 'tf_model')
    make_directory(tf_model_dir)

    images_dir = path_join(experiment_dir, 'images')
    make_directory(images_dir)

    log_path = path_join(experiment_dir, 'log.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)

    vensim_model_file = path_join(VENSIM_MODELS_DIR, '{}.mdl'.format(model_name))
    rnn_model_file = path_join(tf_model_dir, '{}_case{}.ckpt'.format(model_name, mode))

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

    ### === Vensim model to FD === ###
    # if mode == UNKNOWN_MODEL:
    #     FD = get_fd(fields, mode=mode)
    #     FD.dT = general_params['dt']
    # else:
    FD = get_fd(vensim_model_file, mode=mode)
    data = generate_sd_output(vensim_model_file)

    for column in data.columns:
        _slice = data[column].values
        pass

    print('need_train: {}'.format(need_train))
    print('data.shape: {}'.format(data.shape))
    fields = [level for level in FD.names_units_map.keys()]

    simulation_data, (train_X, train_Y) = generate_train_data(fields, data)

    logging.info('###=== Fields ===###')
    logging.info(fields)
    logging.info('###=== train_X ===###')

    ### === FD model to RNN === ###
    FDRNN_converter = FDRNNConverter(general_params['phi_h'], general_params['phi_o'])
    rnn_model = FDRNN_converter.fd_to_rnn(FD)
    levels = FD.levels

    logging.info('###=== Levels ===###')
    logging.info(levels)

    if need_train:
        rnn_model.train(train_X, train_Y, train_params=train_params, model_file=rnn_model_file)

    weight = get_weights(rnn_model, rnn_model_file, FDRNN_converter)
    logging.info('###=== Weights ===###')
    logging.info(weight)

    initial_value = np.reshape(train_X[0], [1, train_X.shape[1]])

    iterations_count = args.iterations_count
    if iterations_count == 0:
        iterations_count = simulation_data.shape[0]-1

    simulation_data = simulation_data[:iterations_count + 1]
    output = run_simulation(rnn_model, rnn_model_file, initial_value, iterations_count)

    logging.info('###=== Simulation output ===###')
    logging.info(output)
    logging.info(initial_value)

    error = calculate_error(simulation_data, output)

    logging.info('###=== Simulation table ===###')
    logging.info(output)
    logging.info('###=== Error ===###')
    logging.info(error)

    for level in levels:
        i = fields.index(level)
        level_output = output[:, i]
        level_y = simulation_data[:, i]

        biplot_name = 'biplot {} sd and prn simulation'.format(level)
        graph_name = 'graph {} sd and prn graphs'.format(level)

        biplot(level_output, level_y, biplot_name, images_dir)
        plot_two_graphs(level_output, level_y, graph_name, images_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
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
        default=2e4,
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
