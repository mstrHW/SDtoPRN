from module.fd_model.fd_rnn_converter import FDRNNConverter
from module.fd_model.vensim_fd_converter import get_fd, KNOWN_MODEL, UNKNOWN_MODEL
from module.data_loader import np_preproc_for_rnn2d
import numpy as np
import pysd


def case1():
    # the model is known, the coefficients are unknown
    general_params =\
        {
            'mode': KNOWN_MODEL,
            'seq_size': 240,
            'phi_h': lambda x: x,
            'phi_o': lambda x: x,
        }

    train_params =\
        {
            'learning_rate': 1e-1,
            'epochs_before_decay': 0.1,
            'epochs_count': 2e4,
            'learning_rate_decay': 1/3
        }

    return general_params, train_params


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


def generate_train_data(FD, data):
    fields = [level for level in FD.names_units_map.keys()]
    required_columns_data = data[fields].as_matrix()
    # my_data = my_data/abs(my_data).max()
    # my_data = preprocessing.normalize(my_data)
    # my_data = preprocessing.scale(my_data)
    return required_columns_data, np_preproc_for_rnn2d(required_columns_data)


def get_weights(rnn_model, rnn_model_file, FDRNN_converter, demo):
    weights = rnn_model.get_weights(model_file=rnn_model_file)
    w = weights['W']
    FDRNN_converter.print_w(w)

    if demo:
        print('###=== W ===###')
        print(w)


def run_simulation(rnn_model, rnn_model_file, initial_value, iterations_count, demo):
    output = rnn_model.get_simulation(initial_value=initial_value, iterations_count=iterations_count, model_file=rnn_model_file)

    if demo:
        print('###=== Simulation output ===###')
        print(output)
        print(initial_value)

    return output


def calculate_error(required_columns_data, output, demo):
    output = np.array(output)
    cost = sum(abs((output-required_columns_data)/required_columns_data))/required_columns_data.shape[0]

    if demo:
        print('###=== Simulation table ===###')
        print(output)
        print('###=== Cost ===###')
        print(cost)


def main(params, model, TRAIN_RNN=True, DEMO=False):
    model_name = model
    general_params, train_params = params
    mode = general_params['mode']

    models_directory = '../vensim_models/'
    vensim_model_file = models_directory + '{}.mdl'.format(model_name)
    rnn_model_file = '../models/demo/{}_case{}.ckpt'.format(model_name, mode)

    ### === Generate data === ###
    data = generate_sd_output(vensim_model_file)
    fields = get_sd_components(data)

    ### === Vensim model to FD === ###
    if mode == UNKNOWN_MODEL:
        FD = get_fd(fields, mode=mode)
        FD.dT = general_params['dt']
    else:
        FD = get_fd(vensim_model_file, mode=mode)

    required_columns_data, (train_X, train_Y) = generate_train_data(FD, data)

    if DEMO:
        print('###=== Fields ===###')
        print(fields)
        print('###=== train_X ===###')
        # print(train_X)

    ### === FD model to RNN === ###
    FDRNN_converter = FDRNNConverter(general_params['phi_h'], general_params['phi_o'])
    rnn_model = FDRNN_converter.fd_to_rnn(FD)

    if DEMO:
        print('###=== Levels ===###')
        print(FD.levels)

    if TRAIN_RNN:
        rnn_model.train(train_X, train_Y, train_params=train_params, model_file=rnn_model_file)

    if DEMO:
        get_weights(rnn_model, rnn_model_file, FDRNN_converter, DEMO)

    initial_value = np.reshape(train_X[0], [1, train_X.shape[1]])
    iterations_count = required_columns_data.shape[0]-1
    output = run_simulation(rnn_model, rnn_model_file, initial_value, iterations_count, DEMO)

    calculate_error(required_columns_data, output, DEMO)


if __name__ == '__main__':
    # case1()
    # case2()
    # teacup_case1
    # pp_model
    model = 'teacup'
    # model = 'teacup2.2'
    # model = 'pp_model'
    params = case1()
    main(params, model, TRAIN_RNN=True, DEMO=True)
