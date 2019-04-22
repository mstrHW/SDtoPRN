import os
import matplotlib.pyplot as plt

from module.pysd_simulation.pysd_simulation import *
from module.fd_model.fd_rnn_converter import FDRNNConverter
from module.fd_model.vensim_fd_converter import get_fd, UNKNOWN_MODEL
from module.print_results.stats import *


class Params(object):

    def __init__(self, vensim_file, mode, mask, simulation_type, general_params, train_params, period):

        self.mode = mode
        self.root_dir = 'results/'
        self.data_directory = '{}data_{}/'.format(self.root_dir, mask)
        self.models_dir = self.root_dir + 'models/'
        self.simulations_dir = self.root_dir + 'simulations/{}/{}/'.format(mask, simulation_type)
        self.images_dir = self.simulations_dir + 'images/'
        self.rnn_model_file = '{}{}'.format(self.models_dir, mask)
        self.mask = mask
        self.full_mask = '{}_sim{}'.format(mask, simulation_type)
        self.general_params = general_params
        self.train_params = train_params
        self.simulation_type = simulation_type
        self.period = period
        self.create_dirs()
        self.vensim_file = vensim_file

    def create_dirs(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        if not os.path.exists(self.simulations_dir):
            os.makedirs(self.simulations_dir)

        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def load_data(self):
        self.train_X = np.load('{}X.npy'.format(self.data_directory)).astype(float)
        self.train_Y = np.load('{}Y.npy'.format(self.data_directory)).astype(float)
        # self.test_X = np.load('{}test_X.npy'.format(self.data_directory)).astype(float)
        # self.test_Y = np.load('{}test_Y.npy'.format(self.data_directory)).astype(float)
        self.names = np.load('{}names.npy'.format(self.data_directory))

def case_params():

    general_params =\
        {
            'seq_size': 200,
            'phi_h': lambda x: x,
            'phi_o': lambda x: x,
        }

    train_params =\
        {
            'learning_rate': 1e-3,
            'epochs_before_decay': 0.1,
            'epochs_count': 10,
            'learning_rate_decay': 0.7
        }
    # train_params =\
    #     {
    #         'learning_rate': 1e-3,
    #         'epochs_before_decay': 0.001,
    #         'epochs_count': 1000,
    #         'learning_rate_decay': 0.9
    #     }
    return general_params, train_params


def __df_preproc_for_rnn2d(data_frame):
    X = data_frame.loc[:-1, :].values
    Y = data_frame.loc[1:, :].values
    return [X, Y]


def __np_preproc_for_rnn2d(numpy_array):
    X = numpy_array[:-1]
    Y = numpy_array[1:]
    return [X, Y]


def generate_pysd_data(vensim_file):
    data, fields = generate_data(vensim_file)
    fields = data.columns

    dt = data['TIME STEP'].values[0]

    general_stopwords = ['INITIAL TIME', 'FINAL TIME', 'SAVEPER', 'TIME STEP', 'Time', 'TIME']
    # stopwords = ['predator births', 'predator deaths', 'prey births', 'prey deaths', 'Heat Loss to Room']
    fields = [key for key in fields if key not in general_stopwords]
    # fields = [key for key in fields if key not in stopwords]

    data = data[fields]
    print(data.columns)
    my_data = data[fields].as_matrix()

    train_X, train_Y = __np_preproc_for_rnn2d(my_data)
    return train_X, train_Y, dt, data.columns


def get_FD(params: Params):
    FD = get_fd(params, mode=params.mode)
    FD.dT = params.period
    print('###=== Levels ===###')
    print(FD.levels)
    return FD


def get_FDRNNConverter(params: Params):
    general_params = params.general_params
    FDRNN_converter = FDRNNConverter(general_params['phi_h'], general_params['phi_o'])
    return FDRNN_converter


def create_rnn(params: Params):
    FD = get_fd(params)

    ### === FD model to RNN === ###
    FDRNN_converter = get_FDRNNConverter(params)
    rnn_model = FDRNN_converter.fd_to_rnn(FD, params.mode)

    return rnn_model, FD.levels+FD.constants, [name for name in FDRNN_converter.FD.names_hidden_map.keys()]


def train_phases_rnn(params, last_time=24*16):

    train_X = params.train_X
    train_Y = params.train_Y


    rnn_model, v1, v2 = create_rnn(params)

    for i in range(0, last_time, params.period):
        _train_X = train_X[train_X[:, -1].astype(int) == int(i)][:, 1:-1]
        _train_Y = train_Y[train_Y[:, -1].astype(int) == int(i+params.period)][:, 1:-1]
        # print(len(_train_X))
        # if len(_train_Y) > 0:
        rnn_model.train_batches_v2(_train_X, _train_Y, train_params=params.train_params, model_file=params.rnn_model_file+'_'+str(i))

        weights = rnn_model.get_weights(model_file=params.rnn_model_file+'_'+str(i))
        W = weights['W_ah']

        np.save('{}W_{}_{}.npy'.format(params.simulations_dir, params.mask, i), W)


def get_weights(W, column_index):
    W = W.T
    # result = sorted(abs(W[column_index]), reverse=True)
    result = abs(W[column_index])
    # result = W[column_index]
    return result


def test_rnn(params, last_time=24*16):
    # FD = get_FD(params)
    rnn_model, input_names, hidden = create_rnn(params)
    # _mask = [True if name in input_names else False for name in params.names]

    test_X = params.train_X
    # test_X = test_X[:, _mask]

    outputs = []
    # if params.simulation_type == 1:
    #     output = rnn_model.get_batches_simulation(test_x=test_X, model_file=params.rnn_model_file)
    # if params.simulation_type == 2:
    patient_ids = test_X[:, 0]
    for patient in patient_ids:
        _mask = test_X[:, 0] == patient
        patient_data = test_X[_mask, :]
        for row in patient_data:
            output = rnn_model.get_batches_simulation_v2(test_x=row[:-1], model_file=params.rnn_model_file+'_'+str(row[-1]))
            outputs.append(output)
    outputs = np.array(outputs)
    np.save('{}rnn_output_{}.npy'.format(params.simulations_dir, params.full_mask), outputs)


def test_sd(params):
    rnn_model, input_names, hidden = create_rnn(params)
    _mask = [True if name in input_names else False for name in params.names]

    test_X = params.test_X
    test_X = test_X[:, _mask]

    # if params.simulation_type == 1:
    #     output = get_pysd_simulation(test_X, names)
    # if params.simulation_type == 2:
    #     output = get_pysd_simulation_v2(test_X, names)

    if params.simulation_type == 1:
        output = rnn_model.new_sim_func(test_X, model_file=params.rnn_model_file)
    else:
        output = rnn_model.new_sim_func2(test_X, model_file=params.rnn_model_file)

    np.save('{}pysd_output_{}.npy'.format(params.simulations_dir, params.full_mask), output)


def get_results(params):
    test_Y = params.test_Y
    _Xs = []
    for i in range(test_Y.shape[0]):
        _X = []
        _test_i = test_Y[i].astype(float)
        for j in range(_test_i.shape[0]):
            if np.isnan(_test_i[j]).sum() < 1:
                _X.append(_test_i[j])
        _test_X = np.array(_X)
        _Xs.append(_test_X)
    Y = np.array(_Xs)

    names = params.names
    names = names[1:]

    steps_count = 8
    steps = np.arange(0, steps_count, 1)
    xs = np.arange(12, (steps_count+1)*12, 12)
    min_steps = steps_count
    max_steps = steps_count


    rnn_output = np.load('{}rnn_output_{}.npy'.format(params.simulations_dir, params.full_mask))

    column = 1
    name = names[column-1]
    print(name)
    # assert (rnn_output.shape == Y.shape)
    ys, es, ms, mes = get_statictics(rnn_output, Y, min_steps, max_steps, steps, column)
    # pys, pes, pms, pmes = get_statictics(pysd_output, Y, min_steps, max_steps, steps, column)
    # plot_std_error_v4(xs, [ys, pys], [es, pes], 'StDev ({})'.format(name), 'StDev ({}) for ANN and SD Simulation 1'.format(name))
    # plot_std_error_v2(xs, [ms, pms], [mes, pmes], 'MAE ({})'.format(name), 'MAE ({}) for ANN and SD  Simulation 1'.format(name))

    _Y = get_with_steps_size(Y, min_steps, max_steps)
    # _pysd_output = get_with_steps_size(pysd_output, min_steps, max_steps)
    _output = get_with_steps_size(rnn_output, min_steps, max_steps)

    # y = [_output[patient], _pysd_output[patient], _Y[patient]]



    # plot_distribution(ms, 'MAE (HGB) distribution')
    for column in range(1, len(names)+1):
        name = names[column - 1]
        print(name)
        steps_count = 10
        min_steps = 9
        max_steps = 11
        # _output = get_by_step(rnn_output, steps_count, column=column)
        # _Y = get_by_step(Y, steps_count, column=column)
        _Y = get_with_steps_size(Y, min_steps, max_steps, column=column)
        _output = get_with_steps_size(rnn_output, min_steps, max_steps, column=column)

        _y = []
        _o = []
        for i in range(_Y.shape[0]):
            for j in range(_Y[i].shape[0]):
                _y.append(_Y[i][j])
                _o.append(_output[i][j])

        _Y = np.array(_y)
        _output =  np.array(_o)

        mae = np.absolute(_output - _Y)

        plot_distribution(mae, 'AE ({}) distribution'.format(name), params.images_dir)

        # _Y = get_with_steps_size(Y, min_steps, max_steps)
        # _pysd_output = get_with_steps_size(pysd_output, min_steps, max_steps)
        # _output = get_with_steps_size(rnn_output, min_steps, max_steps)


        biplot(_Y, _output, 'Biplot ({})'.format(name), params.images_dir)
        # again(_Y, 'Correct value ({})'.format(name), name, params.images_dir)
        # again(_Y, 'SD model ({})'.format(name), name, params.images_dir)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def moving_average2(a, n=3):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(a, 1):
        cumsum.append(cumsum[i-1]+x)
        if i >= n:
            moving_ave = (cumsum[i]-cumsum[i-n])/n
            moving_aves.append(moving_ave)
        if i < n:
            moving_ave = (cumsum[i])/i
            # moving_ave = i
            moving_aves.append(moving_ave)

    return moving_aves


def my_alg(a):
    result = [a[0]]
    for i in range(1, len(a)-1):
        result.append((a[i-1] + a[i] + a[i+1])/3)
    result.append(a[-1])
    return result


def print_phases(xs, matrix, images_dir, title, column_names, input_threshold=None):
    _matrix = matrix.T

    plt.figure(figsize=(24, 16))
    plt.grid(True)

    plt.xlabel('Time (in hours)')
    xticks = [xs[i] for i in range(0, len(xs), 2)]
    plt.xticks(xticks)

    # if input_threshold:
    #     threshold = input_threshold
    # else:
    #     threshold = 0.005
    #     _threshold = _matrix.max()/4
    #     if _threshold > threshold:
    #         threshold = _threshold
    #
    #     threshold = int(threshold / 0.005) * 0.005
    #     plt.yticks([threshold*i for i in range(0, int(0.5/threshold))])
    threshold = input_threshold


    xs = [0, 24, 72, 168]
    ys = [int(x/6) for x in xs]
    plt.xlim(xmin=0, xmax=xs[-1]+1)
    # plt.ylim(ymin=0, ymax=_matrix.max()+0.05)
    # plt.title(title)
    # ax.ylabel(y_label)
    # x_smooth = np.linspace(xs.min(), xs.max(), int(len(xs)*1.1))


    handles = []
    names = []

    for i, coeff in enumerate(_matrix):
        if max(coeff > threshold):
            # ys = moving_average2(coeff, 5)
            # ys = coeff[:len(xs)]
            y_smooth = coeff[ys]
            x_smooth = xs
            # if len(xs) > 2:
            #     y_smooth = spline(xs, ys, x_smooth)

            if i % 2 == 0:
                handles.append(plt.plot(x_smooth, y_smooth, label=str(i), linestyle='--', linewidth=2)[0])
            else:
                handles.append(plt.plot(x_smooth, y_smooth, label=str(i), linewidth=4)[0])
            names.append(column_names[i])
        # _xs = xs
        # _ys = coeff
        # y_smooth = spline(_xs, _ys, x_smooth)
        #
        # y_smooth_bigger = y_smooth.copy()
        # y_smooth_lower = y_smooth.copy()
        # y_smooth_bigger[y_smooth < threshold] = np.nan
        # y_smooth_lower[y_smooth >= threshold] = np.nan
        #
        # if y_smooth_bigger[~pd.isna(y_smooth_bigger)].shape[0] > 5:
        #     handles.append(plt.plot(x_smooth, y_smooth_bigger, label=str(i))[0])
        # plt.plot(x_smooth, y_smooth_lower, alpha=0.0, color='g')

    plt.legend(handles, names, loc='upper right', prop={'size': '15'})

    # plt.show()
    if '/' in title:
        title = title[title!= '/']
    plt.savefig(images_dir + title + '.png')
    plt.gcf().clear()


def get_phases(params, last_time, images_dir, input_threshold=None):
    names = params.names
    # print(names)

    # target_names = np.array(['исход  -', 'duration', 'activ', 'Bleeding', 'Contrast-induced nephropathy',
    # 'Stress-induced hyperglycemia', 'Systemic inflammatory response'])

    target_names = ['исход  -']
    for column_name in target_names:
        Ws = []
        for i in range(0, last_time+1, params.period):
            W = np.load('{}W_{}_{}.npy'.format(params.simulations_dir, params.mask, i))
            _W = get_weights(W, names.tolist().index(column_name))
            Ws.append(_W)
        Ws = np.array(Ws)

        xs = np.arange(0, last_time+1, params.period)
        print_phases(xs, Ws, images_dir, column_name, names, input_threshold)


def train_rnn(params):
    rnn_model, input_names, hidden = create_rnn(params)

    _mask = [True if name in input_names else False for name in params.names]
    rnn_model.train_batches_v2(params.train_X[:, _mask], params.train_Y[:, _mask], train_params=params.train_params, model_file=params.rnn_model_file)

    weights = rnn_model.get_weights(model_file=params.rnn_model_file)
    W = weights['W_ah']

    np.save('{}W_{}.npy'.format(params.simulations_dir, params.mask), W)

    return input_names, _mask


def medical():
    # _SCALLED
    method = 'new_nearest'
    period = 6
    mask = method + str(period)
    simulation_type = 2
    case = 2
    _tmp_params = case_params()
    my_params = Params('', case, mask, simulation_type, _tmp_params[0], _tmp_params[1], period)
    my_params.load_data()
    print(my_params.names)
    my_params.names = my_params.names[1:-1]

    train_phases_rnn(my_params)
    # test_rnn(my_params)
    # get_results(my_params)
    # get_weights(my_params)
    #[24, 72, 120, 240, 360]

    for last_time in [7*24]:
        images_dir = my_params.images_dir + '' + str(last_time) + '/' + 'all/'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        get_phases(my_params, last_time, images_dir, 0.000)

        images_dir = my_params.images_dir + '' + str(last_time) + '/' + 'threshold/'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        get_phases(my_params, last_time, images_dir, 0.002)
        print('{} was plotted'.format(last_time))


def plot_track(xs, ys, title, images_dir):
    plt.figure(figsize=(8, 8))
    plt.grid(True)

    plt.xlabel('Time (in hours)')
    plt.xticks([i * 20 for i in range(6)])
    plt.plot(xs, ys)
    plt.title(title)

    # plt.show()
    plt.savefig(images_dir + title + '.png')
    plt.gcf().clear()


def prepator_prey():
    simulation_type = 1
    case = 1
    mask = 'predator_prey_{}'.format(case)

    vensim_name = 'Lotka-Volterra.mdl'
    vensim_file = '../vensim_models/{}'.format(vensim_name)
    train_X, train_Y, period, names = generate_pysd_data(vensim_file)
    _tmp_params = case_params()
    my_params = Params(vensim_file, case, mask, simulation_type, _tmp_params[0], _tmp_params[1], period)
    my_params.train_X = train_X
    my_params.train_Y = train_Y
    my_params.test_X = train_X
    my_params.test_Y = train_Y
    my_params.period = period
    my_params.names = names

    # print(names)
    # input_names = train_rnn(my_params)
    # hidden_names = test_rnn(my_params)
    # test_sd(my_params)
    rnn_model, input_names, hidden_names = create_rnn(my_params)
    # _mask = [True if name in input_names else False for name in my_params.names]
    # print('###=== Input names ===###')
    # print(input_names)
    # print('###=== Hidden names ===###')
    # print(hidden_names)
    #
    W = np.load('{}W_{}.npy'.format(my_params.simulations_dir, my_params.mask))
    # get_weights(my_params, W, 0)
    #
    print(input_names)
    for i, name in enumerate(hidden_names):
        print(name)
        print(get_weights(my_params, W, i))
    # test_rnn(my_params)
    # output = np.load('{}rnn_output_{}.npy'.format(my_params.simulations_dir, my_params.full_mask))
    # dt = 0.125
    # xs1 = [i * dt for i in range(output.shape[0])]
    # xs2 = [i * dt for i in range(my_params.test_X.shape[0])]
    # for name in hidden_names:
    #     column_index = hidden_names.index(name)
    #     column_track = output[:, column_index]
    #     plot_track(xs1, column_track, name + ' (ANN Model)', my_params.images_dir)
    #     column_track = my_params.test_X[:, column_index]
    #     plot_track(xs2, column_track, name + ' (SD model)', my_params.images_dir)
    # W1 = np.load('{}W_{}.npy'.format(my_params.simulations_dir, my_params.mask))

    # case = 2
    # my_params = Params(vensim_file, case, mask, simulation_type, _tmp_params[0], _tmp_params[1], period)
    # W2 = np.load('{}W_{}.npy'.format(my_params.simulations_dir, my_params.mask))

    # test_rnn(my_params)
    # test_sd(my_params)

    # print((W1 == W2).min())


def linear_regression():
    method = '3new_gradient'
    period = 6
    mask = method + str(period) + '_SCALLED'
    simulation_type = 2
    case = UNKNOWN_MODEL
    _tmp_params = case_params()
    my_params = Params('', case, mask, simulation_type, _tmp_params[0], _tmp_params[1], period)
    my_params.load_data()
    print(my_params.names)
    my_params.names = my_params.names[1:-1]

    train_phases_rnn(my_params)
    # test_rnn(my_params)
    # get_results(my_params)
    # get_weights(my_params)
    for last_time in [24, 72, 120, 240, 360]:
        images_dir = my_params.images_dir + str(last_time) + '/' + 'all/'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        get_phases(my_params, last_time, images_dir, 0.000)

        images_dir = my_params.images_dir + str(last_time) + '/' + 'threshold/'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        get_phases(my_params, last_time, images_dir, 0.025)
        print('{} was plotted'.format(last_time))


if __name__ == '__main__':
    medical()
    # prepator_prey()
    # linear_regression()
    pass