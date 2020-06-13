import numpy as np
import tensorflow as tf
# from module.nn_model import NNModel
from module.fd_model.fd_model import FD
import logging
# from arch.NewRNN import NewRNN

my_float = np.float32


class FDRNNConverter(object):

    def __init__(self, hidden_activation, output_activation):
        self.phi_h = hidden_activation
        self.phi_o = output_activation

    def fd_to_rnn(self, flow_diagram: FD, nn_model):

        self.flow_diagram = flow_diagram

        levels_count = len(flow_diagram.levels)
        constants_count = len(flow_diagram.constants)

        rates_count = len(flow_diagram.rates)
        units_count = levels_count + constants_count

        hidden_count = rates_count

        self.units_count = units_count
        self.hidden_count = hidden_count
        self.rates_count = rates_count

        W_xy_shape = [units_count, units_count]
        W_xy = np.zeros(shape=W_xy_shape, dtype=my_float)
        for i in range(W_xy_shape[0]):
            W_xy[i][i] = 1

        W_ah_shape = [units_count, rates_count]
        W_ah = np.zeros(shape=W_ah_shape, dtype=my_float)

        W_ry_shape = [rates_count, units_count]
        W_ry = np.zeros(shape=W_ry_shape, dtype=my_float)

        for rate in flow_diagram.rates:

            start_point = rate.flow.start_point
            end_point = rate.flow.end_point

            rate_index = flow_diagram.names_hidden_map[rate.name]

            if start_point != 'None':
                start_index = flow_diagram.names_units_map[start_point]
                W_ry[rate_index, start_index] = -flow_diagram.dT

            if end_point != 'None':
                end_index = flow_diagram.names_units_map[end_point]
                W_ry[rate_index, end_index] = flow_diagram.dT

            # for inf_sourse, coef in rate.expression.elements.items():
            #     W[FD.names_units_map[inf_sourse], rate_index] = coef

        tf_W_xy = self.create_constant(W_xy, W_xy_shape, 'W_xy')
        # tf_W_ah = tf.Variable(tf.random.truncated_normal((W_ah.shape[0], W_ah.shape[-1]), stddev=0.02), name='W_ah')
        tf_W_ah = self.create_variable(W_ah, W_ah_shape, 'W_ah')
        tf_W_ry = self.create_constant(W_ry, W_ry_shape, 'W_ry')

        parameters = {
            'hidden_count': hidden_count,
            'units_count': units_count,
            'phi_h': self.phi_h,
            'phi_o': self.phi_o,
            'W_xy': tf_W_xy,
            'W_ah': tf_W_ah,
            'W_ry': tf_W_ry,
        }

        model = nn_model(parameters)
        # model = NewRNN(parameters)

        return model

    def weights_to_tensorflow(self, weights, name):

        tf_weights_list = []
        for line_weights in weights:

            tf_line_list = []

            for weight in line_weights:
                if weight == 0:
                    tf_weight = self.create_constant(weight, [1], name)
                else:
                    tf_weight = self.create_variable(weight, [1], name)
                tf_line_list.append(tf_weight)

            tf_line = tf.stack(tf_line_list)
            tf_weights_list.append(tf_line)

        tf_weights = tf.stack(tf_weights_list)
        tf_weights = tf.reshape(tf_weights, [self.units_count, self.hidden_count])

        return tf_weights

    def create_constant(self, initial_value, shape, name):
        return tf.constant(initial_value, shape=shape, dtype=my_float, name=name)

    def create_variable(self, initial_value, shape, name):
        initial = tf.constant(initial_value, shape=shape)
        return tf.Variable(initial, dtype=my_float, name=name)

    def rnn_to_fd(self, RNN):
        pass

    def print_w(self, W):
        edges_list = []

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                for key, value in self.flow_diagram.names_units_map.items():
                    if value == i:
                        infsource = key

                for key, value in self.flow_diagram.names_hidden_map.items():
                    if value == j:
                        rate = key

                edges_list.append((infsource, rate, W[i][j]))
                logging.info('{} -> {} : {}'.format(infsource, rate, W[i][j]))

        return edges_list
