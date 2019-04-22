import tensorflow as tf
import numpy as np

my_float = tf.float64


class NewRNN(object):

    def __init__(self, parameters):
        self.name = 'NewRNN'

        self.parameters = parameters

        self.input_dim = parameters['units_count']
        self.hidden_dim = parameters['hidden_count']
        self.output_dim = parameters['units_count']

        self.build_model()

    def build_model(self):

        self.X, self.Y, self.A = self.create_placeholders(self.input_dim, self.hidden_dim, self.output_dim)
        self.learning_rate = tf.placeholder(dtype=my_float, name="learning_rate")

        # parameters = self.initialize_parameters(self.input_dim, self.hidden_dim, self.output_dim)
        parameters = self.parameters

        self.y_pred = self.rnn_forward(self.X, self.A, parameters)

        self.cost = self.compute_cost(self.y_pred, self.Y)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.saver = tf.train.Saver(save_relative_paths=True)

    def create_placeholders(self, n_x, n_h, n_y=1):

        X = tf.placeholder(dtype=my_float, shape=[None, n_x], name="X")
        Y = tf.placeholder(dtype=my_float, shape=[None, n_y], name="Y")
        A = tf.placeholder(dtype=my_float, shape=[None, n_y], name="A")

        return X, Y, A

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def initialize_parameters(self, n_x, n_h=10, n_y=1):

        parameters = {}

        Wax = self.weight_variable(shape=[n_x, n_h], name='Wax')
        ba = self.bias_variable(shape=[n_h], name='ba')

        Wya = self.weight_variable(shape=[n_h, n_y], name='Wya')
        by = self.bias_variable(shape=[n_y], name='by')

        Waa = tf.constant(1, shape=[1], name='Waa', dtype=my_float)

        parameters['Wax'] = Wax
        parameters['ba'] = ba
        parameters['Wya'] = Wya
        parameters['by'] = by
        parameters['Waa'] = Waa

        return parameters

    def rnn_forward(self, x, at, parameters):

        yt_pred = self.rnn_cell_forward(x, at, parameters)

        return yt_pred

    def rnn_cell_forward(self, xt, a_prev, parameters):

        U = parameters['U']
        W = parameters['W']
        V = parameters['V']

        phi_h = parameters['phi_h']
        phi_o = parameters['phi_o']

        W_aprev = tf.matmul(a_prev, W)
        a_next = phi_h(W_aprev)

        V_anext = tf.matmul(a_next, V)
        U_xt = tf.matmul(xt, U)

        yt_pred = phi_o(V_anext + U_xt)

        return yt_pred

    def rnn_cell_forward2(self, xt, a_prev, parameters):

        U = parameters['U']
        W = parameters['W']
        V = parameters['V']
        W_ay = parameters['W_ay']

        phi_h = parameters['phi_h']
        phi_o = parameters['phi_o']

        W_ah = tf.matmul(a_prev, W)
        a_next = phi_h(W_ah)

        V_anext = tf.matmul(a_next, V)
        U_xt = tf.matmul(xt, U)
        ay = tf.matmul(a_prev, W_ay)

        yt_pred = phi_o(V_anext + U_xt + ay)

        return yt_pred

    def rnn_cell_forward3(self, xt, a_prev, parameters):

        U = parameters['U']
        W = parameters['W']
        V = parameters['V']
        # W_ay = parameters['W_ay']
        W_hh1 = parameters['W_hh1']
        b_hh1 = parameters['b_hh1']

        phi_h = parameters['phi_h']
        phi_o = parameters['phi_o']

        W_ah = tf.matmul(a_prev, W)
        a_next = tf.nn.tanh(W_ah + b_hh1)
        hh1 = tf.matmul(a_next, W_hh1)
        a_next1 = phi_h(hh1)

        a_last = a_next1
        V_anext = tf.matmul(a_last, V)
        U_xt = tf.matmul(xt, U)
        # ay = tf.matmul(a_prev, W_ay)

        yt_pred = phi_o(V_anext + U_xt)

        return yt_pred

    def compute_cost(self, y_pred, y, regularization=0):

        mse = tf.losses.mean_squared_error(y_pred, y)
        rmse = tf.sqrt(mse)
        # mae = tf.reduce_sum(tf.abs(y_pred - y))/y.shape[0]

        loss = mse
        cost = tf.reduce_mean(loss + regularization)

        return cost

    def train(self, train_x, train_y, train_a, train_params, model_file):

        epochs_count = train_params['epochs_count']
        learning_rate = train_params['learning_rate']
        epochs_before_decay = train_params['epochs_before_decay']
        decay_base = train_params['learning_rate_decay']

        with tf.Session() as sess:

            writer = tf.summary.FileWriter('logs', sess.graph)

            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(int(epochs_count)+1):
                _, mse, lr = sess.run([self.train_op, self.cost, self.learning_rate],
                                  feed_dict={self.X: train_x, self.Y: train_y, self.A: train_a, self.learning_rate: learning_rate})
                if i % int(epochs_count * 0.01) == 0:
                    print(i, mse)
                if i % int(epochs_count * epochs_before_decay) == 0:
                    learning_rate = learning_rate * decay_base
                    print(lr)

            writer.close()
            save_path = self.saver.save(sess, model_file)
            print('Model saved to {}'.format(save_path))

    def test(self, test_x, test_a, model_file=None):
        output = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.get_variable_scope().reuse_variables()
            if model_file != None:
                self.saver.restore(sess, model_file)
            output = sess.run(self.y_pred, feed_dict={self.X: test_x, self.A: test_a})
        return output

    def get_weights(self, model_file=None):

        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()

            W = self.parameters['W']
            V = self.parameters['V']
            U = self.parameters['U']

            if model_file != None:
                self.saver.restore(sess, model_file)

            W = W.eval()
            V = V.eval()
            U = U.eval()

            output =\
                {
                    'W': W,
                    'V': V,
                    'U': U
                }

        return output

    def get_simulation(self, initial_value, iterations_count, model_file):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            tf.get_variable_scope().reuse_variables()
            if model_file != None:
                self.saver.restore(sess, model_file)

            outputs = []

            last_a = initial_value[0]
            last_output = initial_value[1]

            outputs.append(last_output[0].tolist())

            for i in range(iterations_count):
                output = sess.run(self.y_pred, feed_dict={self.X: last_output, self.A: last_a})
                last_a = last_output
                last_output = output
                outputs.append(last_output[0].tolist())

        return outputs


if __name__ == '__main__':
    pass