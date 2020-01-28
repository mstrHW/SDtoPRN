from definitions import tf
import numpy as np
from tqdm import tqdm
import logging

my_float = tf.float64


class NNModelLog(object):

    def __init__(self, parameters):
        self.name = 'MyRNN'

        self.parameters = parameters

        self.input_dim = parameters['units_count']
        self.hidden_dim = parameters['hidden_count']
        self.output_dim = parameters['units_count']

        self.build_model()

    def build_model(self):

        self.X, self.Y = self.create_placeholders(self.input_dim, self.hidden_dim, self.output_dim)
        self.learning_rate = tf.placeholder(dtype=my_float, name="learning_rate")

        # parameters = self.initialize_parameters(self.input_dim, self.hidden_dim, self.output_dim)
        parameters = self.parameters

        self.y_pred, regularizers = self.rnn_forward(self.X, parameters)

        self.cost = self.compute_cost(self.y_pred, self.Y, regularizers)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.saver = tf.train.Saver(save_relative_paths=True)

    def create_placeholders(self, n_x, n_h, n_y=1):

        X = tf.placeholder(dtype=my_float, shape=[None, n_x], name="X")
        Y = tf.placeholder(dtype=my_float, shape=[None, n_y], name="Y")

        return X, Y

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

    def rnn_forward(self, x, parameters):

        at = x
        yt_pred, regularizers = self.rnn_cell_forward(x, at, parameters)

        return yt_pred, regularizers

    def rnn_cell_forward(self, xt, a_prev, parameters):

        W_xy = parameters['W_xy']

        self.W_ah = parameters['W_ah']
        b_ah = parameters['b_ah']
        self.W_ah_log = tf.Variable(self.W_ah.initialized_value())
        b_ah_log = tf.Variable(b_ah.initialized_value())

        W_ry = parameters['W_ry']

        phi_h = parameters['phi_h']
        phi_o = parameters['phi_o']

        ah = tf.matmul(a_prev, self.W_ah) + b_ah

        a_prev_log = tf.log(a_prev)
        ah_log = tf.matmul(a_prev_log, self.W_ah_log) + b_ah_log
        ah_log = tf.exp(ah_log)

        r = phi_h(ah + ah_log)

        # ry = tf.matmul(ah, W_ry)
        ry_log = tf.matmul(ah_log, W_ry)
        xy = tf.matmul(xt, W_xy)

        yt_pred = phi_o(ry_log + xy)

        regularizers = tf.cast(tf.nn.l2_loss(self.W_ah) + tf.nn.l2_loss(self.W_ah_log), tf.float32)

        return yt_pred, regularizers

    def compute_cost(self, y_pred, y, regularizers, regularization=0):

        # ez_nan = tf.abs((y_pred - y))/y
        mse = tf.losses.mean_squared_error(y_pred, y)
        rmse = tf.sqrt(mse)
        mae = tf.abs(y_pred - y)/y

        loss = mse + regularization * regularizers
        cost = tf.reduce_mean(loss)

        return cost

    def train(self, train_x, train_y, train_params, model_file):

        epochs_count = train_params['epochs_count']
        learning_rate = train_params['learning_rate']
        epochs_before_decay = train_params['epochs_before_decay']
        decay_base = train_params['learning_rate_decay']

        with tf.Session() as sess:

            writer = tf.summary.FileWriter('logs', sess.graph)

            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())

            for i in range(0, int(epochs_count)):
                _, mse, lr = sess.run([self.train_op, self.cost, self.learning_rate],
                                  feed_dict={self.X: train_x, self.Y: train_y, self.learning_rate: learning_rate})

                if i % max(int(epochs_count * 0.01), 1) == 0:
                    print(i, mse)
                if i % max(int(epochs_count * epochs_before_decay), 1) == 0:
                    learning_rate = learning_rate * decay_base
                    print(lr)

            writer.close()
            save_path = self.saver.save(sess, model_file)
            print('Model saved to {}'.format(save_path))

    def train_batches(self, train_x, train_y, train_params, model_file):

        epochs_count = train_params['epochs_count']
        learning_rate = train_params['learning_rate']
        epochs_before_decay = train_params['epochs_before_decay']
        decay_base = train_params['learning_rate_decay']

        with tf.Session() as sess:

            writer = tf.summary.FileWriter('logs', sess.graph)

            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())

            def get_batches(lst, n=64):
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]

            batches_count = len(train_x)

            for i in tqdm(range(1, int(epochs_count))):
                mse = 0
                lr = 0
                count = 0
                for train_x_batch, train_y_batch in zip(get_batches(train_x), get_batches(train_y)):
                    _, _mse, _lr, _wah, _wah_log = sess.run([self.train_op, self.cost, self.learning_rate, self.W_ah, self.W_ah_log],
                                            feed_dict={self.X: train_x_batch, self.Y: train_y_batch,
                                                       self.learning_rate: learning_rate})
                    mse += _mse
                    lr = _lr

                    # print(mse)
                    # print(_wah)
                    # count += 1
                    # if count == 2:
                    #     return

                    # print('here')
                mse /= batches_count

                if i % max(int(epochs_count * 0.01), 1) == 0:
                    print(i, mse)
                if i % max(int(epochs_count * epochs_before_decay), 1) == 0:
                    learning_rate = learning_rate * decay_base
                    print(lr)
            print(mse)
            print(_wah)
            print(_wah_log)

            writer.close()
            save_path = self.saver.save(sess, model_file)
            print('Model saved to {}'.format(save_path))

    def test(self, test_x, model_file=None):
        output = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.get_variable_scope().reuse_variables()
            if model_file is not None:
                self.saver.restore(sess, model_file)
            output = sess.run(self.y_pred, feed_dict={self.X: test_x})
        return output

    def test_batches(self, test_x, model_file=None):
        output = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.get_variable_scope().reuse_variables()
            if model_file != None:
                self.saver.restore(sess, model_file)
            output = sess.run(self.y_pred, feed_dict={self.X: test_x})
        return output

    def get_weights(self, model_file=None):

        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()

            W = self.parameters['W_ah']
            V = self.parameters['W_xy']
            U = self.parameters['W_ry']

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

    def get_simulation_step(self, initial_value, iterations_count, sess):
        outputs = []
        last_output = initial_value
        outputs.append(last_output[0].tolist())

        for i in range(iterations_count):
            output = sess.run(self.y_pred, feed_dict={self.X: last_output})
            last_output = output
            outputs.append(last_output[0].tolist())
        outputs = np.array(outputs)
        return outputs

    def get_simulation(self, initial_value, iterations_count, model_file):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            tf.get_variable_scope().reuse_variables()

            if model_file != None:
                self.saver.restore(sess, model_file)

            output = self.get_simulation_step(initial_value, iterations_count, sess)

        return output

    def get_batches_simulation(self, test_x, model_file):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            tf.get_variable_scope().reuse_variables()
            if model_file != None:
                self.saver.restore(sess, model_file)

            outputs = []
            test_count = len(test_x)
            for x, i in zip(test_x, range(test_count)):
                if len(x) > 0:
                    initial_value = x[0]
                    initial_value = np.reshape(initial_value, [1, initial_value.shape[0]])
                    iterations_count = x.shape[0]-1
                    output = self.get_simulation_step(initial_value, iterations_count, sess)
                    outputs.append(output)

                if i % int(test_count * 0.01) == 0:
                    print('{}% of {} batches was simulated'.format(int(i / test_count * 100), test_count))

        outputs = np.array(outputs)
        return outputs

    def calculate_trainable_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        logging.info('total_parameters: {}'.format(total_parameters))
