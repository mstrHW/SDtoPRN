import tensorflow as tf
import numpy as np

class ElmanNetwork(object):

    name = ''

    def __init__(self, input_dim, hidden_dim, output_dim, seq_size=5):
        self.name = 'ElmanNetwork'

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_size = seq_size

        self.build_model()

    def build_model(self):

        self.seq_size = self.seq_size

        self.X, self.Y = self.create_placeholders(self.input_dim, self.hidden_dim, self.output_dim)
        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

        parameters = self.initialize_parameters(self.input_dim, self.hidden_dim, self.output_dim)
        self.parameters = parameters

        self.y_pred = self.rnn_forward(self.X, parameters)[1]

        self.cost = self.compute_cost(self.y_pred, self.Y)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.saver = tf.train.Saver()

    def create_placeholders(self, n_x, n_h, n_y=1):

        X = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_size, n_x], name="X")
        Y = tf.placeholder(dtype=tf.float32, shape=[self.seq_size, None, n_y], name="Y")
        self.a0 = tf.placeholder(dtype=tf.float32, shape=[None, n_h], name="a0")
        self.y0 = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name="y0")

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

        Waa = tf.constant(1, shape=[1], name='Waa', dtype=tf.float32)

        parameters['Wax'] = Wax
        parameters['ba'] = ba
        parameters['Wya'] = Wya
        parameters['by'] = by
        parameters['Waa'] = Waa

        return parameters

    def rnn_forward(self, x, parameters):

        caches = []

        m, n_x = tf.shape(x)[0], self.input_dim
        n_a, n_y = self.hidden_dim, self.output_dim

        a = []
        y_pred = []
        a_next = []

        a.append(self.a0)
        y_pred.append(self.y0)

        for t in range(1, self.seq_size):
            xt = x[:, t - 1, :] # TODO: ВОПРОС 1
            at = a[t - 1]
            a_next, yt_pred, cache = self.rnn_cell_forward(xt, at, parameters)
            a.append(a_next)
            y_pred.append(yt_pred)
            caches.append(cache)

        caches = (caches, x)

        # y_pred = tf.reshape(y_pred, [-1, 1, self.seq_size]) # TODO: ВОПРОС 2

        return a, y_pred, caches

    def rnn_cell_forward(self, xt, a_prev, parameters):

        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        # Waa_aprev = tf.matmul(a_prev, Waa)
        Waa_aprev = a_prev * Waa
        Wax_xt = tf.matmul(xt, Wax)
        a_next = tf.nn.tanh(Waa_aprev + Wax_xt + ba)
        # a_next = Waa_aprev + Wax_xt + ba
        # yt_pred = tf.nn.softmax(tf.matmul(a_next, Wya) + by)
        yt_pred = tf.matmul(a_next, Wya) + by

        cache = (a_next, a_prev, xt, parameters)

        return a_next, yt_pred, cache

    def compute_cost(self, y_pred, y, regularization=0):

        standart_deviation = tf.square(y_pred - y)
        cost = tf.reduce_mean(standart_deviation + regularization)

        return cost

    def train(self, train_x, train_y, a0, y0, learning_rate, model_file):
        decay_base = 1/3
        with tf.Session() as sess:

            writer = tf.summary.FileWriter('logs', sess.graph)

            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1, 500000):
                _, mse, lr = sess.run([self.train_op, self.cost, self.learning_rate],
                                  feed_dict={self.X: train_x, self.Y: train_y, self.a0: a0, self.y0: y0, self.learning_rate: learning_rate})
                if i % 10000 == 0:
                    print(i, mse)
                    # print(self.parameters['Waa'].eval(), self.parameters['Wax'].eval())
                if i % 100000 == 0:
                    learning_rate = learning_rate * decay_base
                    print(lr)

            writer.close()
            save_path = self.saver.save(sess, model_file)
            print('Model saved to {}'.format(save_path))

    def test(self, test_x, a0, y0, model_file):
        output = []
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, model_file)
            output = sess.run(self.y_pred, feed_dict={self.X: test_x, self.a0: a0, self.y0: y0})
        return output

def print_y(y):
    result = np.array(y)
    new_result = np.zeros([result.shape[1], result.shape[2], result.shape[0]])
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                new_result[j][k][i] = result[i][j][k]
    print(new_result)

def demo1():

    input_dim = 1
    hidden_dim = 16
    output_dim = 1
    seq_size = 4
    predictor = ElmanNetwork(input_dim, hidden_dim, output_dim, seq_size)

    model_file = '../models/model1.ckpt'

    train_x = np.array(np.random.rand(10000, seq_size, input_dim) * 10000, dtype=np.int32)
    train_y = np.zeros([seq_size, train_x.shape[0], output_dim])
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[1]):
            train_y[j][i][0] = sum(train_x[i][:j + 1][0])

    a0 = np.zeros([train_x.shape[0], hidden_dim])
    y0 = np.zeros([train_x.shape[0], output_dim])
    # predictor.train(train_x, train_y, a0, y0, 1e-3, model_file)

    test_x = [[[1], [2], [3], [4]],
              [[1], [3], [6], [10]]]
    test_x = np.array(test_x)

    a0 = np.zeros([test_x.shape[0], hidden_dim])
    y0 = np.zeros([test_x.shape[0], output_dim])
    result = predictor.test(test_x, a0, y0, model_file)

    print_y(result)

def demo2():

    input_dim = 4
    hidden_dim = 16
    output_dim = 4
    seq_size = 4
    predictor = ElmanNetwork(input_dim, hidden_dim, output_dim, seq_size)

    model_file = '../models/model2.ckpt'

    m = 10000
    train_x = np.zeros([m, seq_size, input_dim])
    train_y = np.zeros([seq_size, m, output_dim])
    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[1]):
            for z in range(train_x.shape[2]):
                train_x[i][j][z] = j+i

    for i in range(train_x.shape[0]):
        for j in range(train_x.shape[1]):
            for z in range(train_x.shape[2]):
                train_y[j][i][z] = train_x[i][j][z] + 1

    a0 = np.zeros([train_x.shape[0], hidden_dim])
    y0 = np.zeros([train_x.shape[0], output_dim])
    predictor.train(train_x, train_y, a0, y0, 1e-3, model_file)

    test_x = [[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
              [[100500, 100500, 100500, 100500], [100501, 100501, 100501, 100501], [100502, 100502, 100502, 100502],
               [100503, 100503, 100503, 100503]]]
    test_x = np.array(test_x)

    a0 = np.zeros([test_x.shape[0], hidden_dim])
    y0 = np.zeros([test_x.shape[0], output_dim])
    result = predictor.test(test_x, a0, y0, model_file)

    print_y(result)

if __name__ == '__main__':
    # demo1()
    demo2()