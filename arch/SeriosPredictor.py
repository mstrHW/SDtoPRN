import tensorflow as tf

class SeriesPredictor(object):

    def __init__(self, input_dim, seq_size, hidden_dim=10):

        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')

        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = tf.contrib.rnn.BasicRNNCell(self.hidden_dim)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)

        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])

        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)

        return out

    def train(self, train_x, train_y):
        with tf.Session() as sess:

            writer = tf.summary.FileWriter('/logs/test1', sess.graph)

            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1, 1001):
                _, mse = sess.run([self.train_op, self.cost],
                                  feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print(i, mse)

            writer.close()
            save_path = self.saver.save(sess, '../models/model.ckpt')
            print('Model saved to {}'.format(save_path))

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, '../models/model.ckpt')
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            print(output)

if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=10)
    train_x = [[[1], [2], [5], [6]],
               [[5], [7], [7], [8]],
               [[3], [4], [5], [7]]]

    train_y = [[1, 3, 7, 11],
               [5, 12, 14, 15],
               [3, 7, 9, 12]]

    predictor.train(train_x, train_y)

    test_x = [[[1], [2], [3], [4]],
              [[4], [5], [6], [7]]]

    predictor.test(test_x)

