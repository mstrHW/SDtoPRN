import tensorflow as tf

class ElmanRNN(object):

    pass

    name = ''

    def __init__(self):
        self.name = 'ElmanRNN'

    def create_placeholders(self, n_x, n_y):
        X = tf.placeholder(dtype=tf.float32, shape=[None, n_x], name="X")
        Y = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name="Y")

        return X, Y

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def initialize_parameters(self, layers):

        parameters = {}

        parameters['layers'] = layers
        layers_count = len(layers)
        parameters['layers_count'] = layers_count

        for i in range(1, layers_count):

            Wi_name = 'W{}'.format(i)
            bi_name = 'b{}'.format(i)
            Wi = self.weight_variable([layers[i-1], layers[i]], name=Wi_name)
            bi = self.bias_variable([layers[i]], name=bi_name)

            parameters[Wi_name] = Wi
            parameters[bi_name] = bi

        return parameters

    def forward_propagation(self, X, parameters, keep_prob=1):

        hd_last = X
        layers_count = parameters['layers_count']

        for i in range(1, layers_count):

            Wi = parameters['W{}'.format(i)]
            bi = parameters['b{}'.format(i)]

            if i < layers_count - 1:
                hi = tf.nn.sigmoid(tf.matmul(hd_last, Wi) + bi)
                hdi = tf.nn.dropout(hi, keep_prob)
                hd_last = hdi
            else:
                hd_last = tf.matmul(hd_last, Wi) + bi

        h_last = hd_last

        return h_last

    def compute_cross_entropy(self, Ln, Y):

        logits = Ln
        labels = Y

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        return cross_entropy

    def compute_regularization(self, parameters, beta, m):

        regularizers = 0

        for i in range(1, parameters['layers_count']):
            Wi_name = 'W{}'.format(i)
            regularizers += tf.nn.l2_loss(parameters[Wi_name])

        regularization = beta * regularizers / (2 * m)

        return regularization

    def compute_cost(self, cross_entropy, regularization=0):

        cost = tf.reduce_mean(cross_entropy + regularization)

        return cost

    def initialize_optimizer(self, cost, starter_learning_rate, decay_after_epochs, decay_base):
        #don't working
        global_step = tf.Variable(1, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_after_epochs, decay_base, staircase=True)

        optimizer = (
            tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
        )


        return optimizer

    def build_model(self, train, test, layers, learning_rate=0.0001,
                    num_epochs=1500, beta=0, keep_prob_value=1, save_model_file=''):

        tf.reset_default_graph()

        X_train = train.X
        Y_train = train.Y

        X_test = test.X
        Y_test = test.Y

        (m, n_x) = X_train.shape
        n_y = Y_train.shape[1]


        X, Y = self.create_placeholders(n_x, n_y)
        keep_prob = tf.placeholder(tf.float32)

        parameters = self.initialize_parameters(layers)

        Ln = self.forward_propagation(X, parameters, keep_prob)

        cross_entropy = self.compute_cross_entropy(Ln, Y)
        regularization = self.compute_regularization(parameters, beta, m)

        cost = self.compute_cost(cross_entropy, regularization)

        decay_after_epochs = 3000
        decay_base = 0.35
        # optimizer = self.initialize_optimizer(cost, learning_rate, decay_after_epochs, decay_base)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(Ln, 1), tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.InteractiveSession()
        sess.run(init)


        for epoch in range(1, num_epochs):

            _, epoch_cost = sess.run([optimizer, cost],
                                     feed_dict={X: X_train, Y: Y_train, keep_prob: keep_prob_value})

            if epoch % 1000 == 0:
                print("Cost after epoch " + str(epoch) + " : " + str(epoch_cost))
                print("Train Accuracy:", accuracy.eval(feed_dict={X: X_train, Y: Y_train, keep_prob: 1.0}))
                print("Test Accuracy:", accuracy.eval(feed_dict={X: X_test, Y: Y_test, keep_prob: 1.0}))
            if epoch % decay_after_epochs == 0:
                learning_rate = learning_rate * decay_base
                print('Learning rate : {}'.format(learning_rate))

        save_path = saver.save(sess, save_model_file)
        print("Model saved in file: %s" % save_path)

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1.0})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1.0})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        sess.close()

        return train_accuracy, test_accuracy