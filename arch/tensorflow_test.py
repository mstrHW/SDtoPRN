import tensorflow as tf
import numpy as np


def stack_constants_and_variables():
    train_X = np.asarray([[3.3, 4.4], [5.5, 6.71], [6.93, 4.168], [9.779, 6.182], [7.59, 2.167]])
    train_Y = np.asarray([[1.7], [2.76], [2.09], [3.19], [1.694]])
    n_samples = train_X.shape[0]

    X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    w1 = tf.constant(5, shape=[1], name='w1', dtype=tf.float32)

    initW = tf.truncated_normal([1], stddev=0.1)
    w2 = tf.Variable(initW, name='w2', dtype=tf.float32)

    W = tf.stack([w1, w2])

    initB = tf.truncated_normal([1], stddev=0.1)
    b = tf.Variable(initB, name='b', dtype=tf.float32)

    mm = tf.matmul(X, W)
    pred = tf.add(mm, b)

    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(10):
            sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        print(sess.run(W))

def weight_variable(shape, name, dtype):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=dtype)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name, dtype):
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.Variable(initial, name=name)

def simple_nn():
    my_type_tf = tf.float64
    my_type_np = np.float64

    devide_by = 1e2
    data_preproc = lambda x: x/devide_by
    func = lambda x: np.prod(x, axis=1)
    # func = lambda x: np.sum(x, axis=1)
    multiply_count = 5

    train_X = np.random.randint(1, 100, size=(100, multiply_count))
    train_X = data_preproc(train_X)
    train_Y = func(train_X)
    train_Y = np.reshape(train_Y, [train_X.shape[0], 1])
    # train_X = data_preproc(train_X)

    test_X = np.random.randint(1, 100, size=(100, multiply_count))
    test_X = data_preproc(test_X)
    test_Y = func(test_X)
    test_Y = np.reshape(test_Y, [test_X.shape[0], 1])
    # test_X = data_preproc(test_X)

    phi_in = lambda x: tf.log(x)
    # phi_in = lambda x: x

    phi_h = lambda x: tf.nn.tanh(x)
    # phi_h = lambda x: x

    phi_out = lambda x: tf.exp(x)
    # phi_out = lambda x: x

    n_x = multiply_count
    n_y = 1
    n_h = multiply_count * 10
    # n_h2 = 100
    n_h3 = 5
    X = tf.placeholder(shape=[None, n_x], dtype=my_type_tf)
    Y = tf.placeholder(shape=[None, n_y], dtype=my_type_tf)

    W3_shape = [n_x, n_x]
    _W3 = np.zeros(shape=W3_shape, dtype=my_type_np)
    for i in range(W3_shape[0]):
        _W3[i][i] = 1

    W3 = tf.constant(_W3, dtype=my_type_tf)

    W = weight_variable(shape=[n_x, n_h], name='W', dtype=my_type_tf)
    b = bias_variable(shape=[n_h], name='b', dtype=my_type_tf)

    # W2 = weight_variable(shape=[n_h, n_y], name='W2', dtype=my_type_tf)
    # b2 = bias_variable(shape=[n_y], name='b2', dtype=my_type_tf)
    W2_shape = [n_h, n_y]
    W2 = tf.constant(1.0, shape=W2_shape, dtype=my_type_tf)

    # W3 = weight_variable(shape=[n_h2, n_y], name='W3')
    # b3 = bias_variable(shape=[n_y], name='b3')
    #
    # W4 = weight_variable(shape=[n_h3, n_y], name='W4')
    # b4 = bias_variable(shape=[n_y], name='b4')

    h_in = tf.matmul(X, W3)
    a_in = phi_in(h_in)

    h = tf.matmul(a_in, W) + b
    a = phi_h(h)

    h2 = tf.matmul(a, W2)
    a2 = phi_out(h2)

    # h3 = tf.matmul(a2, W3) + b3
    # a3 = phi_out(h3)
    #
    # h4 = tf.matmul(a3, W4) + b4
    # a4 = phi_out(h4)

    y = a2
    # loss = tf.losses.mean_squared_error(y, Y)
    cost = tf.losses.mean_squared_error(y, Y)
    # cost = tf.losses.sigmoid_cross_entropy(y, Y)
    # loss = tf.square(y - Y)
    # cost = tf.reduce_mean(loss)

    lr = tf.placeholder(dtype=my_type_tf, name="learning_rate")
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    epochs_count = 5e5
    learning_rate = 1e-2
    epochs_before_decay = 0.1
    decay_base = 1/3

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        last_mse = 1
        k = 0
        times = 10
        sess.run(init)
        for i in range(1, int(epochs_count)):
            _, mse, w = sess.run([train_op, cost, W],
                                  feed_dict={X: train_X, Y: train_Y, lr: learning_rate})

            if i % int(epochs_count * 0.01) == 0:
                print(i, mse)

                if last_mse == mse:
                    k += 1
                last_mse = mse
                if k == 5:
                    if times > 0:
                        learning_rate = learning_rate * decay_base
                        print(lr)
                        times -= 1
                        k = 0
                    else:
                        break


            if i % int(epochs_count * epochs_before_decay) == 0:
                learning_rate = learning_rate * decay_base
                times -= 1
                print(lr)

        output, _cost = sess.run([y, cost], feed_dict={X: test_X, Y: test_Y})

    test_cost = sum(np.abs((output-test_Y)/test_Y))/test_Y.shape[0]
    print(test_Y)
    print(output)
    print('test_cost : {}'.format(test_cost))
    print(_cost)


if __name__ == '__main__':
    # stack_constants_and_variables()
    simple_nn()
    pass
