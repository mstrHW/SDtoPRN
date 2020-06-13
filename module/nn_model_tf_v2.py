import tensorflow as tf
import numpy as np

from definitions import os
from module.tf_base_model import TFBaseModel


class ComplexFuncsBlock(tf.keras.Model):
    def __init__(self, input_shape, output_shape):
        super(ComplexFuncsBlock, self).__init__(name='')
        self.W = tf.Variable(tf.zeros((input_shape, output_shape)))
        self.gate = tf.Variable(tf.random.truncated_normal((self.W.shape[0], self.W.shape[-1]), stddev=0.02))

    def call(self, input_tensor, training=False, mask=None):
        x = input_tensor
        shift = 0
        _g = self.gate
        _W = self.W

        _W1 = tf.multiply(_W, _g)
        x_1 = tf.matmul(x, _W1)

        x_2 = tf.math.log(x + shift)
        _W2 = tf.multiply(_W, -_g + 1)
        x_2 = tf.matmul(x_2, _W2)
        x_2 = tf.math.exp(x_2) - shift

        x = x_1 + x_2
        return x


class PRNBlock(tf.keras.Model):
    def __init__(self, parameters):
        super(PRNBlock, self).__init__(name='')
        self.W_xy = parameters['W_xy']
        self.W_ah = parameters['W_ah']
        self.W_ry = parameters['W_ry']
        self.phi_h = parameters['phi_h']
        self.phi_o = parameters['phi_o']
        # self.gate = tf.Variable(tf.random.truncated_normal((self.W_ah.shape[0], self.W_ah.shape[-1]), stddev=0.02))
        self.gate = tf.Variable(tf.zeros((self.W_ah.shape[0], self.W_ah.shape[-1])))
        # self.W_ah = tf.Variable([
        #     [1, 1, 0, 1],
        #     [1, 0, 1, 1],
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        # ], dtype=tf.float32)

    def call(self, input_tensor, training=False, mask=None):
        # _gate = tf.sigmoid(self.gate)
        _gate = self.gate
        _W = self.W_ah

        xy = tf.matmul(input_tensor, self.W_xy)

        w_1 = tf.multiply(_W, _gate)
        ah_1 = tf.matmul(input_tensor, w_1)

        a_log = tf.math.log(input_tensor)
        w_2 = tf.multiply(_W, -_gate + 1)
        ah_log = tf.matmul(a_log, w_2)
        ah_2 = tf.math.exp(ah_log)

        r = ah_1 + ah_2
        # r = self.phi_h(r)

        ry = tf.matmul(r, self.W_ry)

        # yt_pred = self.phi_o(ry + xy)
        yt_pred = ry + xy

        return yt_pred


class NNModel(TFBaseModel):
    def __init__(self, parameters):
        TFBaseModel.__init__(self)
        self.name = 'MyRNN'

        self.parameters = parameters

        self.input_dim = parameters['units_count']
        self.hidden_dim = parameters['hidden_count']
        self.output_dim = parameters['units_count']

        model = PRNBlock(parameters)
        import tensorflow.keras.backend as kb

        def rmse(y_actual, y_pred):
            return kb.sqrt(kb.mean(kb.pow((y_actual - y_pred), 2)))

        def mae(y_actual, y_pred):
            return kb.mean(kb.abs((y_actual - y_pred)))

        model.compile(optimizer='RMSprop',
                      loss='mse',
                      metrics=['mse'])

        self.model = model

    def get_weights(self, model_dir=None):
        W = self.parameters['W_ah']
        V = self.parameters['W_xy']
        U = self.parameters['W_ry']

        if model_dir is not None:
            model_file = os.path.join(model_dir, 'my_checkpoint')
            self.model.load_weights(model_file)

        W = W.numpy()
        V = V.numpy()
        U = U.numpy()

        output =\
            {
                'W': W,
                'V': V,
                'U': U
            }

        return output
