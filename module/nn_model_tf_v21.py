import tensorflow as tf
import numpy as np

from definitions import os
from module.tf_base_model import TFBaseModel


class PRNBlock(object):
    def __init__(self, parameters):
        super(PRNBlock, self).__init__(name='')
        self.W_xy = parameters['W_xy']
        self.W_ah = parameters['W_ah']
        self.W_ry = parameters['W_ry']
        self.phi_h = parameters['phi_h']
        self.phi_o = parameters['phi_o']
        self.W_ah_log = parameters['W_ah_log']
        # self.W_ah_log = tf.identity(self.W_ah)
        # self.gate = tf.Variable([0.0 for i in range(self.W_ah.shape[1])])
        self.gate = tf.Variable(tf.random.truncated_normal((self.W_ah.shape[0], self.W_ah.shape[-1]), stddev=0.02))

    def block1(self, input_tensor):
        w_1 = tf.multiply(self.W_ah, self.gate)
        ah_1 = tf.matmul(input_tensor, w_1)

        a_log = tf.math.log(input_tensor)
        w_2 = tf.multiply(self.W_ah, -self.gate + 1)
        ah_log = tf.matmul(a_log, w_2)
        ah_2 = tf.math.exp(ah_log)

        r = ah_1 + ah_2
        r = self.phi_h(r)
        r = tf.keras.layers.BatchNormalization()(r)

        return r

    def block2(self, input_tensor, r):
        xy = tf.matmul(input_tensor, self.W_xy)
        ry = tf.matmul(r, self.W_ry)
        yt_pred = self.phi_o(ry + xy)

        return yt_pred


class NNModel(TFBaseModel):
    def __init__(self, parameters):
        TFBaseModel.__init__(self)
        self.name = 'MyRNN'

        self.parameters = parameters

        self.input_dim = parameters['units_count']
        self.hidden_dim = parameters['hidden_count']
        self.output_dim = parameters['units_count']

        model = tf.keras.Sequential()
        model.add(PRNBlock1(parameters))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(PRNBlock2(parameters))

        import tensorflow.keras.backend as kb

        def rmse(y_actual, y_pred):
            return kb.sqrt(kb.mean(kb.pow((y_actual - y_pred), 2)))

        def mae(y_actual, y_pred):
            return kb.mean(kb.abs((y_actual - y_pred)))

        model.compile(optimizer='adam',
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
