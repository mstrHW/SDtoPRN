import tensorflow as tf
import numpy as np

from definitions import os
from module.tf_base_model import TFBaseModel


class PRNBlock(tf.keras.Model):
    def __init__(self, parameters, y_size):
        super(PRNBlock, self).__init__(name='')
        self.W_xy = parameters['W_xy']
        self.W_ah = parameters['W_ah']
        self.W_ry = parameters['W_ry']
        self.phi_h = parameters['phi_h']
        self.phi_o = parameters['phi_o']
        self.y_size = y_size

    def call(self, input_tensor, training=False, mask=None):
        a = input_tensor

        ah = tf.matmul(a, self.W_ah)
        r = self.phi_h(ah)

        ry = tf.matmul(r, self.W_ry)
        xy = tf.matmul(input_tensor, self.W_xy)
        # ay = tf.matmul(a, self.W_xy)

        yt_pred = self.phi_o(ry + xy)
        # yt_pred = tf.reshape(yt_pred, [-1, 1, yt_pred.shape[1]])
        preds = [yt_pred]

        for i in range(1, self.y_size):
            a = yt_pred
            ah = tf.matmul(a, self.W_ah)
            r = self.phi_h(ah)

            ry = tf.matmul(r, self.W_ry)
            xy = tf.matmul(a, self.W_xy)
            # ay = tf.matmul(a, self.W_xy)

            yt_pred = self.phi_o(ry + xy)
            # yt_pred = tf.reshape(yt_pred, [-1, 1, yt_pred.shape[1]])
            # yt_pred = tf.reshape(yt_pred, [None, 1, yt_pred.shape[1]])
            preds.append(yt_pred)

        # preds = tf.convert_to_tensor(preds)
        # preds = tf.transpose(preds, [1, 0, 2])
        # preds = tf.concat(preds, axis=1)
        print(preds)
        return preds


class NNModel(TFBaseModel):
    def __init__(self, parameters, y_size=1):
        TFBaseModel.__init__(self)
        self.name = 'MyRNN'

        self.parameters = parameters

        self.input_dim = parameters['units_count']
        self.hidden_dim = parameters['hidden_count']
        self.output_dim = parameters['units_count']

        model = PRNBlock(parameters, y_size)
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

    def get_simulation(self, initial_value, iterations_count, model_dir):
        if model_dir is not None:
            model_file = os.path.join(model_dir, 'my_checkpoint')
            self.model.load_weights(model_file)

        outputs = [initial_value[0]]
        last_output = initial_value
        # outputs.append(initial_value[0].tolist())

        for i in range(iterations_count):
            # print(last_output)
            output = self.model.predict(last_output)
            # print(output)
            # _last_output = last_output[:, 1:, :]
            # _output = output.reshape(1, 1, last_output.shape[2])
            last_output = [output[0, 0].tolist()]
            outputs.append(output[0, 0].tolist())
        outputs = np.array(outputs)
        return outputs