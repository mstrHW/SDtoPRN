import tensorflow as tf
import numpy as np

from definitions import os
from module.tf_base_model import TFBaseModel


class PRNBlock(tf.keras.Model):
    def __init__(self, parameters):
        super(PRNBlock, self).__init__(name='')
        self.W_xy = parameters['W_xy']
        self.W_ah = parameters['W_ah']
        self.W_ry = parameters['W_ry']
        self.phi_h = parameters['phi_h']
        self.phi_o = parameters['phi_o']

    def call(self, input_tensor, training=False, mask=None):
        ah = tf.matmul(input_tensor, self.W_ah)
        r = self.phi_h(ah)

        ry = tf.matmul(r, self.W_ry)
        xy = tf.matmul(input_tensor, self.W_xy)

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
        self.regularizer = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
        self.reg_loss = self.regularizer(self.parameters['W_ah'])

        model = PRNBlock(parameters)
        # model.add_loss(self.reg_loss)
        model.compile(optimizer='adam',
                      loss=self.custom_loss,
                      metrics=[self.custom_loss])

        self.model = model

    def custom_loss(self, y_actual, y_pred):
        custom_loss = tf.keras.losses.MSE(y_actual, y_pred) + self.reg_loss
        return custom_loss

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

        outputs = []
        last_output = initial_value
        outputs.append(last_output[0].tolist())

        for i in range(iterations_count):
            output = self.model.predict(last_output)
            last_output = output
            outputs.append(last_output[0].tolist())
        outputs = np.array(outputs)
        return outputs
