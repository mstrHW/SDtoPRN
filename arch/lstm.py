import tensorflow as tf
from tensorflow.keras import layers

from module.tf_base_model import TFBaseModel, os, np


class LSTMModel(TFBaseModel):

    def __init__(self, input_shape, hidden, output):
        TFBaseModel.__init__(self)
        model = tf.keras.Sequential()
        model.add(layers.LSTM(hidden, input_shape=input_shape))
        model.add(layers.Dense(output))
        # model.add(layers.Dense(output, activation='linear'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mse'])
        self.model = model

    def get_simulation(self, initial_value, iterations_count, model_dir):
        if model_dir is not None:
            model_file = os.path.join(model_dir, 'my_checkpoint')
            self.model.load_weights(model_file)

        outputs = initial_value[0].tolist()
        last_output = initial_value
        # outputs.append(initial_value[0].tolist())

        for i in range(iterations_count):
            output = self.model.predict(last_output)
            _last_output = last_output[:, 1:, :]
            _output = output.reshape(1, 1, last_output.shape[2])
            last_output = np.concatenate([_last_output, _output], axis=1)
            outputs.append(output[0].tolist())
        outputs = np.array(outputs)
        return outputs