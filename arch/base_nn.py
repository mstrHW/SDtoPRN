from definitions import tf
import logging

from tensorflow.keras import layers
import numpy as np


class BaseNN(object):
    def __init__(self, hidden, output):
        model = tf.keras.Sequential()
        # model.add(layers.Dense(hidden, activation='relu'))
        # model.add(layers.Dense(hidden, activation='relu'))
        model.add(layers.Dense(output, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                      loss='mse',
                      metrics=['accuracy'])
        self.model = model

    def train(self, train_X, train_Y, train_params, model_file=None):
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file,
        #                                                  save_weights_only=True,
        #                                                  verbose=1)
        epochs_count = int(train_params['epochs_count'])
        learning_rate = train_params['learning_rate']
        epochs_before_decay = train_params['epochs_before_decay']
        decay_base = train_params['learning_rate_decay']

        self.model.fit(train_X, train_Y, epochs=epochs_count)

        if model_file is not None:
            self.model.save_weights(model_file)

    def test(self, test_X, model_file=None):
        if model_file is not None:
            self.model.load_weights(model_file)
        output = self.model.predict(test_X)
        # initial_value = test_X[0].reshape(1, test_X.shape[1])
        # output = np.concatenate((initial_value, output), axis=0)

        return output

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
