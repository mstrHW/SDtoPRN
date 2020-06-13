import tensorflow as tf
from datetime import datetime
import pickle
import numpy as np

from definitions import os, make_directory


class TFBaseModel(object):
    def __init__(self):
        self.model = None

    def train(self, train_X, train_y, valid_data, train_params, model_dir):
        epochs_count = int(train_params['epochs_count'])
        learning_rate = train_params['learning_rate']
        epochs_before_decay = train_params['epochs_before_decay']
        decay_base = train_params['learning_rate_decay']
        early_stopping_patience = train_params['early_stopping_patience']

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     learning_rate,
        #     decay_steps=epochs_before_decay,
        #     decay_rate=decay_base,
        #     staircase=True,
        # )
        # lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        log_dir = os.path.join(model_dir, 'log', 'fit') + datetime.now().strftime("%Y%m%d-%H%M%S")
        make_directory(log_dir)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

        callbacks = [tensorboard_callback, early_stopping_callback]
        history = self.model.fit(train_X, train_y, batch_size=32, validation_data=valid_data, epochs=epochs_count, callbacks=callbacks)

        if model_dir is not None:
            loss_history_file_name = os.path.join(model_dir, 'loss_history')
            with open(loss_history_file_name, 'wb') as file:
                pickle.dump(history.history, file)

        if model_dir is not None:
            model_file = os.path.join(model_dir, 'my_checkpoint')
            self.model.save_weights(model_file)

    def test(self, test_X, model_dir=None):
        if model_dir is not None:
            model_file = os.path.join(model_dir, 'my_checkpoint')
            self.model.load_weights(model_file)
        output = self.model.predict(test_X)

        return output

    def load(self, model_dir):
        model_file = os.path.join(model_dir, 'my_checkpoint')
        self.model.load_weights(model_file)

    def calculate_trainable_parameters(self):
        total_parameters = 0
        for variable in self.model.trainable_variables:
            shape = variable.numpy().shape
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters

        return total_parameters

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
