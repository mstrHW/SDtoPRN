import tensorflow as tf
from tensorflow.keras import layers

from module.tf_base_model import TFBaseModel


class BaseNN2Layers(TFBaseModel):

    def __init__(self, hidden, output):
        TFBaseModel.__init__(self)
        model = tf.keras.Sequential()
        # model.add(layers.Dense(hidden, activation='relu'))
        # model.add(layers.Dense(hidden, activation='relu'))
        model.add(layers.Dense(output, activation='relu'))
        model.add(layers.Dense(output, activation='linear'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mse'])
        self.model = model
