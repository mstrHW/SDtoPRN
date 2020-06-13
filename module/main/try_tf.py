import tensorflow as tf


class NALU(tf.keras.Model):
    def __init__(self, input_shape, output_shape):
        super(NALU, self).__init__(name='')
        self.eps = 1e-7
        W_shape = (input_shape, output_shape)
        self.W_hat = tf.Variable(tf.random.truncated_normal(W_shape, stddev=0.02))
        self.M_hat = tf.Variable(tf.random.truncated_normal(W_shape, stddev=0.02))
        self.gate = tf.Variable(tf.random.truncated_normal(W_shape, stddev=0.02))

    def call(self, input_tensor, training=False, mask=None):
        # NAC cell
        self.W = tf.tanh(self.W_hat) * tf.sigmoid(self.M_hat)
        a = tf.matmul(input_tensor, self.W)
        # NALU
        m = tf.exp(tf.matmul(tf.math.log(tf.abs(input_tensor) + self.eps), self.W))
        g = tf.sigmoid(tf.matmul(input_tensor, self.gate))
        out = g * a + (1 - g) * m
        return out


class Block(tf.keras.Model):
    def __init__(self, input_shape, output_shape):
        super(Block, self).__init__(name='')
        self.W = tf.Variable(tf.zeros((input_shape, output_shape)))
        # self.W_log = tf.Variable(tf.zeros((input_shape, output_shape)))
        # self.gate = tf.Variable([[0.], [0.], [1.]])
        self.gate = tf.Variable(tf.random.truncated_normal((self.W.shape[0], self.W.shape[-1]), stddev=0.02))

    def call(self, input_tensor, training=False, mask=None):
        x = input_tensor
        shift = 0
        _g = self.gate
        _W = self.W
        # g = self.gate
        _W1 = tf.multiply(_W, _g)
        x_1 = tf.matmul(x, _W1)

        x_2 = tf.math.log(x + shift)
        _W2 = tf.multiply(_W, -_g + 1)
        x_2 = tf.matmul(x_2, _W2)
        x_2 = tf.math.exp(x_2) - shift

        # g = tf.sigmoid(self.gate)
        # print(x_1)
        # print(g)
        # x_1 = tf.multiply(x_1, g)
        # x_2 = tf.multiply(x_2, (-g + 1))

        # g = tf.sigmoid(tf.matmul(input_tensor, self.gate))
        # _gate = tf.nn.sigmoid(self.gate)
        # x_1 = tf.multiply(g, x_1)
        # x_2 = tf.multiply((-g + 1), x_2)

        x = x_1 + x_2
        return x


from itertools import product
import numpy as np
import pandas as pd

np.random.seed(123)
tf.random.set_seed(123)

data_size = 10000
a = np.random.uniform(0, 10, data_size).reshape((data_size, 1))
b = np.random.uniform(0, 10, data_size).reshape((data_size, 1))
c = np.random.uniform(0, 10, data_size).reshape((data_size, 1))

# answer = np.array(list(product(a, b, c)))
answer = np.concatenate([a, b, c], axis=1)
print(answer.shape)
# print(answer)


df = pd.DataFrame(answer, columns=['a', 'b', 'c'])
df['a + 2b'] = df['a'] + 2 * df['b']
df['a * b'] = df['a'] * df['b']
df['a * b ^ 2 + c'] = df['a'] * (df['b'] ** 2) + df['c']
df['a / b + c'] = df['a'] / df['b'] + df['c']
df['(a + b) * c'] = (df['a'] + df['b']) * df['c']
df = df.astype(np.float32)


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df[df.columns] = scaler.fit_transform(df[df.columns])
# df.head(10)


from sklearn.model_selection import train_test_split
train, valid = train_test_split(df, test_size=0.2, random_state=123)

input_columns = ['a', 'b', 'c']
train_X = train[input_columns].values
valid_X = valid[input_columns].values

# target_columns = ['a + b', 'a * b']
target_columns = ['a + 2b', 'a * b', 'a * b ^ 2 + c', 'a / b + c', '(a + b) * c']
target_columns = ['(a + b) * c']
train_y = train[target_columns].values
valid_y = valid[target_columns].values


model = NALU(len(input_columns), len(target_columns))
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])
model.fit(train_X, train_y, validation_data=(valid_X, valid_y), epochs=150)

y_pred = model.predict(valid_X[:5])
print(valid_y[:5], y_pred)
print(model.gate)
print(tf.round(model.gate))
# print(tf.sigmoid(model.gate))
print(model.W)
print(tf.round(model.W))
# print(model.W_log)
# print(tf.multiply(model.W, -model.gate + 1))

