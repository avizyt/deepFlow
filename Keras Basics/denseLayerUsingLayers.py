import tensorflow as tf
from tensorflow import keras


class SimpleDense(keras.layers.Layer):

    def __init__(self, units, activation=None):
        super().__init__()
        self.W = None
        self.b = None
        self.units = units
        self.activation = activation

    # weight creation
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, self.units),
            initializer="random_normal"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros"
        )

    # forward pass
    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


dense1 = SimpleDense(units=32, activation=tf.nn.relu)
input_data = tf.random.normal(shape=(2, 784))
output_tensor = dense1(input_data)
print(output_tensor.shape)