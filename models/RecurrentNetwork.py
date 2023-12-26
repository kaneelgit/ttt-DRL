import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

tfd = tfp.distributions
tfpl = tfp.layers

#create a convolution block
class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = "same"):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding=padding)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(in_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, padding="same")
        else:
            self.shortcut = lambda x: x

    def call(self, inputs):
        residual = self.shortcut(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = x + residual
        x = tf.nn.relu(x)
        return x


RecurrentNN= tf.keras.Sequential([
    ConvBlock(3, 64, kernel_size = 3),
    ResidualBlock(64, 64),
    ResidualBlock(64, 64),
    ResidualBlock(64, 64),
    ResidualBlock(64, 64),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(9, activation = 'linear')
])