import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model


class EncoderCNN(Model):
    def __init__(self, in_shape, output_len):
        self.in_shape = in_shape
        self.output_len = output_len
        super(EncoderCNN, self).__init__()
        self.encoder = Sequential([
            layers.Input(shape=self.in_shape),
            layers.Conv1D(filters=30, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(filters=30, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten()])
        self.decoder = Sequential([
            layers.Dense(64, activation='tanh'),
            layers.Dropout(0.3),
            layers.Dense(32, activation="softmax"),
            layers.Dense(self.output_len, activation='relu')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        tf.summary.histogram("outputs", decoded)
        return decoded


class EncoderDNN(Model):
    def __init__(self, in_shape, output_len):
        self.in_shape = in_shape
        self.output_len = output_len
        super(EncoderDNN, self).__init__()
        self.encoder = Sequential([
            layers.Input(shape=self.in_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Flatten()])
        self.decoder = Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(self.output_len, activation='relu')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        tf.summary.histogram("outputs", decoded)
        return decoded
