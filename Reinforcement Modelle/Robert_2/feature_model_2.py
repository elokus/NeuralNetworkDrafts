import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model


class EncoderCNN():
    def __init__(self, input_shape, output_len):
        self.input_shape = input_shape
        self.output_len = output_len
        super(EncoderCNN, self).__init__()
        self.encoder = Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv1D(filters=30, kernel_size=3, activation='relu'),
            layers.Conv1D(filters=30, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten()])
        self.decoder = Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation="softmax"),
            layers.Dense(self.output_len, activation='relu')])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    class EncoderDNN():
        def __init__(self, input_shape, output_len):
            self.input_shape = input_shape
            self.output_len = output_len
            super(EncoderCNN, self).__init__()
            self.encoder = Sequential([
                layers.Input(shape=self.input_shape),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Flatten()])
            self.decoder = Sequential([
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(16, activation="softmax"),
                layers.Dense(self.output_len, activation='relu')])
        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
