import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import datetime

class Baseline(tf.keras.Model):
  def __init__(self, label_index='label'):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]



#--------Multi Step DENSE LAYERS--------
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
    ])
#-------LINEAR MODEL--------
# linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

#--------Convolution neural network--------
cnn = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(12,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

cnn_optimal = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=128,
                           kernel_size=12,
                           activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv1D(filters=64,
                           kernel_size=12,
                           activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling1D(pool_size=2, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='sigmoid'),
    tf.keras.layers.Dense(units=1),
    tf.keras.layers.Reshape([1, -1]),
])

#-------LSTM MODEL--------
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

lstm_optimal= tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(126, return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(126, activation="tanh"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation="tanh"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="sigmoid"),
    tf.keras.layers.Dropout(0.5),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

def compile_and_fit(model, window, patience=5, MAX_EPOCHS=20, optimizer="adam"):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    patience=patience,
                                                    mode='max')
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=bce, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val, callbacks=[tensorboard_callback])
     #                 , callbacks=[early_stopping])
    return history

