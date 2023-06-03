# LSTM Test Skript
# Importation
# Learning rate scheduler for when we reach plateaus
import predict
from keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Dense
from keras.models import Sequential
import os
import matplotlib.pyplot as plt
import pandas as pd


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                 Data Preperation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from data_prep import *

input_data = pd.read_csv("data.csv",
                         parse_dates=["date"])

feature_col = "USD_price_change_1_day"
date_col = "trade_date"
testsize = 100
seq_len = 60
split = 0.7

train_df, validation_df = data_preperation(
    input_data, feature_col, date_col, testsize, True)
series_prep = Series_Prep(train_df)

window, X_min, X_max = series_prep.make_window(seq_len, split)
print(f'X min={X_min} and max={X_max}')


X_train, X_test, y_train, y_test = series_prep.reshape_window(
    window, train_test_split=split)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                 Building the LSTM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)

# Reset model if we want to re-train with different splits


def reset_weights(model):
    import keras.backend as K
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)


# Epochs and validation split
EPOCHS = 201
validation = 0.05

# Instantiate the model
model = Sequential()

# Add the first layer.... the input shape is (Sample, seq_len-1, 1)
model.add(LSTM(
        input_shape=(seq_len-1, 1), return_sequences=True,
        units=100))

# Add the second layer.... the input shape is (Sample, seq_len-1, 1)
model.add(LSTM(
        input_shape=(seq_len-1, 1),
        units=100))

# Add the output layer, simply one unit
model.add(Dense(
        units=1,
        activation='sigmoid'))

model.compile(loss='mse', optimizer='adam')


# History object for plotting our model loss by epoch
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=validation,
                    callbacks=[rlrop])
# Loss History
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#              Predicting the future
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Creating our future object

future = predict.Predict_Future(
    X_test=X_test, validation_df=validation_df, lstm_model=model)
# Checking its accuracy on our training set
future.predicted_vs_actual(X_min=X_min, X_max=X_max, numeric_colname='feature')
# Predicting 'x' timesteps out
future.predict_future(X_min=X_min, X_max=X_max, numeric_colname='feature',
                      timesteps_to_predict=15, return_future=True)
