import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Reshape, Input, concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from data_preprocessing import get_aggregated_data
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.compose import ColumnTransformer


def get_data():
    df_1D, df_4H, df_5T = get_aggregated_data()
    print(df_1D.head())
    A = df_1D.to_numpy()
    B = df_4H.to_numpy()
    C = df_5T.to_numpy()
    return C  # Open [0], High [1], Low[2], Close[3], Volume[4], VWAP [5]


def train_test_split(data_array, train_test_ratio=0.75):
    train_len = int(len(data_array)*train_test_ratio)
    test_len = len(data_array)-train_len
    train, test = data_array[0:train_len,
                             :], data_array[train_len:len(data_array), :]
    return train, test


def x_y_split(data, pred_column=3, lookback=30, step_size=1):
    data_X, data_Y = [], []
    for i in range(0, len(data) - lookback-step_size, step_size):
        d = i+lookback
        data_X.append(data[i:d, :])
        # +lookback letzten 48 candles predicten close in 4h in der zukunft
        data_Y.append(data[d+lookback, pred_column])
    return np.array(data_X), np.array(data_Y)


def create_model_dnn_64():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='relu'))
    return model


def create_model_dnn_32():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='relu'))
    return model


def dnn_model(model):
    model = create_model_dnn_32()
    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['mse', 'mae'])
    return model


def model_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show()


def prediction_plot(testY, test_predict):
  len_prediction = [x for x in range(len(testY))]
  plt.figure(figsize=(8, 4))
  plt.plot(len_prediction, testY[:], marker='.', label="actual")
  plt.plot(len_prediction, test_predict[:], 'r', label="prediction")
  plt.tight_layout()
  sns.despine(top=True)
  plt.subplots_adjust(left=0.07)
  plt.ylabel('Ads Daily Spend', size=15)
  plt.xlabel('Time step', size=15)
  plt.legend(fontsize=15)
  plt.show()


# data preparation
split_ratio = 0.75
lookback_window = 48
pred_column = 3
steps = 48

train_org, test_org = train_test_split(get_data(), split_ratio)
train_scaler = MinMaxScaler()
test_scaler = MinMaxScaler()

train = train_scaler.fit_transform(train_org)
test = test_scaler.fit_transform(test_org)

pred_minmax = [[test_scaler.data_min_[pred_column]],
               [test_scaler.data_max_[pred_column]]]
pred_scaler = MinMaxScaler()
pred_scaler.fit(pred_minmax)

x_train, y_train = x_y_split(
    train, pred_column, lookback_window, step_size=steps)
x_test, y_test = x_y_split(
    test, pred_column, lookback_window, step_size=steps)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# FIT Model
model = model_dnn()

history = model.fit(x_train, y_train, epochs=40, batch_size=48, verbose=1, validation_data=(
    x_test, y_test), shuffle=False)


train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '
      % (np.sqrt(train_score[1]), train_score[2]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
      % (np.sqrt(test_score[1]), test_score[2]))

model_loss(history)
test_prediction = model.predict(x_test)
prediction_plot(pred_scaler.inverse_transform(y_test.reshape(-1, 1)),
                pred_scaler.inverse_transform(test_prediction.reshape(-1, 1)))
model.save_weights("5T_weights.h5")
