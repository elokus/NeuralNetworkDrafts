# Multivariate Timeseries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf


def custom_data_prep(dataset, target, start, end, window, horizon):
    x = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        x.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(x), np.array(y)


def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(
        f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}', end='\n\n')


data = pd.read_csv("data.csv", parse_dates=["date"])

print(data.dtypes)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x_data = x_scaler.fit_transform(data[["dprice", "nupl"]])
y_data = y_scaler.fit_transform(data[["dprice"]])

hist_window = 30
horizon = 5
train_split = 1500

x_train, y_train = custom_data_prep(
        x_data, y_data, 0, train_split, hist_window, horizon)
x_vali, y_vali = custom_data_prep(
        x_data, y_data, train_split, None, hist_window, horizon)

print('Multiple window of past history')
print(x_train[0])
print('Target horizon')
print(y_train[0])

batch_size = 60
buffer_size = 45
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
val_data = val_data.batch(batch_size).repeat()

lstm_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True),
                                                                       input_shape=x_train.shape[-2:]),
                                         tf.keras.layers.Dense(
                                         20, activation='tanh'),
                                         tf.keras.layers.Bidirectional(
                                         tf.keras.layers.LSTM(150)),
                                         tf.keras.layers.Dense(
                                         20, activation='tanh'),
                                         tf.keras.layers.Dense(
                                         20, activation='tanh'),
                                         tf.keras.layers.Dropout(0.25),
                                         tf.keras.layers.Dense(
                                            units=horizon),
                                         ])


lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary()

model_path = 'Bidirectional_LSTM_Multivariate.h5'
early_stopings = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
callbacks = [early_stopings, checkpoint]
history = lstm_model.fit(train_data, epochs=150, steps_per_epoch=60,
                         validation_data=val_data, validation_steps=30, verbose=1)

plt.figure(figsize=(16, 9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
plt.show()

data_val = x_scaler.fit_transform(data[["dprice", "nupl"]].tail(14))
val_rescaled = data_val.reshape(
                             1, data_val.shape[0], data_val.shape[1])
pred = lstm_model.predict(val_rescaled)
pred_Invers = y_scaler.inverse_transform(pred)
print(pred_Invers)

timeseries_evaluation_metrics_func(data_val["dprice"], pred_Invers[0])

plt.figure(figsize=(16, 9))
plt.plot(list(data_val['dprice']))
plt.plot(list(pred_Invers[0]))
plt.title("Actual vs Predicted")
plt.ylabel("BTC Price Change")
plt.legend(('Actual', 'predicted'))
plt.show()
