import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import parameter.param as param
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import io
from datetime import datetime


def frac_diff(df, timeframes):
    order = 0.3
    cutoff = 2277
    # cutoff for order 0.4 = 1460
    for column in df:
        series = df[column]
        df[column] = ts_differencing(series, order, cutoff)
    print("frational difference was applied to DataFrame")
    return df


def agg_timeframes(df, timeframes):
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum', 'VWAP': 'mean'}
    out = []
    for interval in timeframes:
        frames = df.resample(interval).agg(agg_dict)
        out.append(frames)
    print(f"Time Aggregation with {timeframes} was applied to DataFrame")
    return out


def log_diff(df, timeframes):
    for column in df:
        series = df[column]
        df[column] = np.log(series)
    print("Log difference was applied to DataFrame")
    return df


def minmax_scaler(df, timeframes):
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    print("MinMax Scalling was applied to DataFrame")
    return df


#fractional_differencing:
def getWeights(d, lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w = [1]
    for k in range(1, lags):
        w.append(-w[-1]*((d-k+1))/k)
    w = np.array(w).reshape(-1, 1)
    return w


def ts_differencing(series, order, cutoff):
    # return the time series resulting from (fractional) differencing
    # for real orders order up to lag_cutoff coefficients
    #cutoff 0.3 = 2277, 0.4=1460
    weights = getWeights(order, cutoff)
    res = 0
    for k in range(cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[cutoff:]

#batch functions


def data_batch(data_dict, orig_dict, param_dict):
    #---description---#
    #prepare data input for Neurol Network processing
    #multi input timeframes are aggregatet according to parameter.param file
    # parameter:
    # window_size for each timeframe
    # key_timeframe
    # starting point is declared in parameter file so that each timeframe window fit
    # for the first value
    # e.g. 30 days on daily frame window size means that we have to skip the first
    #30 days on 5 Min frame
    # step_parameter: e.g. key timeframe == 5min (aka 5T):  window  per step "5T" +1
    # each 12 steps "1H" +1, each 48 steps "4H" +1 and each 288 steps "1D" +1
    key_timeframe = param_dict["key_timeframe"]
    steps = param_dict["steps"]
    prediction_frame = param_dict["prediction_frame"]
    window_parameter = param_dict["window_parameter"]
    start_windows = window_parameter["start_windows"]
    step_parameter = window_parameter["step_parameter"][key_timeframe]
    window_sizes = window_parameter["window_sizes"]
    price_data = orig_dict[key_timeframe].reset_index().drop(
        ["VWAP", "Volume"], axis=1).to_numpy()
    if param_dict["predict_from_original"]:
        pred_data = orig_dict[key_timeframe].reset_index().drop(
            ["VWAP", "Volume"], axis=1).to_numpy()[:, param_dict["prediction_column"]]
    else:
        pred_data = data_dict[key_timeframe].reset_index().drop(
            ["VWAP", "Volume"], axis=1).to_numpy()[:, param_dict["prediction_column"]]
    array_dict = {key: value.to_numpy() for key, value in data_dict.items()}
    window_data = []
    step_price = []
    step_pred = []
    step_dict = {}
    for key in array_dict.keys():
        step_dict[key] = []
    if steps == 0:
        steps = len(data_dict[key_timeframe]) - \
                    start_windows[key_timeframe] - prediction_frame
    for step in range(steps):
        step_counter = step + 1
        for key, array in array_dict.items():
            start = start_windows[key]
            end = start_windows[key]+window_sizes[key]
            #print(f"{step_counter}: {key} starts: {start} ends {end}")
            data = array[start:end]
            step_dict[key].append(data)
            if key == key_timeframe:
                step_price.append(price_data[end-1, :])
                step_pred.append(pred_data[end-1])
                if step_counter == steps:
                    for j in range(prediction_frame):
                        start_windows[key] += 1
                        start = start_windows[key]
                        end = start_windows[key]+window_sizes[key]
                        step_price.append(price_data[end-1, :])
                        step_pred.append(pred_data[end-1])
            if step_counter % step_parameter[key] == 0:
                start_windows[key] += 1
    for key, value in step_dict.items():
        frame_data = np.array(value)
        window_data.append(frame_data)
        print(
            f"batch data of {key} append to input list with shape {frame_data.shape} and type {frame_data.dtype}")
    return window_data, step_pred, step_price


def predict_updown(last, new):
    x = new - last
    if x > 0:
        return 1
    else:
        return -1


def predict_log(last, new):
    x = np.log(new) - np.log(last)
    return x


def predict_next(last, new):
    return new

# parameter_dict = {"prediction_function": uf.predict_updown,
#                   "prediction_frame": 6, "steps": 10}


def prediction_generator(price_data, parameter):
    #:param: pred_frame how many timesteps should be predicted
    #:param: function for price representation
    #:param: steps length of output_array -> shape(steps, pred_frame)
    function = parameter["prediction_function"]
    pred_frame = parameter["prediction_frame"]
    steps = parameter["steps"]
    prediction = []
    for i in range(steps):
        y_step = []
        for j in range(pred_frame):
            n = i+j
            last_price = price_data[n]
            new_price = price_data[n+1]
            y = function(last_price, new_price)
            y_step.append(y)
        prediction.append(y_step)
    return np.array(prediction).reshape(steps, pred_frame)


#model debuging


def analyze_inputs_outputs(train_x, test_x, train_y, test_y):
    i = 0
    print("--------------------------train data input----------------")
    print("----------------------------------------------------------")
    print(f">>>>>>>>>>>>>       {len(train_x)} INPUTs registrated  ")
    print(f">>>>>>>>>>>>>       analyzing Shape for each INPUT  ")
    print("----------------------------------------------------------")
    print(".")
    for input in train_x:
        print(
            f">>>>>>>>>>>>> {i+1}. Input >>>> with shape {input.shape} and type {input.dtype}")
        print(".")
        i += 1
    print(".")
    print("--------------------------valid data input----------------")
    print("----------------------------------------------------------")
    print(f">>>>>>>>>>>>>       {len(test_x)} vaildation INPUTs registrated  ")
    print(f">>>>>>>>>>>>>       analyzing Shape for each validation INPUT  ")
    print(".")
    i = 0
    for input in test_x:
        print(
            f">>>>>>>>>>>>> {i+1}. Input >>>> with shape {input.shape} and type {input.dtype}")
        print(".")
        i += 1
    print(".")
    print("--------------------------train data output----------------")
    print("-----------------------------------------------------------")
    print(".")
    print(
        f">>>>>>>>>>>>> {i+1}. Output >>>> with shape {train_y.shape} and type {train_y.dtype}")
    print(".")
    print("--------------------------test data output-----------------")
    print("-----------------------------------------------------------")
    print(".")
    print(
        f">>>>>>>>>>>>> {i+1}. Output >>>> with shape {test_y.shape} and type {test_y.dtype}")
    print(".")

    # def prediction_batch(step_price, param_dict):
    #     """prepare prediction data after data_batch, prediction_frame = 1 means predict next candle..."""
    #     #parameter: predict price_change in %
    #     prediction_frame = param_dict["prediction_frame"]
    #     prediction_column = param_dict["prediction_column"]
    #     prediction = []
    #     steps = len(step_price) - 1
    #     for i in range(steps):
    #         last_price = step_price[i][4]
    #         new_price = step_price[i+1][4]
    #         delta = (new_price - last_price)/last_price
    #         prediction.append(delta)
    #     if prediction_frame > 1:
    #         values = prediction
    #         prediction = []
    #         for i in range(steps-prediction_frame+1):
    #             prediction_window = values[i:i+prediction_frame]
    #             prediction.append(prediction_window)
    #     return prediction


def train_test_split_mlp(data, num_inputs=4, split=0.75):
    #:description: prepares trainint/testing data based on multiple inputs default 4
    train = []
    test = []
    if type(data[0]) == float:
        split = int(len(data)*split)
        return np.array(data[:split]), np.array(data[split:])
    split = int(data[0].shape[0]*split)
    for i in range(num_inputs):
        train.append(data[i][:split, :])
        test.append(data[i][split:, :])
    return train, test


def train_test_split_single(data, target_input=0, split=0.75):
    #:description: prepares trainint/testing data based on multiple inputs default 4
    print(f"Dimensions: {type(data)}")
    if type(data) != list:
        split = int(len(data)*split)
        if data.ndim == 1:
            return np.array(data[:split]), np.array(data[split:])
        else:
            return np.array(data[:split, :]), np.array(data[split:, :])
    split = int(data[0].shape[0]*split)
    train = data[target_input][:split, :]
    test = data[target_input][split:, :]
    return train, test
#model evalution


def compile_and_evaluate(model, train_x, test_x, train_y, test_y, param):
    epochs = param["epochs"]
    batch_size = param["batch_size"]
    name = param["name"]
    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['mse', 'mae'])
    model.summary()
    # uf.analyze_inputs_outputs(train_x, test_x, train_y, test_y)
    tb_callback = tf.keras.callbacks.TensorBoard(
        'data/tensorboard', update_freq=1)
    history = model.fit(train_x, train_y, epochs=epochs, verbose=1,
                        validation_data=(test_x, test_y), batch_size=batch_size, callbacks=[tb_callback])

    train_score = model.evaluate(
        train_x, train_y, verbose=1, callbacks=[tb_callback])
    print(f"----------{name}-------------")
    print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '
          % (np.sqrt(train_score[1]), train_score[2]))
    test_score = model.evaluate(
        test_x, test_y, verbose=0, callbacks=[tb_callback])
    print('Test Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
          % (np.sqrt(test_score[1]), test_score[2]))
    #
    #model_loss(history)
    test_prediction = model.predict(test_x)
    if len(test_y[0]) > 1:
        prediction_plot_multi_timeframe(test_y, test_prediction)
    else:
        prediction_plot(test_y, test_prediction)


def model_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.savefig("data/plots/model_loss.png")
    plt.close()


def prediction_plot(testY, test_predict):
  len_prediction = [x for x in range(len(testY))]
  figure = plt.figure(figsize=(8, 4))
  plt.plot(len_prediction, testY[:], marker='.', label="actual")
  plt.plot(len_prediction, test_predict[:], 'r', label="prediction")
  plt.tight_layout()
  sns.despine(top=True)
  plt.subplots_adjust(left=0.07)
  plt.ylabel('Ads Daily Spend', size=15)
  plt.xlabel('Time step', size=15)
  plt.legend(fontsize=15)
  return figure


def log_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def prediction_plot_multi_timeframe(testY, test_predict):
    len_prediction = len(testY[0])
    y_actual = []
    x_actual = []
    for i in range(len(testY)):
        y_actual.append(testY[i, 0])
        x_actual.append(i)
    plt.figure(figsize=(8, 4))
    plt.plot(x_actual, y_actual, marker='.', label="actual")
    for i in range(0, len(testY), len_prediction):
        j = i + len_prediction
        x_pred = [x for x in range(i, j)]
        y_pred = test_predict[i]
        plt.plot(x_pred, y_pred, 'r', label=f"prediction_{i}")
    plt.tight_layout()
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('frac_price', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.savefig("data/plots/prediction.png")
    plt.close


def print_data_info(data):
    print(f"data is type: {type(data)}")
    if type(data) == np.array:
        print(f"data has shape: {data.shape}, type:{type(data)}")


def plot_model_inputs(data_x, data_y, window=24):
    x = []
    y = []
    for i in range(len(data_x)):
        x.append(data_x[i, 0, :])
        y.append(data_y[i])
    plt.figure(figsize=(100, 50))
    plt.plot(x, label="X")
    plt.plot(y, label="Y")
    plt.ylabel('frac_calculus', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    plt.show()
