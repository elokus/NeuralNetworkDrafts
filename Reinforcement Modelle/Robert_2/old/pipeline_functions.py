import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
def data_batch(data_dict, orig_dict, key_timeframe="5T", steps=0):
    #---description---#
    #prepare data input for Neurol Network processing
    #multi input timeframes are aggregatet according to parameter.param file
        # parameter:
        # window_size for each timeframe
        # key_timeframe
        # starting point is declared in parameter file so that each timeframe window fit for the first value
            # e.g. 30 days on daily frame window size means that we have to skip the first 30 days on 5 Min frame
        # step_parameter: e.g. key timeframe == 5min (aka 5T):  window  per step "5T" +1, each 12 steps "1H" +1, each 48 steps "4H" +1 and each 288 steps "1D" +1

    start_windows = param.rolling_window_start
    step_parameter = param.step_parameter[key_timeframe]
    window_sizes = param.window_size
    price_data = orig_dict[key_timeframe].reset_index().drop(
        ["VWAP", "Volume"], axis=1).to_numpy()
    window_data = []
    step_price = []
    array_dict = {key: value.to_numpy() for key, value in data_dict.items()}
    if steps == 0:
        steps = len(data_dict[key_timeframe]) - start_windows[key_timeframe]
    for step in range(steps):
        step_counter = step + 1
        step_data = []
        for key, array in array_dict.items():
            start = start_windows[key]
            end = start_windows[key]+window_sizes[key]
            #print(f"{step_counter}: {key} starts: {start} ends {end}")
            data = array[start:end]
            step_data.append(data)
            if key == key_timeframe:
                step_price.append(price_data[end-1, :])
            if step_counter % step_parameter[key] == 0:
                start_windows[key] += 1
        window_data.append(step_data)
    return window_data, step_price


def prediction_batch(step_price, prediction_frame=1, prediction_column=4, delta_percent=True):
    """prepare prediction data after data_batch, prediction_frame = 1 means predict next candle..."""
    #parameter: predict price_change in %
    prediction = []
    steps = len(step_price)-prediction_frame
    if delta_percent:
        for i in range(steps):
            last_price = step_price[i][4]
            new_price = step_price[i+1][4]
            delta = (new_price - last_price)/last_price
            prediction.append(delta)
    else:
        print("not defined")
    if prediction_frame > 1:
        values = prediction
        prediciton = []
        for i in range(steps):
            prediction_window = values[i:i+prediction_frame]
            prediction.append(prediction_window)
    return prediction
