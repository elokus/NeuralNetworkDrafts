import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Reshape, Input, concatenate, Flatten
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math


def timeframe_agg(df, interval='5min'):
    """Tranform 1 Minute OHLC to 5min, 1D, 1W dataframe"""
    df = df.resample(interval).agg({'Open': 'first', 'High': 'max',
                                    'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'VWAP': 'mean'})
    return df


def load_cleaned_data(filename="data/BTC_2015-2019_5min_cleaned.csv"):
    df = pd.read_csv("data/BTC_2015-2019_5min_cleaned.csv", parse_dates=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def get_aggregated_data(start=0, end=3000):
    df_5min = load_cleaned_data()[start:end]
    df_1D = timeframe_agg(df_5min, interval="D")
    df_4H = timeframe_agg(df_5min, interval="4H")
    return df_1D, df_4H, df_5min


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
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='relu'))
    return model


def load_model_weights(model_type, ):
    if model_type == "1D":
        model = create_model_dnn_32()
        _ = model(tf.ones((1, 180)))
        weights_file = "trained_models/1D_weights.h5"
    elif model_type == "4H":
        model = create_model_dnn_32()
        _ = model(tf.ones((1, 180)))
        weights_file = "trained_models/4H_weights.h5"
    elif model_type == "5T":
        model = create_model_dnn_64()
        _ = model(tf.ones((1, 288)))
        weights_file = "trained_models/5T_weights.h5"
    else:
        return "invalid model type"
    model.load_weights(weights_file)
    model.pop()
    # model.summary()
    return model


#


def data_batch(price_column=3, start=0, end=210240):
    """prepare batches for X 4 hour batches end= 105120 x years"""
    start_A = 0
    start_B = 150
    start_C = 8592
    model_A = load_model_weights(model_type="1D")
    model_B = load_model_weights(model_type="4H")
    model_C = load_model_weights(model_type="5T")

    df_1D, df_4H, df_5T = get_aggregated_data(start=start, end=end+48)
    A_org = df_1D.to_numpy()
    B_org = df_4H.to_numpy()
    C_org = df_5T.to_numpy()
    steps = len(B_org) - 180
    print(steps)
    print(
        f"A_shape: {A_org.shape}, B_shape: {B_org.shape}, C_shape{C_org.shape}")

    #Scale data with MinMaxScaler
    A_scaler = MinMaxScaler()
    B_scaler = MinMaxScaler()
    C_scaler = MinMaxScaler()

    A = A_scaler.fit_transform(A_org)
    B = B_scaler.fit_transform(B_org)
    C = C_scaler.fit_transform(C_org)

    data = []
    price_data = []
    param = []
    for step in range(steps):
        end_A = start_A+30
        A_out = A[start_A:end_A].reshape(1, 180)
        #for step in range(6):
        end_B = start_B+30
        end_C = start_C+48
        B_out = B[start_B:end_B].reshape(1, 180)
        C_out = C[start_C:end_C].reshape(1, 288)
        A_dnn = model_A(A_out).numpy()
        B_dnn = model_B(B_out).numpy()
        C_dnn = model_C(C_out).numpy()
        price_data.append(B_org[end_B-1, price_column])
        data.append(np.concatenate((A_dnn, B_dnn, C_dnn), axis=None))
        str = f"A:{start_A}-{end_A}, B:{start_B}-{end_B}, C:{start_C}-{end_C}"
        param.append(str)
        start_C = start_C + 48
        start_B = start_B + 1
        day_counter = step + 1
        if day_counter % 6 == 0:
            start_A = start_A + 1
    return np.array(data), np.array(price_data),  param
