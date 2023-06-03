import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Reshape, Input, concatenate, Flatten
from tensorflow.keras.utils import plot_model
from data_preprocessing import get_aggregated_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math


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


def data_batch(steps):
    """prepare batches for X 4 hour batches"""
    start_A = 0
    start_B = 150
    start_C = 8592
    model_A = load_model_weights(model_type="1D")
    model_B = load_model_weights(model_type="4H")
    model_C = load_model_weights(model_type="5T")

    df_1D, df_4H, df_5T = get_aggregated_data()
    A_org = df_1D.to_numpy()
    B_org = df_4H.to_numpy()
    C_org = df_5T.to_numpy()

    #Scale data with MinMaxScaler
    A_scaler = MinMaxScaler()
    B_scaler = MinMaxScaler()
    C_scaler = MinMaxScaler()

    A = A_scaler.fit_transform(A_org)
    B = B_scaler.fit_transform(B_org)
    C = C_scaler.fit_transform(C_org)

    data = []
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
        print(A_dnn)
        B_dnn = model_B(B_out).numpy()
        C_dnn = model_C(C_out).numpy()
        data.append(np.concatenate((A_dnn, B_dnn, C_dnn), axis=None))
        str = f"A:{start_A}-{end_A}, B:{start_B}-{end_B}, C:{start_C}-{end_C}"
        param.append(str)
        start_C = start_C + 48
        start_B = start_B + 1
        day_counter = step + 1
        if day_counter % 6 == 0:
            start_A = start_A + 1
    return data, param


print("Output")
data, param = data_batch(2)
print(param)
print(data)
feature = data[0]
print(feature)

# def train_test_split():

#
# # define two sets of inputs
# inputA = Input(shape=(180,))
# inputB = Input(shape=(180,))
# inputC = Input(shape=(288,))
# # the first branch operates on the first input
# x = Dense(32, activation="relu")(inputA)
# x = Dense(32, activation="relu")(x)
# x = Dense(16, activation="relu")(x)
# x = Dropout(0.25)(x)
# x = Model(inputs=inputA, outputs=x)
# # the second branch opreates on the second input
# y = Dense(32, activation="relu")(inputB)
# y = Dense(32, activation="relu")(y)
# y = Dense(16, activation="relu")(y)
# y = Dropout(0.25)(y)
# y = Model(inputs=inputB, outputs=y)
# # the third branch operates on the third inputA
# z = Dense(64, activation="relu")(inputC)
# z = Dense(32, activation="relu")(z)
# z = Dense(16, activation="relu")(z)
# z = Dropout(0.25)(z)
# z = Model(inputs=inputC, outputs=z)
#
# # combine the output of the two branches
# combined = concatenate([x.output, y.output, z.output])
#
#
# model = Model(inputs=[x.input, y.input, z.input], outputs=combined)
# #plot_model(model, "model.png", show_shapes=True)
# model.summary()
# #model.compile(optimizer='adam', loss='mse', loss_weights=[1.])
#
# d = model(data_batch())
# print(d)


# def data_1batch(steps):
#     df_1D, df_4H, df_5T = get_aggregated_data()
#     A = df_1D.to_numpy()[:30].reshape(1, 180)
#     B = df_4H.to_numpy()[:30].reshape(1, 180)
#     C = df_5T.to_numpy()[:48].reshape(1, 288)
#     return [A, B, C]
