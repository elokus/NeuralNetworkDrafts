import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import utility_functions as uf
import matplotlib.pyplot as plt
#custom
import feature_pipeline as pipeline
import parameter.param as parameter













path = 'data/pipeline2'
prefix = "BTCUSDT"
timeframes = ["5T", "1H", "4H", "1D"]
first_date = "2015-01-15 00:00:00"
#
data = pipeline.model_data(path, prefix, timeframes,
                           first_date).prepare_batch(parameter.batch)
#Output_data [input_frame[5T, 1H, 4H, 1D], prediction_data[price_change%], price_data[OHLC]]


train_x, test_x = train_test_split_mlp(data[0])
train_y, test_y = train_test_split_mlp(data[1])


model = model_generator_4_inputs(train_x)
model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['mse', 'mae'])
model.summary()
uf.analyze_inputs_outputs(train_x, test_x, train_y, test_y)
history = model.fit(train_x, train_y, epochs=20, verbose=1,
                    validation_data=(test_x, test_y), batch_size=50)
#

#
train_score = model.evaluate(train_x, train_y, verbose=0)
print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '
      % (np.sqrt(train_score[1]), train_score[2]))
test_score = model.evaluate(test_x, test_y, verbose=0)
print('Test Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
      % (np.sqrt(test_score[1]), test_score[2]))
#
model_loss(history)
test_prediction = model.predict(test_x)
print(test_prediction)
prediction_plot(test_y, test_prediction)
