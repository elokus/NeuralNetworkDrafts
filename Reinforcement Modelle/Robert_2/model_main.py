from feature_pipeline import data_pipeline, batch_pipeline
import utility_functions as uf
import numpy as np
from models import EncoderCNN, EncoderDNN
import parameter.param as parameter

#preprocessing data
path = 'data/pipeline4'
prefix = "BTCUSDT"
timeframes = ["5T", "1H", "4H"]
processing_steps = [uf.agg_timeframes, uf.log_diff]
first_date = "2015-01-15 00:00:00"

features = data_pipeline("data/BTC_2015-2019_5min_cleaned.csv", timeframes=timeframes,
                         steps=processing_steps, out_path=path, save_original=True, processor=True)
#batch prepeation
batch = {"key_timeframe": "1H",
         "steps": 10000,
         "prediction_frame": 5,
         "predict_from_original": True,  # when using frac_diff set False
         "prediction_column": 4,
         # uf.predict_log, uf.predict_next, uf.predict_updown,
         "prediction_function": uf.predict_updown,
         "window_parameter": {"start_windows": parameter.rolling_window_start,
                              "step_parameter": parameter.step_parameter,
                              "window_sizes": parameter.window_size}}
data = batch_pipeline(path, prefix, timeframes,
                      first_date).prepare_batch(batch)

train_x, test_x = uf.train_test_split_single(data[0], 0, 0.8)
train_y, test_y = uf.train_test_split_single(data[1], 0, 0.8)
#uf.plot_model_inputs(train_x, train_y)


#uf.analyze_inputs_outputs(train_x, test_x, train_y, test_y)

#model creation
cnn_autoencoder = EncoderCNN((24, 6), batch["prediction_frame"])
dnn_autoencoder = EncoderDNN((24, 6), batch["prediction_frame"])
cnn_autoencoder.build(input_shape=(None, 24, 6))
dnn_autoencoder.build(input_shape=(None, 24, 6))

#model evaluation
cnn_parameter = {"epochs": 100,
                 "batch_size": 50,
                 "prediction_frame": batch["prediction_frame"],
                 "name": "CNN_Autoencoder"}
dnn_parameter = {"epochs": 100,
                 "batch_size": 50,
                 "prediction_frame": batch["prediction_frame"],
                 "name": "DNN_Autoencoder"}
uf.compile_and_evaluate(cnn_autoencoder, train_x, test_x,
                        train_y, test_y, cnn_parameter)
#uf.compile_and_evaluate(dnn_autoencoder, train_x, test_x,
#                        train_y, test_y, dnn_parameter)
