# import utility_functions as uf
# pipeline = {"input_file":"data/BTC_2015-2019_5min_cleaned.csv",
# "timeframes":["5T", "1H", "4H", "1D"],
# "steps":[uf.frac_diff, uf.minmax_scaler, uf.agg_timeframes],
# "out_path":"data/pipeline",
# }

rolling_window_start = {"1D": 0,
                        "4H": 150,
                        "1H": 696,
                        "5T": 8616,
                        }
rolling_window_firstend = {"1D": 30,
                           "4H": 180,
                           "1H": 720,
                           "5T": 8640,
                           }
window_size = {"1D": 30,
               "4H": 30,
               "1H": 24,
               "5T": 24}

step_parameter = {"5T": {"5T": 1, "1H": 12, "4H": 48, "1D": 288},
                  "1H": {"5T": 1, "1H": 1, "4H": 4, "1D": 24},
                  "4H": {"5T": 1, "1H": 1, "4H": 1, "1D": 6}}


#PARAMETER FOR BATCH PREPERATION
#steps = 0 => use all data
batch = {"key_timeframe": "5T",
         "steps": 100000,
         "prediction_frame": 1,
         "prediction_column": 4,
         "prediciton_type": "price change percent",
         "window_parameter": {"start_windows": rolling_window_start,
                              "step_parameter": step_parameter,
                              "window_sizes": window_size}
         }
