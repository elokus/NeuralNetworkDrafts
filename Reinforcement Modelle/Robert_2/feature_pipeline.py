import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Reshape, Input, concatenate, Flatten
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
#custom
import utility_functions as uf
import parameter.param as param
import os


class data_pipeline():
    #------classmethod description------#
    #Initializing: load data from main file (optional: save original dataframes)
    #processor will process functions provide via steps [function_1, function_2, function_3,...]
    #processing functions are provided in pipeline_functions.py (e.g.: frac_diff, minmax_scaler, log_diff, agg_timeframes)
    #when agg_timeframes is executed via processor the output will contain multiple dataframes in self.dataframes
    #after that each remaining processor step will be applied to each dataframe in dataframes

    def __init__(self, input_file, timeframes, steps, out_path="data/pipeline", save_original=True, processor=True, save_and_plot=True):
        self.input_file = input_file
        self.dataframe = self.load_data()
        self.timeframes = timeframes
        self.dataframes = None
        self.steps = steps
        self.original_df = None
        self.original_dfs = None
        self.output_dir = out_path
        #save input dataframe
        self.save_original = save_original
        if self.save_original:
            self.original_df = self.dataframe.copy(deep=True)
        if processor:
            self.processor()
        if save_and_plot:
            self.save_all_files()

    def load_data(self):
        df = pd.read_csv(self.input_file, parse_dates=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df

    def processor(self):
        for foo in self.steps:
            if self.dataframes is None:
                output = foo(self.dataframe, self.timeframes)
                if type(output) == list:
                    self.dataframes = output
                    if self.save_original:
                        df_copys = []
                        for i in range(len(self.dataframes)):
                            df = self.dataframes[i].copy(deep=True)
                            df_copys.append(df)
                        self.original_dfs = df_copys
                else:
                    self.dataframe = output
            else:
                for i in range(len(self.dataframes)):
                    self.dataframes[i] = foo(
                        self.dataframes[i], self.timeframes)
        print("all feature transformer processed")

    def plot_data(self, column="Close"):
        fig, axs = plt.subplots(len(self.dataframes), 2, figsize=(15, 6))
        for i in range(0, len(self.dataframes)):
            axs[i, 0].plot(self.original_dfs[i][column])
            axs[i, 0].set_title(
                f'Original series with timeframe {self.timeframes[i]}')
            axs[i, 1].plot(self.dataframes[i][column], 'g-')
            axs[i, 1].set_title(
                f'Fractional differenciated series with d= 0.3  {self.timeframes[i]}')
            plt.subplots_adjust(bottom=0.01)
        plt.show()

    def save_all_files(self, prefix="BTCUSDT", save_only_originals=False):
        count = 0
        folder = self.output_dir
        if not os.path.exists(folder):
            # Create a new directory because it does not exist
            os.makedirs(folder)
            print("The new directory is created!")
        for i in range(len(self.original_dfs)):
            filename = f"{folder}/{prefix}_{self.timeframes[i]}.csv"
            self.original_dfs[i].to_csv(filename)
            count += 1
        if save_only_originals:
            print(f"{count} Files saved to data")
            return
        else:
            for i in range(len(self.dataframes)):
                filename = f"{folder}/_{prefix}_{self.timeframes[i]}.csv"
                self.dataframes[i].to_csv(filename)
                count += 1
        print(f"{count} Files saved to data")

    def save_and_plot(self):
        self.save_all_files()
        self.plot_data()

# features = data_pipeline("data/BTC_2015-2019_5min_cleaned.csv", timeframes=["5T", "1H", "4H", "1D"], steps=[
#                        uf.agg_timeframes], out_path="data/pipeline", save_original=True, processor=True)
# features.save_and_plot()
# features = data_pipeline("data/BTC_2015-2019_5min_cleaned.csv", timeframes=["5T", "1H", "4H", "1D"], steps=[
#                          uf.frac_diff, uf.minmax_scaler, uf.agg_timeframes], save_data=True, processor=True)
# features.save_and_plot()


#############################################################################
#######
#######------MODEL PIPELINE loading from files after FEATURE PIPELINE --------
#######
#######
##############################################################################

class batch_pipeline():
    #------classmethod description------#
    #Initializing: load data from files after saving in data_pipeline
    #transformed dataframes are named "_" before trading pair original without
    #TODO: load all files in folder and find timeframes vs. load files via filename

    def __init__(self, input_folder, prefix, timeframes, start_date):
        self.path = input_folder
        self.prefix = prefix
        self.timeframes = timeframes
        self.start_date = start_date
        self.dataframes = []
        self.original_dfs = []
        self.df_dict = {}
        self.orig_dict = {}
        self.load_files_from_filename()

    def load_files_from_filename(self):
        for t in self.timeframes:
            str_org = f"{self.path}/{self.prefix}_{t}.csv"
            str_tra = f"{self.path}/_{self.prefix}_{t}.csv"
            try:
                df_org = pd.read_csv(str_org, parse_dates=True)
            except:
                print(f"File not found check {str_org}")
            try:
                df_tra = pd.read_csv(str_tra, parse_dates=True)
            except:
                print(f"File not found check {str_tra}")
            df_org["Date"] = pd.to_datetime(df_org["Date"])
            df_org = df_org.set_index("Date").sort_index()[self.start_date:]
            df_tra = pd.read_csv(str_tra, parse_dates=True)
            df_tra["Date"] = pd.to_datetime(df_tra["Date"])
            df_tra = df_tra.set_index("Date").sort_index()[self.start_date:]
            self.dataframes.append(df_tra)
            self.df_dict[t] = df_tra
            self.original_dfs.append(df_org)
            self.orig_dict[t] = df_org

    def plot_data(self, column="Close"):
        fig, axs = plt.subplots(len(self.dataframes), 2, figsize=(15, 6))
        for i in range(0, len(self.dataframes)):
            axs[i, 0].plot(self.original_dfs[i][column])
            axs[i, 0].set_title(
                f'Original series with timeframe {self.timeframes[i]}')
            axs[i, 1].plot(self.dataframes[i][column], 'g-')
            axs[i, 1].set_title(
                f'Fractional differenciated series with d= 0.3 {self.timeframes[i]}')
            plt.subplots_adjust(bottom=0.01)
        plt.show()

    def prepare_batch(self, batch_parameter):
        data_windows, pred_data, price_data = uf.data_batch(
            self.df_dict, self.orig_dict, batch_parameter)
        pred_data = uf.prediction_generator(pred_data, batch_parameter)
        return [data_windows, pred_data, price_data]

#
# path = 'data/pipeline2'
# prefix = "BTCUSDT"
# timeframes = ["5T", "1H", "4H", "1D"]
# first_date = "2015-01-15 00:00:00"
# steps = 100
# data = model_pipeline(path, prefix, timeframes,
#                       first_date).prepare_batch(param.batch_parameter)
# print(len(data[0]))
# print(data[1])


#
# features.save_and_plot()
