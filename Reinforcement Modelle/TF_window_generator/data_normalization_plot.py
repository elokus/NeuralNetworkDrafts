import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sklp
import numpy as np
import seaborn as sns
from finta import TA
#custom
from window_generator import WindowGenerator
from preprocesing.feature_pipeline import DataPreprocessor, FileExtractor
import preprocesing.utility_functions as uf


path = 'data/aggregated'
prefix = "BTCUSDT"
timeframes = ["1H"]
processing_steps = [uf.agg_timeframes]
start_date = "2015-01-01 00:00:00"
indicator = [TA.SMA, TA.WMA, TA.RSI, TA.ADX, TA.ATR, TA.SMA, TA.VWAP, TA.MSD]
indicator_name_arg = [("sma",10), ("wma",10), ("rsi",10), ("adx",10), ("atr",10), ("sma_100",100), ("vwap",None), ("msd", None)]
#normalize by z transformation (x-mean)/std
z_norm_cols = ["open", "high", "low", "close", "vwap", "wma", "sma_100"]
#normalize relation x/mean
rel_col = ["atr", "msd"]
#normalize by minmax scaling
minmax_col=["rsi", "adx"]
feature_types = [z_norm_cols, rel_col, minmax_col]
start_date_post_preperation = "2015-01-15 00:00:00"
start_date_train = "2015-02-01 00:00:00"
pd.set_option('display.max_columns', None)
#features = data_pipeline("data/raw/BTC_2015-2019_5min_cleaned.csv", timeframes=timeframes,
#                         steps=processing_steps, out_path=path, save_original=False, processor=True)
def label_movement(df):
    df["label"] = df["close"].pct_change()
    df.loc[(df.label > 0), 'label'] = 1
    df.loc[(df.label < 0), 'label'] = 0
    return df

def label_direction_change(df):
    return

def add_indicator(df, indicator, args):
    #preprocesses
    if len(indicator) != len(args):
        print("ERROR: Indicator and Args not maching!")
        return
    df = df.drop("vwap", axis=1)
    for index, func in enumerate(indicator):
        col_name = args[index][0]
        if args[index][1] == None:
            col_value = func(df)
        else:
            col_value = func(df,args[index][1])
        df[col_name] = col_value
    return df

def normalization(df, z_cols, rel_col, mean_col="sma", std_col="msd"):
    mean = df[mean_col]
    std = df[std_col]
    #normalize by z-transform
    for col in z_cols:
        df[col] = (df[col] - mean) / std
    #normalize Volume
    mean_vol = TA.SMA(df, column="volume")
    std_vol = TA.MSD(df, column="volume")
    df["volume"] = ( df["volume"] - mean_vol ) / std_vol
    #normalize by sma
    for col in rel_col:
        df[col] = df[col] / mean
    df = df.drop(mean_col, axis=1)
    return df

def FitNegMinMaxscaler(df):
    for col in df.columns:
        _df = df[[col]]
        signs = np.sign(_df)
        _df = np.absolute(_df)
        scaler = sklp.MinMaxScaler()
        _df = scaler.fit_transform(_df)
        df[col] = _df * signs
    return df
def ScalerFoo(scaler_func, arg):
    if arg is not None:
        scaler = scaler_func(method=arg)
    else:
        scaler = scaler_func()
    return scaler
def FitScaler(df, scaler_func, feature_types, arg=None):
    norm_scaler = ScalerFoo(scaler_func, arg)
    minmax_scaler = ScalerFoo(scaler_func, arg)
    rel_scaler = ScalerFoo(sklp.MinMaxScaler, arg)
    df[feature_types[0]] = norm_scaler.fit_transform(df[feature_types[0]])
    df[feature_types[1]] = rel_scaler.fit_transform(df[feature_types[1]])
    df[feature_types[2]] = minmax_scaler.fit_transform(df[feature_types[2]])
    return df

#data preperation
data = FileExtractor(path, prefix, timeframes, start_date)
df = data.data_dict["1H"]
df = label_movement(df)
df = add_indicator(df, indicator, indicator_name_arg)
df = df[start_date_post_preperation:]
df = normalization(df, z_norm_cols, rel_col, mean_col="sma")
df = FitScaler(df, sklp.MinMaxScaler, feature_types)
df = df[start_date_train:]

#plot normalization
def plot_violin(df):
    df_std = df.melt(var_name="Column", value_name="Normalized")
    plt.figure(figsize=(12,6))
    ax = sns.violinplot(x="Column", y="Normalized", data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.show()

def plot_multi_violin(dfs):
    fig, axs = plt.subplots(len(dfs), 1, figsize=(15, 6))
    for n in range(len(dfs)):
        df = dfs[n].melt(var_name="Column", value_name="Normalized")
        sns.violinplot(ax=axs[n], x="Column", y="Normalized", data=df)
    plt.show()

#train, val, test splitting
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
num_features = df.shape[1]

dfs = [train_df, val_df, test_df]
plot_multi_violin(dfs)

# wd = WindowGenerator(input_width=24, label_width=1, shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=["label"], conv_window=24)
# print(wd)
#%%
