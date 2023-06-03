import pandas as pd
import numpy as np

#load candlestick data from csv and preprocess column names


def timeframe_agg(df, interval='5min'):
    """Tranform 1 Minute OHLC to 5min, 1D, 1W dataframe"""
    df = df.resample(interval).agg({'Open': 'first', 'High': 'max',
                                    'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'VWAP': 'mean'})
    return df


def load_1min_OHLC():
    drop_columns = ["Unix_Timestamp", "DateTime", "Volume_(BTC)"]
    df = pd.read_csv("data/bitstamp_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Unix_Timestamp"], unit="s")
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"Volume_(Currency)": "Volume",
                            "Weighted_Price": "VWAP"}).drop(columns=drop_columns)
    return df


#Define dataframe with exact date range date and merge with ohlc data => Lückenlose Daten je minute
def clean_data(df):
    date_range = pd.DataFrame({"Date": pd.date_range(
        start='1/1/2015', end='31/12/2019', freq="min")})
    data = load_1min_OHLC()
    df = date_range.merge(data, on="Date", how="left")
    print(df)
    df = timeframe_agg(df.set_index("Date"))
    return df

# replace missing minute data with previous closing price and a volume of 0


def replace_missing_candles(df, save=True):
    last_idx = None
    for idx, row in df.iterrows():
        if pd.isnull(df.at[idx, "Open"]):
            last_close = df.at[last_idx, "Close"]
            df.loc[idx, "Open"] = last_close
            df.loc[idx, "Low"] = last_close
            df.loc[idx, "High"] = last_close
            df.loc[idx, "Close"] = last_close
            df.loc[idx, "Volume"] = 0.0
            df.loc[idx, "VWAP"] = last_close
        last_idx = idx
    if save:
        df.to_csv("data/BTC_2015-2019_5min_cleaned.csv")
    return df


def load_cleaned_data(filename="data/BTC_2015-2019_5min_cleaned.csv"):
    df = pd.read_csv("data/BTC_2015-2019_5min_cleaned.csv", parse_dates=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def get_aggregated_data(start=0, end=210240):
    df_5min = load_cleaned_data()[start:end]
    df_1D = timeframe_agg(df_5min, interval="D")
    df_4H = timeframe_agg(df_5min, interval="4H")
    return df_1D, df_4H, df_5min


# Unix_Timestamp         int64
# DateTime              object
# Open                 float64
# High                 float64
# Low                  float64
# Close                float64
# Volume_(BTC)         float64
# Volume_(Currency)    float64
# Weighted_Price       float64

#Length DateTimeIndex 2628001
# Volume_(Currency)    float64
# Weighted_Price       float64

#Length DateTimeIndex 2628001
