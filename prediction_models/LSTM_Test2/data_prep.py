#data_preprocessing
import pandas as pd
import numpy as np

df = pd.read_csv("BTC_input.csv", parse_dates=["trade_date"])
nupl = pd.read_json(
    "net-unrealized-profit-loss-nupl-btc-24h.json", orient='records')

nupl["t"] = pd.to_datetime(nupl["t"], infer_datetime_format=True)


df2 = pd.merge(df, nupl, left_on="trade_date", right_on='t')
df2 = df2.drop(["BTC_price_change_1_day", "t", "days", "Day",
                "months", "weeks", "years", "Month", "Week", "Year", "DIndex", "GL_USDx100"], axis=1)
df = df2.rename(columns={"trade_date": "date",
                         "USD_price_change_1_day": "dprice", "v": "nupl"})
print(df)
df.to_csv("data.csv")
