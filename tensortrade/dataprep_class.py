import mplfinance as mpf
from finta import TA
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import moontest as moon

from urllib.request import Request, urlopen
import re

url = "https://www.cryptodatadownload.com/data/binance"


class Download_Prep:
    def __init__(self, url):
        self.url = url

        self.d = {}
        self.h = {}
        self.m = {}
        self.parse_links()

    def parse_links(self):
        html_page = urlopen(Request(self.url))
        soup = BeautifulSoup(html_page, 'lxml')
        links = []
        for link in soup.findAll('a'):
            links.append(link.get('href'))

        daily = {}
        hourly = {}
        minute = {}
        base_url = "https://www.cryptodatadownload.com"

        for link in links:
            if link.endswith('.csv'):
                parts = link.split('_')
                dlink = base_url+link
                symbol = parts[1]
                tframe = parts[2].split(".")[0]
                if tframe == "d":
                    daily[symbol] = dlink
                if tframe == "1h":
                    hourly[symbol] = dlink
                if tframe == "minute":
                    minute[symbol] = dlink
        self.d = daily
        self.h = hourly
        self.m = minute


#
#
class Import_Data:

    def __init__(self, link_dict, symbol, daily=True, hourly=True, minutes=True):

        if daily:
            self.d = pd.read_csv(link_dict.d[symbol], skiprows=1)
        else:
            self.d = None
        if hourly:
            self.h = pd.read_csv(link_dict.h[symbol], skiprows=1)
        else:
            self.h = None
        if minutes:
            self.m = pd.read_csv(link_dict.m[symbol], skiprows=1)
        else:
            self.m = None

    def clean(self):

        if self.d is not None:
            Date_d = []
            for i in self.d["unix"]:
                len_diff = 12-len(str(i))
                unix = i * (10 ** len_diff)
                Date_d.append(int(unix))
            self.d["Date"] = pd.to_datetime(Date_d, unit="s")
            df = pd.DataFrame.from_dict({"Date": self.d["Date"], "Open": self.d["open"],
                                         "High": self.d["high"], "Low": self.d["low"], "Close": self.d["close"], "Volume": self.d["Volume USDT"]})

            df.set_index("Date", inplace=True)
            self.d = df.sort_index()

        if self.h is not None:
            Date_h = []
            for i in self.h["unix"]:
                len_diff = 12-len(str(i))
                unix = i * (10 ** len_diff)
                Date_h.append(int(unix))
            self.h["Date"] = pd.to_datetime(Date_h, unit="s")
            df = pd.DataFrame.from_dict({"Date": self.h["Date"], "Open": self.h["open"],
                                         "High": self.h["high"], "Low": self.h["low"], "Close": self.h["close"], "Volume": self.h["Volume USDT"]})
            df.set_index("Date", inplace=True)
            self.h = df.sort_index()

        if self.m is not None:
            self.m["Date"] = pd.to_datetime(self.m["unix"], unit="ms")
            df = pd.DataFrame.from_dict({"Date": self.m["Date"], "Open": self.m["open"],
                                         "High": self.m["high"], "Low": self.m["low"], "Close": self.m["close"], "Volume": self.m["Volume USDT"]})
            df.set_index("Date", inplace=True)
            self.m = df.sort_index()


def add_mooncycle(df):
    """For DataFrames with date index. Add Mooncycle in Column Lunation as float 1, 0 =full, 0.5 = new """
    lunations = []
    for date in df.index:
        lunation = moon.lunar_cycle_from_date(date)
        lunations.append(lunation)
    df["Lunation"] = lunations
    return df


def get_dataframe():
    linklist = Download_Prep(url)
    BTCUSDT = Import_Data(linklist, "BTCUSDT", daily=True,
                          hourly=True, minutes=False)

    BTCUSDT.clean()
    df = BTCUSDT.d
    # print(df)
    # df = add_mooncycle(df)
    return df


# df["VWAP"] = TA.VWAP(df)
# vwap = mpf.make_addplot(df['VWAP'], color='g')
# mpf.plot(df, type='candlestick', addplot=vwap)
# mpf.show()

#

# class Data_Prep:
#
#     def __init__(self, dataset_daily, dataset_hourly, dataset_minute):
#         self.df = dataset
#
#     def clean():
#         #datecolumn
#
#
#         #drop columns: symbol, volume crypto,
#
#         #rename columns
#
#     def agg_timeframe():
#p
#     def add_mooncycle():
#
#     def filter_seq():
#
#     def add_indicator():
