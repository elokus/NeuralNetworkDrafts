#test trend detection in dataframe
from scipy.signal import argrelmin, argrelmax
import dataprep_class as ds
import trendet
import investpy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def trenddetect():
    sns.set(style='darkgrid')

    test = ds.get_dataframe()

    res = trendet.identify_df_trends(df=test, column='Close', window_size=10)

    res.reset_index(inplace=True)

    with plt.style.context('dark_background'):
        plt.figure(figsize=(20, 10))

        ax = sns.lineplot(x=res['Date'], y=res['Close'])

        labels = res['Up Trend'].dropna().unique().tolist()

        for label in labels:
            sns.lineplot(x=res[res['Up Trend'] == label]['Date'],
                         y=res[res['Up Trend'] == label]['Close'],
                         color='green')

            ax.axvspan(res[res['Up Trend'] == label]['Date'].iloc[0],
                       res[res['Up Trend'] == label]['Date'].iloc[-1],
                       alpha=0.2,
                       color='green')

        labels = res['Down Trend'].dropna().unique().tolist()

        for label in labels:
            sns.lineplot(x=res[res['Down Trend'] == label]['Date'],
                         y=res[res['Down Trend'] == label]['Close'],
                         color='red')

            ax.axvspan(res[res['Down Trend'] == label]['Date'].iloc[0],
                       res[res['Down Trend'] == label]['Date'].iloc[-1],
                       alpha=0.2,
                       color='red')

        plt.show()


N = 4  # number of iterations
df = ds.get_dataframe()
h = df['High'].dropna().copy()  # make a series of Highs
l = df['Low'].dropna().copy()  # make a series of Lows
for i in range(N):
    h = h.iloc[argrelmax(h.values)[0]]  # locate maxima in Highs
    l = l.iloc[argrelmin(l.values)[0]]  # locate minima in Lows
    h = h[~h.index.isin(l.index)]  # drop index that appear in both
    l = l[~l.index.isin(h.index)]  # drop index that appear in both

hl = pd.concat([h, l])
hl = hl.sort_index()
print(hl)
hli = pd.DataFrame.from_dict({"Date": hl.index, "hl": hl.values})
print(hli)


df_merge = df.merge(hli, on="Date", how="left")
extrema = []
for element in df_merge["hl"]:
    if pd.isnull(element):
        extrema.append("trend")
    else:
        extrema.append("data")
df_merge["extrema"] = extrema
print(df_merge)
df.reset_index(inplace=True)
sns.set_theme()
g = sns.relplot(x="Date", y="Close", hue="extrema", kind="line", data=df_merge)
plt.show()
