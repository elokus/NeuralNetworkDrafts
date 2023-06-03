import numpy as np
import pandas as pd
import gc
import os
import random
import copy
import matplotlib.pyplot as plt
import pandas
import statsmodels
from statsmodels.tsa.stattools import adfuller
#custom


def getWeights(d, lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w = [1]
    for k in range(1, lags):
        w.append(-w[-1]*((d-k+1))/k)
    w = np.array(w).reshape(-1, 1)
    return w


def ts_differencing(series, order, cutoff):
    # return the time series resulting from (fractional) differencing
    # for real orders order up to lag_cutoff coefficients
    #cutoff 0.3 = 2277, 0.4=1460
    weights = getWeights(order, cutoff)
    res = 0
    for k in range(cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[cutoff:]


def ts_differencing_tau(series, order, tau):
    # return the time series resulting from (fractional) differencing
    lag_cutoff = (cutoff_find(order, tau, 1))  # finding lag cutoff with tau
    print(f"Lag Cutoff: {lag_cutoff}")
    weights = getWeights(order, lag_cutoff)
    res = 0
    for k in range(lag_cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[lag_cutoff:]


def cutoff_find(order, cutoff, start_lags):
    #order is our dearest d, cutoff is 1e-5 for us, and start lags is an initial amount of lags in which the loop will start, this can be set to high values in order to speed up the algo
    val = np.inf
    lags = start_lags
    while abs(val) > cutoff:
        w = getWeights(order, lags)
        val = w[len(w)-1]
        lags += 1
    print(f"Lag Cutoff: {lags}")
    return lags


# differences = [0.5, 0.9]
# fig, axs = plt.subplots(len(differences), 2, figsize=(15, 6))
# for i in range(0, len(differences)):
#     axs[i, 0].plot(ts_differencing(df['Close'], differences[i], 20))
#     axs[i, 0].set_title('Original series with d='+str(differences[i]))
#     axs[i, 1].plot(ts_differencing(df_log['Close'], differences[i], 20), 'g-')
#     axs[i, 1].set_title('Logarithmic series with d='+str(differences[i]))
#     plt.subplots_adjust(bottom=0.01)
# plt.show()
