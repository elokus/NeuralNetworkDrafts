# data preperation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_series(df, xcol, datecol):
    # Create a dataframe with the features and the date time as the index
    features_considered = [xcol]
    features = df[features_considered].copy()
    features.index = df[datecol]
    features.index.name = "date"
    features.rename(columns={xcol: "feature"}, inplace=True)
    features.head()
    features.plot(subplots=True)
    return features


def stationarity_test(X, log_x="Y", return_p=False, print_res=True):
    # If X isn't logged, we need to log it for better results
    if log_x == "Y":
        X = np.log(X[X > 0])

    # Once we have the series as needed we can do the ADF test
    from statsmodels.tsa.stattools import adfuller
    dickey_fuller = adfuller(X)

    if print_res:
        # If ADF statistic is < our 1% critical value (sig level) we can conclude it's not a fluke (ie low P val / reject H(0))
        print('ADF Stat is: {}.'.format(dickey_fuller[0]))
        # A lower p val means we can reject the H(0) that our data is NOT stationary
        print('P Val is: {}.'.format(dickey_fuller[1]))
        print('Critical Values (Significance Levels): ')
        for key, val in dickey_fuller[4].items():
            print(key, ":", round(val, 3))

    if return_p:
        return dickey_fuller[1]


def outlier_removal(dat):
    var = "feature"
    IQR = dat[var].describe()['75%'] - dat[var].describe()['25%']
    min_val = dat[var].describe()['25%'] - (IQR * 1.5)
    max_val = dat[var].describe()['75%'] + (IQR * 1.5)
    dat = dat[(dat[var] > min_val) & (dat[var] < max_val)]
    return dat


def data_preperation(input_data, feature_col, date_col, testsize, remove_outlier=True):
    data = create_series(input_data, feature_col, date_col)
    #train and test data split
    trainsize = len(data) - testsize
    test_df = data.iloc[trainsize:]
    train_df = data.iloc[1:trainsize, ]
    #stationarity_test
    print("Summary Statistics - ADF Test For Stationarity\n")
    if stationarity_test(X=train_df["feature"], return_p=True, print_res=False) > 0.05:
        print("P Value is high. Consider Differencing: " + str(stationarity_test(
            X=train_df["feature"], return_p=True, print_res=False)))
    else:
        stationarity_test(X=train_df["feature"])

    #remove_outlier
    if remove_outlier:
        train_df = outlier_removal(train_df)
    train_df = train_df.sort_index(ascending=True)
    train_df = train_df.reset_index()
    return train_df, test_df


class Series_Prep:

    def __init__(self, rnn_df):
        self.rnn_df = rnn_df
        self.numeric_colname = "feature"

    def make_window(self, sequence_length, train_test_split=0.9, return_original_x=True):

        # Create the initial results df with a look_back of 60 days
        result = []

        # 3D Array
        for index in range(len(self.rnn_df) - sequence_length):
            result.append(self.rnn_df[self.numeric_colname]
                          [index: index + sequence_length])

        # Getting the initial train_test split for our min/max val scalar
        row = int(round(train_test_split * np.array(result).shape[0]))
        train = np.array(result)[:row, :]
        X_train = train[:, :-1]
        # Manual MinMax Scaler
        X_min = X_train.min()
        X_max = X_train.max()
        X_min_orig = X_train.min()
        X_max_orig = X_train.max()

        # Minmax scaler and a reverse method
        def minmax(X):
            return (X-X_min) / (X_max - X_min)

        def reverse_minmax(X):
            return X * (X_max-X_min) + X_min

        def minmax_windows(window_data):
            normalised_data = []
            for window in window_data:
                window.index = range(sequence_length)
                normalised_window = [((minmax(p))) for p in window]
                normalised_data.append(normalised_window)
            return normalised_data

        # minmax the windows
        result = minmax_windows(result)
        # Convert to 2D array
        result = np.array(result)
        if return_original_x:
            return result, X_min_orig, X_max_orig
        else:
            return result

    @staticmethod
    def reshape_window(window, train_test_split=0.8):
        # Train/test for real this time
        row = round(train_test_split * window.shape[0])
        train = window[:row, :]

        # Get the sets
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = window[row:, :-1]
        y_test = window[row:, -1]

        # Reshape for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
        return X_train, X_test, y_train, y_test


# #Execution
# input_data = pd.read_csv("./data/ETH_2016-2020.csv",
#                          parse_dates=["trade_date"])
#
# feature_col = "USD_price_change_1_day"
# date_col = "trade_date"
# testsize = 100
# seq_len = 60
# split = 0.8
#
# train_df, validation_df = data_preperation(
#     input_data, feature_col, date_col, testsize, True)
# series_prep = Series_Prep(train_df)
#
# window, X_min, X_max = series_prep.make_window(seq_len, split)
# print(f'X min={X_min} and max={X_max}')
#
#
# X_train, X_test, y_train, y_test = series_prep.reshape_window(
#     window, train_test_split=split)


#X_train, X_test, y_train, y_test =
#X_train, X_test, y_train, y_test =
