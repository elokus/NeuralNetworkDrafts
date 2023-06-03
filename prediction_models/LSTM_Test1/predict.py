import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Predict_Future:

    def __init__(self, X_test, validation_df, lstm_model):
        self.X_test = X_test
        self.validation_df = validation_df
        self.lstm_model = lstm_model

    def predicted_vs_actual(self, X_min, X_max, numeric_colname):

        curr_frame = self.X_test[len(self.X_test)-1]
        future = []

        for i in range(len(self.validation_df)):
            # append the prediction to our empty future list
            future.append(self.lstm_model.predict(
                curr_frame[np.newaxis, :, :])[0, 0])
            # insert our predicted point to our current frame
            curr_frame = np.insert(curr_frame, len(
                 self.X_test[0]), future[-1], axis=0)
            # push the frame up one to make it progress into the future
            curr_frame = curr_frame[1:]

        def reverse_minmax(X, X_max=X_max, X_min=X_min):
            return X * (X_max-X_min) + X_min

        # Plot
        reverse_curr_frame = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in self.X_test[len(self.X_test)-1]],
                                           "historical_flag": 1})
        reverse_future = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in future],
                                       "historical_flag": 0})

        # Change the indicies! Only for FUTURE predictions
        # reverse_future.index += len(reverse_curr_frame)

        print("See Plot for predicted vs. actuals")
        plt.plot(reverse_curr_frame[numeric_colname])
        plt.plot(reverse_future[numeric_colname])
        plt.title("Predicted Points Vs. Actuals (Validation)")
        plt.show()

        # Check accuracy vs. actuals
        comparison_df = pd.DataFrame({"Validation": self.validation_df[numeric_colname],
                                      "Predicted": [reverse_minmax(x) for x in future]})
        print("Validation Vs. Predicted")
        print(comparison_df)

    def predict_future(self, X_min, X_max, numeric_colname, timesteps_to_predict, return_future=True):

        curr_frame = self.X_test[len(self.X_test)-1]
        future = []

        for i in range(timesteps_to_predict):
            # append the prediction to our empty future list
            future.append(self.lstm_model.predict(
                curr_frame[np.newaxis, :, :])[0, 0])
            # insert our predicted point to our current frame
            curr_frame = np.insert(curr_frame, len(
                 self.X_test[0]), future[-1], axis=0)
            # push the frame up one to make it progress into the future
            curr_frame = curr_frame[1:]

        def reverse_minmax(X, X_max=X_max, X_min=X_min):
            return X * (X_max-X_min) + X_min

        # Reverse the original frame and the future frame
        reverse_curr_frame = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in self.X_test[len(self.X_test)-1]],
                                           "historical_flag": 1})
        reverse_future = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in future],
                                       "historical_flag": 0})

        # Change the indicies to show prediction next to the actuals in orange
        reverse_future.index += len(reverse_curr_frame)

        print("See Plot for Future Predictions")
        plt.plot(reverse_curr_frame[numeric_colname])
        plt.plot(reverse_future[numeric_colname])
        plt.title("Predicted Future of " + str(timesteps_to_predict) + " days")
        plt.show()

        if return_future:
            return reverse_future
