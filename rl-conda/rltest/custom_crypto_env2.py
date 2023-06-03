import numpy as np
from TradingEnv2 import TradingEnv

#env2 has another buy and sell logic for future trading. The Agent is able to short or buy


class CryptoEnv(TradingEnv):
    """ Frame bound defines start and end of data frame start is the first trading day so day 30 or 180 on 4h chart"""

    def __init__(self, prices, features):
        super().__init__(prices, features)
        self.tax = 0.01  # unit

    def _calculate_shares(self, action):
        current_price = self.prices[self._current_tick]
        if action >= 0:
            delta_shares = (self._cash * action) / \
                            (current_price * (1 + self.tax))
            delta_cash = self._cash * action
            if action == 0:
                Action_type = "Hold"
            else:
                Action_type = "Buy"
        if action < 0:
            delta_shares = (self._shares * action)
            if self._shares - delta_shares < 0.1:
                delta_shares = 0
            delta_cash = delta_shares * current_price
            Action_type = "Sell"
        self._last_shares = self._shares
        self._shares += delta_shares
        self._cash -= delta_cash
        share_value = self._shares * current_price
        total_value = share_value + self._cash
        pos_info = {"Action": Action_type, "Shares": delta_shares,
                    "Price": current_price, "Portfolio Value": total_value}
        return total_value, Action_type, pos_info

    def _calculate_reward(self):
        last_shares = self._last_shares
        current_shares = self._shares
        last_price = self.prices[self._current_tick - 1]
        current_price = self.prices[self._current_tick]
        step_reward = (current_shares - last_shares) * \
            ((current_price - last_price) / last_price)
        return step_reward

        # def _process_data(self):
        #     prices = self.df.loc[:, 'Close'].to_numpy()
        #
        #     # validate index (TODO: Improve validation)
        #
        #     prices = prices[self.frame_bound[0]
        #                     - self._start_tick:self.frame_bound[1]]
        #     signal_features = self.features
        #     return prices, signal_features
