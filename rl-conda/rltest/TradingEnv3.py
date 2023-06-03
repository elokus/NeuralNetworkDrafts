import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import random

#static variables
max_account_balance = 100000


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, prices, features, max_steps):
        self.seed()
        self.prices = prices
        self.signal_features = features
        self.shape = (1, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(
            low=np.inf, high=np.inf, shape=(1, 51), dtype=np.float32)
        self.reward_range = (-10000000, +1000000)

        # static var
        self.max_account_balance = 10000.
        self.initial_balance = 10000.
        self.maker_fee = 0.0002
        self.taker_fee = 0.0007
        self.max_steps = max_steps

        # episode
        self.episode = 1
        self._start_tick = 0
        self._end_tick = len(self.prices) - 1
        self._cash = 10000.
        self._done = None
        self._current_tick = None
        self._current_price = None
        self._shares = 0
        self._max_shares_held = None
        self._share_value = 0
        self._last_shares = None  # depreciate
        self._total_fees = None
        self._total_value = None
        self._last_value = None
        self._max_value = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._first_rendering = None
        self.history = []
        self.graph_reward = []
        self.graph_profit = []
        self._episode_render = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        #Set the current step to a random point within the data frame
        #Weights of the current step follow the square function
        if self.max_steps > 0:
            start = list(
                range(self._start_tick, self._end_tick-self.max_steps))
            weights = [i for i in start]
            self._current_tick = random.choices(start, weights)[0]
            self._start_tick = self._current_tick

        self._done = False
        self._current_tick = self._start_tick
        self._current_price = self.prices[self._current_tick]
        self._cash = self.initial_balance
        self.episode_reward = 0
        self._share_value = 0
        self._total_value = self._cash + self._share_value
        self._last_value = self._total_value
        self._total_fees = 0
        self._max_value = self._total_value
        self._max_shares_held = self.initial_balance / self._current_price
        self._position_history = (self._start_tick * [None]) + [None]
        self._total_reward = 0
        self._first_rendering = True
        self.history = []

        return self._get_observation()

# ToDo implemenet different logic
#   1: 50% of capital used to buy asset
#   2: sell order without assets used for short_ticks

    def _get_observation(self):
        dnn_features = np.array(self.signal_features[self._current_tick])
        portfolio_state = np.array([self._cash / self._max_value,
                                    self._total_value / self._max_value, self._shares / self._max_shares_held])
        current_state = np.append(
            dnn_features, portfolio_state).reshape(1, 51)
        return current_state

    def _take_action(self, action):
        _action = action[0]
        current_price = self.prices[self._current_tick]
        #buy
        if _action > 0:
            new_shares = self._cash * _action / current_price
            fees = new_shares * current_price * self.maker_fee
            self._total_fees += fees
            self._cash -= new_shares * current_price + fees
            self._shares += new_shares
            position = "Buy"
        if _action < 0:
            sold_shares = -self._shares * _action
            fees = sold_shares * current_price * self.taker_fee
            self._total_fees += fees
            self._cash += sold_shares * current_price - fees
            self._shares -= sold_shares
            position = "Sell"
        self._share_value = self._shares * current_price
        self._last_value = self._total_value
        self._total_value = self._cash + self._share_value
        if self._total_value > self._max_value:
            self._max_value = self._total_value
        return position

    def _calculate_reward(self):
        profit = ((self._total_value - self._last_value)
                  / (self._last_value))*100
        delta_price = (self.prices[self._current_tick]
                       / self.prices[self._current_tick - 1] - 1)*100
        diff = profit - delta_price
        reward = np.sign(diff) * (diff)**2
        return reward

    def step(self, action):
        self._done = False
        position = self._take_action(action)
        self._position_history.append(position)

        self._current_tick += 1
        step_reward = self._calculate_reward()
        self._total_reward += step_reward

        #End after max steps:

        observation = self._get_observation()
        if self.max_steps > 0:
            if self._current_tick >= self.max_steps + self._start_tick:
                end = True
            else:
                end = False
            if end:
                self._done = True
                self.episode_reward = step_reward
                self.graph_profit = self._total_value
                self._render_episode()
                self.episode += 1
        elif self._current_tick == self._end_tick:
            self._done = True
        info = {"Total Value": self._total_value, "Share Value": self._share_value,
                "Cash Balance": self._cash, "Total Profit": self._total_reward}

        return observation, step_reward, self._done, info

    def _update_history(self, info):
        self.history.append(info)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == "Sell":
                color = 'red'
            elif position == "Buy":
                color = 'green'
            else:
                color = 'yellow'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ '
            + "Total Value: %.6f" % self._total_value
        )

        plt.pause(0.01)

    def _render_episode(self, mode="human"):
        self._episode_render.append(
            [self.episode, self.episode_reward, self.graph_profit])

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        # hold_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == "Sell":
                short_ticks.append(tick)
            elif self._position_history[i] == "Buy":
                long_ticks.append(tick)
            # elif self._position_history[i] == "Hold":
            #     hold_ticks.append(tick)
        print(self._episode_render)
        print(self.history)
        print("TICKS Short, Long:")
        print(short_ticks)
        print(long_ticks)
        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')
        # plt.plot(hold_ticks, self.prices[hold_ticks], "yo")

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ '
            + "Portfolio Value: %.6f" % self._total_value
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_shares(self):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
