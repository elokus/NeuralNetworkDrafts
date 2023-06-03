import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, prices, features):
        self.seed()
        self.prices = prices
        self.signal_features = features
        self.shape = (1, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Box(low=-0.9, high=1, shape=(1,))
        self.observation_space = spaces.Box(
            low=-0, high=1, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = 0
        self._end_tick = len(self.prices) - 1
        self._cash = 10000
        self._done = None
        self._current_tick = None
        self._current_price = None
        self._shares = 0
        self._last_shares = None
        self._total_value = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._first_rendering = None
        self.history = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._current_price = self.prices[self._current_tick]
        self._total_value, pos, info = self._calculate_shares(0.5)
        self._position_history = (self._start_tick * [None]) + [pos]
        self._total_reward = 0.

        self._first_rendering = True
        self.history = []
        return self._get_observation()

# ToDo implemenet different logic
#   1: 50% of capital used to buy asset
#   2: sell order without assets used for short_ticks

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self._total_value, pos, info = self._calculate_shares(action)
        step_reward = self._calculate_reward()
        self._total_reward += step_reward
        self._position_history.append(pos)
        observation = self._get_observation()
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[self._current_tick]

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

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
