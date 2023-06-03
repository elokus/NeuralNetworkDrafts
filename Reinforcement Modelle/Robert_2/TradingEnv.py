import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import random


class TradingEnv(gym.Env):

    metadata = {'render.mides': ['human']}

    def __init__(self, prices, features, episode_steps):
        self.seed()
        self.prices = prices
        self.features = features
        #spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(
            low=np.inf, high=np.inf, shape=(1, 52), dtype=np.float32)
        #static
        self.initial_balance = 10000
        self.tax = 0.0025
        self.maker_fee = 0.0002
        self.taker_fee = 0.0007
        self.episode_steps = episode_steps
        self.episode_mode = False
        if self.episode_steps > 0:
            self.episode_mode = True
        self.start_tick = 1
        self.end_tick = len(self.prices) - 1 - self.episode_steps
        #episode
        ### When episode_mode = False Agent should run through all data from start_tick till _end_tick
        ### When episode_mode = True Agent will begin at random point for episode_steps and than reset_episode
        self._start_tick = self.start_tick
        self._end_tick = self.end_tick
        self._episode = 1
        #vars
        self._done = False
        self._current_tick = None
        self._current_price = None
        self._last_price = None
        self._balance = None
        self._shares = None
        self._last_shares = None
        self._share_value = None
        self._last_share_value = None
        self._total_value = None
        self._last_value = None
        self._total_reward = None
        #render...
        self._position_history = (self._start_tick * [None]) + [None]
        self._episode_reward = []
        self._episode_value = []

    def reset(self):
        if self.episode_mode:
            self.reset_episode()
        self._done = False
        self._current_tick = self._start_tick
        self._current_price = self.prices[self._current_tick]
        self._last_price = self.prices[self._current_tick - 1]
        self._balance = self.initial_balance
        self._shares = 0
        self._last_shares = 0
        self._share_value = 0
        self._last_share_value = 0
        self._total_value = self._balance
        self._last_value = self._total_value
        self._total_reward = 0
        return self._get_observation()

    def _take_action(self, action):
        _action = action[0]
        self._last_shares = self._shares
        self._last_share_value = self._last_shares * self._current_price
        if _action > 0:
            new_shares = self._balance * _action / self._current_price
            fees = new_shares * self._current_price * self.maker_fee
            self._balance -= new_shares * self._current_price + fees
            self._shares += new_shares
            position = "Buy"
        if _action < 0:
            sold_shares = -self._shares * _action
            fees = sold_shares * self._current_price * self.taker_fee
            self._balance += sold_shares * self._current_price * self.taker_fee
            self._shares -= sold_shares
            position = "Sell"
        self._share_value = self._shares * self._current_price
        self._total_value = self._balance + self._share_value
        return position

    def step(self, action):
        position = self._take_action(action)
        self._position_history.append(position)
        self._current_tick += 1
        self._current_price = self.prices[self._current_tick]
        self._last_price = self.prices[self._current_tick - 1]
        self._last_value = self._total_value
        self._share_value = self._shares * self._current_price
        self._total_value = self._balance + self._share_value
        #Calculate Reward
        dvalue = (self._total_value - self._last_value)/self._last_value
        if self._last_share_value == 0:
            dshare_value = 0
        else:
            dshare_value = (self._share_value
                            - self._last_share_value) / self._last_share_value
        dprice = (self._current_price - self._last_price)/self._last_price

        step_reward = dshare_value - dprice - abs(action)*self.tax

        # if dprice > 0:
        #     step_reward = (dvalue / dprice)*100
        #     if dvalue == 0:
        #         step_reward = -dprice*100
        # if dprice < 0:
        #     step_reward = (1 - dvalue / dprice)*100
        # else:
        #     step_reward = 0
        self._total_reward += step_reward

        if self._current_tick >= self._end_tick:
            self._done = True
            if self.episode_mode:
                self.reset_episode()
                self._episode_reward.append(self._total_reward)
                self._episode_value.append(self._total_value)
        obs = self._get_observation()
        info = {"Total_Value": self._total_value, "Share_Value": self._share_value,
                "Cash_Balance": self._balance, "Total_Profit": self._total_reward}
        return obs, step_reward, self._done, info

    def _get_observation(self):
        dnn_features = np.array(self.features[self._current_tick])
        dprice = (self._current_price-self._last_price)/self._last_price
        portfolio_state = np.array([self._balance / self.initial_balance,
                                    self._total_value / self.initial_balance, self._share_value / self.initial_balance, dprice])
        current_state = np.append(dnn_features, portfolio_state)
        return current_state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_episode(self):
        start = list(range(self.start_tick, self.end_tick))
        weights = [i for i in start]
        self._start_tick = random.choices(start, weights)[0]
        self._end_tick = self._start_tick + self.episode_steps
        self._current_tick = self._start_tick
        self._episode += 1
        self._current_price = self.prices[self._current_tick]
        self._last_price = self.prices[self._current_tick - 1]
        self._balance = self.initial_balance
        self._shares = 0
        self._share_value = 0
        self._total_value = self._balance
        self._last_value = self._total_value

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
