# Gym
import gym
import gym_anytrading

# Stable baselines
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#custom Indicator
from TradingEnv2 import TradingEnv
from dnn_multitimeframe_preprocessing import data_batch

features, prices, param = data_batch(start=0, end=210240)
env3 = TradingEnv(prices, features, max_steps=42)
# print(env2.signal_features.shape)
# print(len(env2.action_space.shape))
#

#
env = DummyVecEnv([lambda: env3])
env.reset()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_log/")
model.learn(total_timesteps=1000000)
model.save("PPO_exp_reward")
test_features, test_prices, test_param = data_batch(start=315360, end=525600)
env = TradingEnv(prices, features,  max_steps=0)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(20, 10))
plt.cla()
env.render_all()
plt.show()

test_features, test_prices, test_param = data_batch(start=315360, end=525600)
env = TradingEnv(test_prices, test_features,  max_steps=0)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(20, 10))
plt.cla()
env.render_all()
plt.show()
