import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#custom
from TradingEnv import TradingEnv
from data_preprocessing import data_batch
from customCallback import TensorboardCallback

features = np.loadtxt('data/features2.csv', delimiter=',')
prices = np.loadtxt('data/prices2.csv', delimiter=',')

# #features, prices, param = data_batch(start=315360, end=525600)
# features, prices, param = data_batch(start=0, end=210240)
# test_features, test_prices, test_param = data_batch(start=0, end=17760)
# np.savetxt('data/features_small.csv', test_features, delimiter=",")
# np.savetxt('data/prices_small.csv', test_prices, delimiter=",")


env3 = TradingEnv(prices, features, episode_steps=42)

env = DummyVecEnv([lambda: env3])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("PPO_exp_reward4")


env = TradingEnv(prices, features,  episode_steps=0)
model = PPO.load("PPO_exp_reward4", env=env)
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
