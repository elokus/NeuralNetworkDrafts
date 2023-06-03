import numpy as np
import pandas as pd
from stock_env_custom import StocksEnv

df = pd.read_csv("BTC_USDT.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.rename(columns={"volume": "Volume"})
df = df.sort_index()
df = df[1:100]

print(df)

window_size = 3


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close', 'High', 'Low', 'Volume']].to_numpy()[
                                                                        start:end]
    return prices, signal_features


class MyCustomEnv(StocksEnv):
    _process_data = add_signals


env2 = MyCustomEnv(df=df, window_size=window_size, frame_bound=(3, 100))

reset = env2.reset()

print(env2.action_space.sample())
print(en2.)
# print(reset)
# observation, step_reward, done, info, cp, lp = env2.step(0)
# print(observation)
# print(
#     f"step 1 current price: {cp} and last price {lp}, step reward is {step_reward}")
# observation, step_reward, done, info, cp, lp = env2.step(0)
# print(observation)
# print(
#     f"step 2 current price: {cp} and last price {lp}, step reward is {step_reward}")
# observation, step_reward, done, info, cp, lp = env2.step(1)
# print(observation)
# print(
#     f"step 3 print current price: {cp} and last price {lp}, step reward is {step_reward}")
# print(env2._position_history)
# print(info)
