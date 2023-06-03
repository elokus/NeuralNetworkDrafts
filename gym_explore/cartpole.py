import gym

from stable_baselines3 import DQN

env = gym.make('CartPole-v0')

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000, log_interval=4)
model.save("dqn_cartpole")

model = DQN.load("dqn_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()
#
# print(env.action_space)
# for episode in range(10):
#     observation = env.reset()
#     print("initial observation for episode {}".format(episode))
#     print(observation)
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(action)
#         print(observation)
#
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
