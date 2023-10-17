# %% model-based RL
import numpy as np
import random
import gym

# 创建FrozenLake环境
env = gym.make('FrozenLake-v1')

# 初始化Q值表
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# 初始化模型
model = {}
for s in range(num_states):
    model[s] = {a: {'next_state': None, 'reward': 0} for a in range(num_actions)}

# 参数设置
num_episodes = 10000
alpha = 0.01  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 0.1  # 探索率
# %%
# Dyna-Q算法
for episode in range(num_episodes):
    state,_ = env.reset()
    done = False
    while not done:
        # epsilon-greedy策略
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 随机选择动作
        else:
            action = np.argmax(Q[state, :])  # 选择Q值最高的动作

        next_state, reward, done, _, _ = env.step(action) # 环境交互
        if reward != 0:
            print(next_state, reward, done)

        # 更新Q值
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        # 更新模型
        model[state][action]['next_state'] = next_state
        model[state][action]['reward'] = reward

        # 计划更新
        for _ in range(5):  # 进行5次模拟计划
            s = random.randint(0, num_states - 1)
            a = random.randint(0, num_actions - 1)
            if model[s][a]['next_state'] is not None:
                next_s = model[s][a]['next_state']
                r = model[s][a]['reward']
                Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[next_s, :]) - Q[s, a])

        state = next_state

# 输出最终的Q值表
print("Final Q-values:")
print(Q)
# %%
# 测试训练后的策略
num_test_episodes = 10
total_rewards = 0
for _ in range(num_test_episodes):
    state,_ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[int(state), :])
        next_state, reward, done, _ , _= env.step(action)
        total_rewards += reward
        state = next_state

average_reward = total_rewards / num_test_episodes
print("Average reward over {} test episodes: {:.2f}".format(num_test_episodes, average_reward))
