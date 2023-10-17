# %% Nash Q-learning
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 定义博弈矩阵: 决定了奖励函数怎么改
R = np.array([[3, 0], [5, 1]])

# 初始化Q值表
num_states = 2
num_actions = 2
Q = np.zeros((num_states, num_actions)) # 智能体使用共享策略

# 设置算法参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率

# 可视化相关参数
episode_rewards = []

# Nash Q-Learning算法
num_episodes = 10000
for episode in range(num_episodes):
    state = np.random.choice(num_states)
    # 一定概率下随机探索
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(num_actions)
    else:
        action = np.argmax(Q[state, :])

    next_state = state
    while next_state == state:
        next_state = np.random.choice(num_states) # 换一个随机状态，避免重复探索

    # 更新Q值
    reward = R[state, action]
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max(Q[next_state, :]))

    episode_rewards.append(reward)

    # 每1000次迭代可视化一次
    if episode % 1000 == 0:
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

# 打印学到的策略
optimal_policy = np.argmax(Q, axis=1)
print("Optimal Policy:", optimal_policy)

# %% MiniMax Q-Learning
import numpy as np

# 定义博弈矩阵
R = np.array([[3, -1], [0, 2]])

# 初始化Q值表
num_states = 2
num_actions = 2
Q = np.zeros((num_states, num_actions))

# 设置算法参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子

# Minimax Q-Learning算法
num_episodes = 10000
for episode in range(num_episodes):
    state = np.random.choice(num_states)
    action = np.argmax(Q[state, :])  # 初始策略是选择具有最大Q值的动作

    next_state = state
    while next_state == state:
        next_state = np.random.choice(num_states)

    # 更新Q值
    reward = R[state, action]
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.min(Q[next_state, :]))

# 打印学到的策略
optimal_policy = np.argmax(Q, axis=1)
print("Optimal Policy:", optimal_policy)
