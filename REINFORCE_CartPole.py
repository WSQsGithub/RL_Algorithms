# %%
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Training parameters
EPISODES = 1000
GAMMA = 0.99

# Environment setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Network and optimizer setup
policy = PolicyNetwork(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

# %%
# Training loop
for episode in range(EPISODES):
    state, _ = env.reset()
    rewards = []
    log_probs = []

    # Generate an episode
    while True: # 生成一条长度任意的轨迹，直到翻车
        env.render()
        state = torch.tensor(state, dtype=torch.float32)
        probs = policy(state) # 根据当前策略，得到当前状态下的动作分布
        action = torch.distributions.Categorical(probs).sample() # 根据分布采样策略
        log_prob = torch.log(probs[action]) # log(pi(a|s))

        next_state, reward, done, _, _ = env.step(action.item()) # 环境交互

        log_probs.append(log_prob) # 整套轨迹的所选动作概率
        rewards.append(reward)# 整条轨迹的reward

        state = next_state
        if done:
            break
    
    # 从后往前计算初始状态的状态估计
    discounted_rewards = []
    running_add = 0
    for r in rewards[::-1]: 
        running_add = r + GAMMA * running_add
        discounted_rewards.insert(0, running_add)
    discounted_rewards = torch.tensor(discounted_rewards)
    
    # Optimize the policy
    optimizer.zero_grad()
    loss = []
    for log_prob, R in zip(log_probs, discounted_rewards):
        loss.append(-log_prob * R)  # 计算优化目标
    loss = torch.stack(loss).sum() 
    loss.backward() # 反向传播
    optimizer.step()

    # Log training information
    print(f"Episode: {episode+1}, Total Reward: {np.sum(rewards)}")

env.close()
