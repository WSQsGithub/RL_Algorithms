# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random

# 设定超参数
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

# 定义 Dueling DQN 网络
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        # 特征提取网络
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        
        # 优势a函数
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # 状态价值函数
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

# 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.float32)
        
    def __len__(self):
        return len(self.memory)
# %%
# 主训练过程
def train(env, policy_net, target_net, optimizer, memory):
    epsilon = EPSILON_START
    all_rewards = []
    episode_reward = 0
    
    state,_ = env.reset()
    
    for steps in range(1, 10001):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor) 
            action = q_values.argmax().item()
            
        next_state, reward, done, _ , _= env.step(action)
        memory.push(state, action, reward, next_state, done)
        episode_reward += reward
        
        state = next_state
        
        if done:
            state, _ = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(memory) > BATCH_SIZE:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(BATCH_SIZE)
            
            batch_state = torch.FloatTensor(batch_state)
            batch_action = torch.LongTensor(batch_action).unsqueeze(1)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
            batch_next_state = torch.FloatTensor(batch_next_state)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1)
            
            q_values = policy_net(batch_state) 
            q_values = q_values.gather(1, batch_action)
            
            next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1)
            target = batch_reward + (1 - batch_done) * GAMMA * next_q_values
            
            loss = nn.MSELoss()(q_values, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 软更新
        if steps % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
        
        if steps % 100 == 0:
            mean_reward = np.mean(all_rewards[-10:])
            print(f"Step: {steps}, Mean Reward: {mean_reward}")
            
        if np.mean(all_rewards[-10:]) >= 195:
            print(f"Solved in {steps} steps!")
            break
            
    return all_rewards

env = gym.make("CartPole-v1")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy_net = DuelingDQN(input_dim, output_dim) 
target_net = DuelingDQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)

all_rewards = train(env, policy_net, target_net, optimizer, memory)
env.close()
