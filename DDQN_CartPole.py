# %%
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Neural network for Q-function approximation
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
batch_size = 64
epsilon_max = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
memory_size = 10000
train_start = 1000
update_target = 100

# Environment setup
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# DQN and target DQN networks
q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

# Replay memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayBuffer(memory_size)
epsilon = epsilon_max
steps = 0

# %%

# Training loop
for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    
    for _ in range(200):  # maximum episode length is 200
        steps += 1

        # Epsilon-greedy action
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_net(torch.FloatTensor(state)) # 使用q_net确定动作
            action = q_values.argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if done:
            break

        # Training from replay buffer using Double DQN logic
        if len(memory) > train_start:
            transitions = memory.sample(batch_size)
            batch = list(zip(*transitions))

            states = torch.FloatTensor(batch[0])
            actions = torch.LongTensor(batch[1]).unsqueeze(1)
            rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
            next_states = torch.FloatTensor(batch[3])
            dones = torch.FloatTensor(batch[4]).unsqueeze(1)

            current_q = q_net(states).gather(1, actions)

            # DDQN logic
            next_actions = q_net(next_states).argmax(1, keepdim=True) # 使用q_net选择动作
            next_q = target_net(next_states).gather(1, next_actions).detach() # 使用target_net估计q值
            
            target = rewards + gamma * (1 - dones) * next_q

            loss = nn.functional.smooth_l1_loss(current_q, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 延迟更新
            if steps % update_target == 0:
                target_net.load_state_dict(q_net.state_dict()) 

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode}, Total Reward: {total_reward}")

# Close the environment
env.close()
