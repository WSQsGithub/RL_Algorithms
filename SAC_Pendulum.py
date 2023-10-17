# %% import repos
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# %% Hyperparameters
ALPHA = 0.2
GAMMA = 0.99
TAU = 0.005 
LR = 0.0003
POLYAK = 0.995 
UPDATE_AFTER = 1000
BATCH_SIZE = 256

# %% Neural Networks
# Critic网络输入状态，输出Q值，使用TD error进行更新
# 这里两个独立的神经网络成为双Q网络，使用两个网络并取其最小值，主要目的是为了抑制过度估计的倾向，稳定训练过程
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Linear(state_dim, 1)
        self.q2 = nn.Linear(state_dim, 1)

    def forward(self, state):
        q1 = self.q1(state)
        q2 = self.q2(state)
        return q1, q2
    
# Actor网络输入状态，输出动作，使用
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.pi = nn.Linear(state_dim, action_dim) # 输出各个action的概率

    def forward(self, state):
        return torch.tanh(self.pi(state)) # 将各个概率归一化至0-1

# Buffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0 # 用指针实现队列的操作

    # 向replay buffer中装载一个经验
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    # 随机取出batch_size个样本出来
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
        return np.vstack(state), np.vstack(action), np.vstack(reward), np.vstack(next_state), np.vstack(done)

    def __len__(self):
        return len(self.buffer)

# %%
# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.critic_target = Critic(state_dim) # soft update target
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return action

    # 一次学习循环
    def train(self, replay_buffer, batch_size=BATCH_SIZE):
        # Sample from buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            # actor网络给出下一个动作
            next_action = self.actor(next_state)
            # critic网络给出q值，取最小值作为输出
            q1_next, q2_next = self.critic_target(next_state)
            min_q_next = torch.min(q1_next, q2_next)
            next_value = min_q_next - ALPHA * torch.log(next_action).sum(dim=1, keepdim=True)
            # 计算当前状态的q值
            target_q_value = reward + GAMMA * (1-done) * next_value

        q1, q2 = self.critic(state)
        q1_loss = 0.5 * (q1 - target_q_value).pow(2).mean()
        q2_loss = 0.5 * (q2 - target_q_value).pow(2).mean()
        q_loss = q1_loss + q2_loss
        
        # 更新critic网络
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # 更新actor网络
        pi = self.actor(state)
        q1_pi, q2_pi = self.critic(state)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (ALPHA * torch.log(pi).sum(dim=1, keepdim=True) - min_q_pi).mean() 

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新critic_target参数
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(POLYAK * target_param.data + (1-POLYAK) * param.data)
            
# %% Main
env = gym.make('Pendulum-v1') # 连续动作，连续状态
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = SACAgent(state_dim, action_dim)

replay_buffer = ReplayBuffer(1000)

# Training Loop
num_episodes = 2000
total_steps = 0
for i_episode in range(num_episodes):
    state,_ = env.reset()
    episode_reward = 0
    for t in range(1000): # Or until done
        if total_steps > UPDATE_AFTER:
            agent.train(replay_buffer, BATCH_SIZE)

        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        total_steps += 1
        env.render()

        if done or t%1000==0:

            print(f"Episode: {i_episode+1}, Reward: {episode_reward}, Total Steps: {total_steps}")
            break

# Test the policy
# ...

env.close()