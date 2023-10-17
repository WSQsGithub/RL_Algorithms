# %%
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 定义Actor网络，适用于连续状态空间
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()  # 使输出在-1到1之间
        )

    def forward(self, state):
        return self.net(state)

# 定义Critic网络，输出Q值
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# %%
# DDPG更新函数
def train(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, replay_buffer, gamma=0.99, tau=0.005):
    state, action, reward, next_state, done = replay_buffer.sample(64)

    state = torch.FloatTensor(state)
    action = torch.FloatTensor(action)
    reward = torch.FloatTensor(reward).unsqueeze(1)
    next_state = torch.FloatTensor(next_state)
    done = torch.FloatTensor(done).unsqueeze(1)

    # Critic update
    with torch.no_grad():
        next_action = actor_target(next_state)
        target_q = reward + (1 - done) * gamma * critic_target(next_state, next_action)
    current_q = critic(state, action)
    critic_loss = nn.MSELoss()(current_q, target_q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor update
    actor_loss = -critic(state, actor(state)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update target networks
    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# %%
# 主程序
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")

    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    critic = Critic(env.observation_space.shape[0], env.action_space.shape[0])
    
    actor_target = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    critic_target = Critic(env.observation_space.shape[0], env.action_space.shape[0])
    
    actor_target.load_state_dict(actor.state_dict()) # 软更新
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    replay_buffer = ReplayBuffer(1000000)

    for episode in range(1000):
        state,_ = env.reset()
        episode_reward = 0

        for step in range(200):  # Pendulum-v1默认的最大步数是200
            action = actor(torch.FloatTensor(state)).detach().numpy()
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > 1000:
                train(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, replay_buffer)

            if done:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")
