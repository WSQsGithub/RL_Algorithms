# %%
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random

# %%
# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action

# 定义Critic网络, twin结构
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.net2 = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net1(x), self.net2(x)

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
# TD3更新函数
def train(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, replay_buffer, max_action, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    state, action, reward, next_state, done = replay_buffer.sample(64)

    state = torch.FloatTensor(state)
    action = torch.FloatTensor(action)
    reward = torch.FloatTensor(reward).unsqueeze(1)
    next_state = torch.FloatTensor(next_state)
    done = torch.FloatTensor(done).unsqueeze(1)

    # Critic update：q-learning更新公式
    with torch.no_grad():
        noise = torch.normal(0, policy_noise, size=action.shape).clamp(-noise_clip, noise_clip) # 正太噪声
        next_action = (actor_target(next_state) + noise).clamp(-max_action, max_action) # 计算下一个状态的q值时加入噪声
        target_q1, target_q2 = critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2) # 选择最小值避免过低估计
        target_q = reward + (1 - done) * gamma * target_q
        
    current_q1, current_q2 = critic(state, action)
    critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q) # 创建了损失函数对象并立即将 current_q1 和 target_q 传递给它进行计算
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Delayed policy updates
    if it % policy_freq == 0:
        actor_loss = -critic(state, actor(state))[0].mean() # 最大化critic网络估计的q值
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(actor_target.parameters(), actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(critic_target.parameters(), critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# 主程序
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    max_action = float(env.action_space.high[0])

    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], max_action)
    critic = Critic(env.observation_space.shape[0], env.action_space.shape[0])
    actor_target = Actor(env.observation_space.shape[0], env.action_space.shape[0], max_action)
    critic_target = Critic(env.observation_space.shape[0], env.action_space.shape[0])
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    replay_buffer = ReplayBuffer(1000000)

    for episode in range(1000):
        state, info = env.reset()
        episode_reward = 0

        for it in range(200):  # Pendulum-v0默认的最大步数是200
            action = actor(torch.FloatTensor(state)).detach().numpy()
            next_state, reward, done, _ , info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > 1000:
                train(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, replay_buffer, max_action)

            if done:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")
