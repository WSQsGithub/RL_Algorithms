# %%
import numpy as np
import gym
import torch
import torch.nn  as nn
import torch.optim as optim

# %% 定义网络结构

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value


# A2C更新函数
def train(model, optimizer, state, action, reward, next_state, done, gamma=0.99):
    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    reward = torch.FloatTensor([reward])
    action = torch.LongTensor([action])

    probs, value = model(state)
    _, next_value = model(next_state)

    # 计算advantage
    td_target = reward + gamma * next_value * (1 - done)
    delta = td_target - value

    # 计算actor和critic的损失
    actor_loss = -torch.log(probs[action]) * delta.detach()
    critic_loss = delta ** 2

    # 合并损失并进行反向传播
    loss = actor_loss + critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# %% 训练
# 主程序: 离散动作，连续状态
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for episode in range(1000):
        state, info = env.reset()
        episode_reward = 0

        while True:
            probs, _ = model(torch.FloatTensor(state))
            action = np.random.choice(env.action_space.n, p=probs.detach().numpy())
            next_state, reward, done, _, info = env.step(action)
            train(model, optimizer, state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                print(f"Episode {episode}, Reward: {episode_reward}")
                break