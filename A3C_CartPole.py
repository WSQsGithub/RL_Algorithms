# %%
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing

# %%
# 定义网络结构
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

# %%
# A3C更新函数
def train(global_model, optimizer, state, action, reward, next_state, done, gamma=0.99):
    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    reward = torch.FloatTensor([reward])
    action = torch.LongTensor([action])

    probs, value = global_model(state)
    _, next_value = global_model(next_state)

    td_target = reward + gamma * next_value * (1 - done)
    delta = td_target - value

    actor_loss = -torch.log(probs[action]) * delta.detach()
    critic_loss = delta ** 2

    loss = actor_loss + critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 工作线程
def worker(global_model, optimizer, worker_id):
    env = gym.make("CartPole-v1")
    state, info = env.reset()
    while True:
        action_probs, _ = global_model(torch.FloatTensor(state))
        action = np.random.choice(env.action_space.n, p=action_probs.detach().numpy())
        next_state, reward, done, _, info = env.step(action)
        train(global_model, optimizer, state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()

# %%
if __name__ == "__main__":
    global_model = ActorCritic(4, 2)
    global_model.share_memory()  # 允许多进程共享模型参数
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    processes = []
    for i in range(multiprocessing.cpu_count()):  # 使用所有可用的CPU核心
        p = multiprocessing.Process(target=worker, args=(global_model, optimizer, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
