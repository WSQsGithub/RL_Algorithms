# %%
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Neural Network Model Definition
class PolicyNetwork(nn.Module):
    def __init__(self, n_state, n_action, n_hidden=32):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Create the environment
env = gym.make("CartPole-v1")
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

# Hyperparameters
lr = 0.001
gamma = 0.99

# Initialize network and optimizer
policy = PolicyNetwork(n_state, n_action)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# %%
# PPO Training Loop
n_episode = 500
clip_epsilon = 0.2
for episode in range(n_episode):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    states = []
    actions_taken = []
    
    # 生成一条长度为200的轨迹
    for _ in range(200):  # episode length is 200 for CartPole-v1
        states.append(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # 计算向左向右的概率
        probs = policy(state_tensor)
        # 根据多项分布选择一个action
        action = torch.multinomial(probs, 1).item()
        actions_taken.append(action) 
        
        # 环境交互
        next_state, reward, done, _, _= env.step(action) # 每个时间步：只要摆没有掉下或者小车没有移动到屏幕边界之外，agent就会得到一个奖励值为1.0的奖励
        
        # 记录下选择这个action的概率
        log_prob = torch.log(probs[0][action])
        log_probs.append(log_prob)
        rewards.append(reward)
        
        state = next_state
        if done:
            break
    
    # 计算该条轨迹的总return
    returns = []
    G = 0
    for r in reversed(rewards): # 从后往前计算累积奖励 
        G = r + gamma * G  # 为当前时间步t计算折扣累计奖励
        returns.insert(0, G) # 最新的放在最前面
    returns = torch.tensor(returns) # 返回1-T时刻的累积奖励
    
    # Update policy using PPO
    old_probs = torch.stack(log_probs).detach() # 不包含梯度信息
    for _ in range(10):  # 每条轨迹跑10次梯度下降
        current_probs_log = policy(torch.FloatTensor(states)).log() # 最新策略下，轨迹下每个时刻选中各个action的概率
        action_indices = torch.tensor(actions_taken, dtype=torch.int64).unsqueeze(1)  # 每个时刻选择的动作
        selected_log_probs = current_probs_log.gather(1, action_indices).squeeze() # 每个时刻选择动作的概率
        ratios = torch.exp(selected_log_probs - old_probs) # 计算新策略与旧策略的差异
        
        surrogate1 = ratios * returns # 这条轨迹的return根据差别打个折，return更好的轨迹更新的幅度越大
        surrogate2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * returns # 钳位函数
        loss = -torch.min(surrogate1, surrogate2).mean() # 希望新策略比旧策略好，但是更新不要过大

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training Complete!")

# %%

# Test policy
total_reward = 0
state, _ = env.reset()
for _ in range(200):
    env.render()
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    probs = policy(state_tensor)
    action = torch.multinomial(probs, 1).item()
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
    if done:
        break

print(f"Total reward after training: {total_reward}")
