# rl_drug_design.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

def train_policy_gradient(env, policy_network, epochs=1000, lr=0.01):
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)
    gamma = 0.99  # Discount factor

    for epoch in range(epochs):
        state = env.reset()
        rewards = []
        log_probs = []

        for _ in range(env.max_steps):
            state = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_network(state)
            action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
            next_state, reward, done, _ = env.step(action)
            log_probs.append(torch.log(action_probs[action]))
            rewards.append(reward)
            state = next_state
            if done:
                break

        # Compute cumulative rewards
        cumulative_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            cumulative_rewards.insert(0, G)
        cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float32)

        # Update policy
        policy_gradient = []
        for log_prob, G in zip(log_probs, cumulative_rewards):
            policy_gradient.append(-log_prob * G)
        loss = torch.stack(policy_gradient).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}] - Loss: {loss.item()}')

    return policy_network
