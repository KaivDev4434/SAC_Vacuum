import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define experience replay buffer
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# DDPG Agent
class DDPGAgent:
    def __init__(
        self, state_dim, action_dim, max_action, buffer_size=1e6, batch_size=64
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(int(buffer_size))
        self.batch_size = batch_size

        self.max_action = max_action
        self.action_dim = action_dim

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).cpu().data.numpy()

        # Add exploration noise
        action = action + np.random.normal(0, noise, size=self.action_dim)

        # Clip the action to be within bounds
        return np.clip(action, -self.max_action, self.max_action)

    def update(self, gamma=0.99, tau=0.005):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors
        state_batch = torch.FloatTensor(np.vstack(batch.state)).to(device)
        action_batch = torch.FloatTensor(np.vstack(batch.action)).to(device)
        next_state_batch = torch.FloatTensor(np.vstack(batch.next_state)).to(device)
        reward_batch = torch.FloatTensor(np.vstack(batch.reward)).to(device)
        done_batch = torch.FloatTensor(np.vstack(batch.done)).to(device)

        # Compute Q targets
        target_actions = self.actor_target(next_state_batch)
        target_q_values = self.critic_target(next_state_batch, target_actions)
        q_targets = reward_batch + gamma * target_q_values * (1 - done_batch)

        # Update critic
        current_q_values = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q_values, q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
