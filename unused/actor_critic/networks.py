import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc_dim = 10, lr = 1e-4):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_dim = fc_dim
        self.lr = lr

        self.fc = nn.Linear(self.state_dim+self.action_dim, self.fc_dim)
        self.q = nn.Linear(self.fc_dim, 1)
        self.activation = F.relu
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.device = T.device('cuda:0' if T.cuda.is_availabe() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, action):
        state_action = T.cat([state, action], dim=1)
        action_value = self.fc(state_action)
        action_value = self.activation(action_value)
        q = self.q(action_value)
        return q
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc_dim = 10, lr = 1e-4):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_dim = fc_dim
        self.lr = lr

        self.fc = nn.Linear(self.state_dim, self.fc_dim)
        self.action = nn.Linear(self.fc_dim, self.action_dim)
        self.activation = F.relu
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.device = T.device('cuda:0' if T.cuda.is_availabe() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        action_prob = self.fc(state)
        action_prob = self.activation(action_prob)
        action_prob = self.action(action_prob)
        action_prob = F.softmax(action_prob)
        return action_prob
        
