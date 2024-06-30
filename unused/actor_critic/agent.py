
import os
import torch as T
from networks import Critic, Actor
from replay_buffer import ReplayBuffer

def Agent():
    def __init__(self, state_dim, action_dim, discount):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.buffer = ReplayBuffer(10e7, self.state_dim, self.action_dim)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.actor = Actor(self.state_dim, self.action_dim, self.discrete)

    def choose_action(self, state):
        if self.discrete:
            action_prob = self.actor(state)
            action = T.distributions.Categorical(action_prob).sample()
            return action
        
    def action_prob(self, state, action):
        if self.discrete:
            action_prob = self.actor(state)
            return action_prob[action]
    
    def advantage(self, state, action, next_state, next_action, reward):
        state_value = self.critic(state, action)
        next_state_value = self.critic(next_state, next_action)
        advantage = reward + self.discount * next_state_value-state_value
        return advantage
    
    def critic_loss(self, state, action, next_state, next_action, reward):
        advantage = self.advantage(state, action, next_state, next_action, reward)
        return T.square(advantage)
    
    def actor_loss(self, state, action, next_state, next_action, reward):
        advantage = self.advantage(state, action, next_state, next_action, reward)
        action_prob = self.actor(state)
        return T.log(action_prob[action])*advantage
    
    def update_network(self, obj, loss):
        obj.optimizer.zero_grad()
        loss.backward()
        obj.optimizer.step()