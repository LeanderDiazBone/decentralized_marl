import copy
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
#from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal
#from utils import ReplayBuffer, get_env, run_episode
import actor_critic

class Agent:
	def __init__(self, state_dim, action_dim, num_agents, agent_id, mu_0, beta, discount):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.num_agents = num_agents
		self.agent_id = agent_id
		self.mu = mu_0
		self.beta = beta

		self.discount = discount
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		#self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
		self.setup_agent()

	def setup_agent(self):
		self.actor = actor_critic.Actor(self.state_dim, self.action_dim).to(self.device)
		self.critic = actor_critic.Critic(self.state_dim, self.num_agents).to(self.device)

	def get_action(self, s):
		with torch.no_grad():
			state_tensor = torch.Tensor(np.array(s)).to(self.device)
			action_prob = self.actor(state_tensor)
			action = torch.distributions.Categorical(torch.Tensor(action_prob)).sample()
		return np.atleast_1d(action)

	@staticmethod
	def run_gradient_update_step(object, loss: torch.Tensor):
		object.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(object.network.parameters(), 1)
		object.optimizer.step()

	def update_mu(self, r):
		beta = self.beta
		self.mu = (1-beta) * self.mu + beta * r

	def update_critic(self, s_t, a_t, s_tn, a_tn, r):
		delta = r - self.mu + self.critic(s_tn+a_tn)-self.critic(s_t+a_t)
		delta_2 = r+self.discount*self.critic(s_tn+a_tn)-self.critic(s_t+a_t)
		self.run_gradient_update_step(self.critic, delta_2)

	def update_actor(self, s_t, a_t):
		A = self.critic(s_t+a_t)
		action_probs = self.actor.forward(s_t)
		for a_i in range(self.action_dim):
			a = a_t
			a[self.agent_id] = a_i
			A -= action_probs[a_i]*self.critic(s_t+a)
		#log_prob = torch.log(self.actor.get_prob(s_t, a_t[self.agent_id]))
		self.run_gradient_update_step(self.actor, A)
		"""psi = torch.autograd.grad(log_prob, self.actor.parameters(), retain_graph = True)
		psi = [A * grad for grad in psi]
		for param, grad in zip(self.actor.parameters(), psi):
			param.data.add(self.beta * grad)"""

	def get_omega(self):
		return self.critic.network.parameters()
	
	def set_omega(self, other_critics):
		self.critic.network_copy = None
		if len(other_critics) == 0:
			return
		self.critic.network_copy = copy.deepcopy(self.critic.network)
		for i in range(len(self.critic.network.layers)):
			weight = self.critic.network.layers[i].weight.detach()
			bias = self.critic.network.layers[i].bias.detach()
			for j in range(len(other_critics)):
				weight += other_critics[j].network.layers[i].weight.detach()
				bias += other_critics[j].network.layers[i].bias.detach()
			self.critic.network_copy.layers[i].weight.data.copy_(weight)
			self.critic.network_copy.layers[i].bias.data.copy_(bias)
			
	def update_omega(self):
		if not self.critic.network_copy is None:
			self.critic.network = self.critic.network_copy