import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
#from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal
#from utils import ReplayBuffer, get_env, run_episode


class NeuralNetwork(nn.Module):
	'''
	This class implements a neural network with a variable number of hidden layers and hidden units.
	You may use this function to parametrize your policy and critic networks.
	'''
	def __init__(self, input_dim: int, output_dim: int, hidden_size: int, hidden_layers: int, activation: str):
		super(NeuralNetwork, self).__init__()
		layers = [nn.Linear(input_dim, hidden_size)]
		layers += [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
		layers += [nn.Linear(hidden_size, output_dim)]

		self.layers = nn.ModuleList(layers)
		self.activation = activation

	def forward(self, s: torch.Tensor) -> torch.Tensor:
		# TODO: Implement the forward pass for the neural network you have defined.
		for layer in self.layers[:-1]:
			if self.activation == 'softmax':
				s = F.softmax(layer(s))
			else:
				s = F.leaky_relu(layer(s))
		return self.layers[-1](s)


class Actor(nn.Module):
	"""
	Policy
	"""
	def __init__(self, state_dim, action_dim, hidden_size = 10, hidden_layers = 0, actor_lr = 1e-4, device: torch.device = torch.device('cpu')):
		super(Actor, self).__init__()
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers
		self.actor_lr = actor_lr
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.setup_actor()

	def setup_actor(self):
		'''
		This function sets up the actor network in the Actor class.
		'''
		self.network = NeuralNetwork(self.state_dim,self.action_dim,self.hidden_size,self.hidden_layers,'relu').to(self.device)
		self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr, amsgrad=True)
	
	def forward(self, state):
		state_tensor =  torch.Tensor(state).to(self.device)
		output = self.network(state_tensor)
		action_probs = F.softmax(output)
		return action_probs
	
	def get_prob(self, s_t, a_t):
		action_probs = self.forward(torch.Tensor(s_t).to(self.device))
		prob = action_probs[a_t]
		return prob


class Critic(nn.Module):
	"""
	Q-function(s)
	"""
	def __init__(self, state_dim, action_dim, hidden_size = 10, hidden_layers = 0, critic_lr = 1e-10, device: torch.device = torch.device('cpu')):
		super(Critic, self).__init__()
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers
		self.critic_lr = critic_lr
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.network_copy = None
		self.setup_critic()

	def setup_critic(self):
		self.network = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
		self.optimizer = optim.Adam(self.parameters(), lr=self.critic_lr)

	def forward(self, s_a):
		s_a_tensor = torch.Tensor(np.array(s_a))
		return self.network.forward(s_a_tensor)