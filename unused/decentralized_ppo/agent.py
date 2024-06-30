import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
device = T.device('cpu')

class PpoMemory:
	def __init__(self, batch_size: int):
		self.states = []
		self.probs = []
		self.vals = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.batch_size = batch_size

	def generate_batches(self):
		n_states = len(self.states)
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype=np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]
		return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

	def store_memory(self, state, action, probs, vals, reward, done):
		self.states.append(state)
		self.actions.append(action)
		self.probs.append(probs)
		self.vals.append(vals)
		self.rewards.append(reward)
		self.dones.append(done)

	def clear_memory(self):
		self.states = []
		self.probs = []
		self.actions = []
		self.rewards = []
		self.dones = []

	def get_size(self):
		return len(self.states)

class ActorNetwork(nn.Module):
	def __init__(self, n_actions, n_agents, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
		super(ActorNetwork, self).__init__()
		self.n_agents = n_agents
		self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

		self.actor = nn.Sequential(
			nn.Linear(*input_dims, fc1_dims),
			nn.ReLU(),
			nn.Linear(fc1_dims, fc2_dims),
			nn.ReLU(),
			nn.Linear(fc2_dims, n_agents*n_actions),
			nn.Unflatten(1, (n_agents, n_actions)),
			nn.Softmax(dim=-1)
		)
		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = device
		self.to(self.device)

	def forward(self, state):
		probs = self.actor(state)
		distrs = []
		for i in range(state.shape[0]):
			distrs.append([])
			for j in range(self.n_agents):
				distrs[i].append(Categorical(probs[i][j]))
		return distrs
		
		"""x = self.fc1(state)
		x = self.activation(x)
		x = self.fc2(x)
		x = self.activation(x)
		distrs = []
		for i in range(self.n_agents):
			ag = self.agents[i](x)
			ag = self.softmax(ag)
			distrs.append(Categorical(ag))
		return distrs"""

	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
	def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
		super(CriticNetwork, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
		self.critic = nn.Sequential(
			nn.Linear(*input_dims, fc1_dims),
			nn.Tanh(),
			nn.Linear(fc1_dims, fc2_dims),
			nn.Tanh(),
			nn.Linear(fc2_dims, 1)
		)
		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = device
		self.to(self.device)
	
	def forward(self, state):
		value = self.critic(state)
		return value
	
	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(T.load(self.checkpoint_file))

class Centralized_PPO_Agent:
	# N = horizon, steps we take before we perform an update
	def __init__(self, n_actions, n_agents, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
			policy_clip=0.2, batch_size=10, n_epochs=3, train = True):
		self.plotter_x = []
		self.plotter_y = []
		self.gamma = gamma
		self.policy_clip = policy_clip
		self.n_epochs = n_epochs
		self.gae_lambda = gae_lambda
		self.n_agents = n_agents
		self.train = train

		self.actor = ActorNetwork(n_actions, n_agents, input_dims, alpha)
		self.critic = CriticNetwork(input_dims, alpha)
		self.memory = PpoMemory(batch_size)

	def remember(self, state, action, probs, vals, reward, done):
		if self.train:
			self.memory.store_memory(state, action, probs, vals, reward, done)
			#if self.memory.get_size() >= 50:
			#	self.learn()

	def save_models(self):
		print('... saving models ...')
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()

	def load_models(self):
		print('... loading models ...')
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()

	def choose_action(self, observation: np.array):
		#print(observation)
		state = T.tensor([observation], dtype=T.float32).to(self.actor.device)
		distrs = self.actor(state)[0]
		value = self.critic(state)
		actions = []
		probs = []
		for i in range(self.n_agents):
			action = distrs[i].sample()
			actions.append(T.squeeze(action).item())
			probs.append(T.squeeze(distrs[i].log_prob(action)).item())

		#probs = T.squeeze(dist.log_prob(action)).item()
		#action = T.squeeze(action).item()
		#value = T.squeeze(value).item()
		return actions, probs, T.squeeze(value).item(), None

	def learn(self):
		for _ in range(self.n_epochs):
			state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
			values = vals_arr
			advantages = np.zeros(len(reward_arr), dtype=np.float32)

			for t in range(len(reward_arr) - 1):
				discount = 1
				a_t = 0
				for k in range(t, len(reward_arr) - 1):
					a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
					discount *= self.gamma * self.gae_lambda
				advantages[t] = a_t
			advantages = T.tensor(advantages, dtype=T.float32).to(self.actor.device)
			values = T.tensor(values, dtype=T.float32).to(self.actor.device)
			for batch in batches:
				states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
				old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device)
				actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.actor.device)
				critic_values = self.critic(states)
				critic_values = T.squeeze(critic_values)
				distrs = self.actor(states)
				new_probs = []
				for j in range(actions.shape[0]):
					new_prob = 0
					for i in range(self.n_agents):
						new_prob += distrs[j][i].log_prob(actions[j][i])
					new_probs.append(new_prob)
				new_probs = T.Tensor(new_probs)

				prob_ratio = new_probs.exp() / old_probs.exp()

				weighted_probs = advantages[batch] * prob_ratio
				weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch]
				actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
				returns = advantages[batch] + values[batch]
				critic_loss = (returns - critic_values) ** 2
				critic_loss = critic_loss.mean()

				self.plotter_x.append(len(self.plotter_x) + 1)
				self.plotter_y.append(critic_loss.item())
				total_loss = actor_loss + 0.5 * critic_loss
				self.actor.optimizer.zero_grad()
				self.critic.optimizer.zero_grad()
				total_loss.mean().backward()
				self.actor.optimizer.step()
				self.critic.optimizer.step()

				# if len(self.plotter_x) > 10000:
				# 	# print a plot and save it with the self.plotter_x and self.plotter_y
				# 	plt.plot(self.plotter_x, self.plotter_y)
				# 	plt.savefig('/Users/georgye/Documents/repos/ml/backprop/plots/ppo.png')
				# 	plt.close()
				# 	raise Exception('plotted')
		self.memory.clear_memory()
