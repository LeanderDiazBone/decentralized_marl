import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Categorical
import matplotlib.pyplot as plt

if T.cuda.is_available():
	device = T.device('cuda')
	print('using cuda')
elif T.backends.mps.is_available():
	device = T.device('mps')
	print('using mps')
else:
	device = T.device('cpu')
	print('using cpu')

class PpoMemory(Dataset):
	def __init__(self, batch_size):
		# self.state_dim = state_dim
		# self.episode_limit = episode_limit
		# self.n_agents = n_agents
		self.batch_size = batch_size

		self.states = []
		self.probs = []
		self.vals = []
		self.actions = []
		self.rewards = []
		self.dones = []
		# self.reset_buffer()

	# def reset_buffer():
	# 	self.states = torch.zeros((self.episode_limit, self.state_dim), dtype = torch.float32)
	# 	self.probs = torch.zeros((self.episode_limit, n_agents * 4), dtype = torch.float32) # 4 discrete actions
	# 	self.vals = torch.zeros((self.episode_limit), dtype = torch.float32)
	# 	self.actions = torch.zeros((self.episode_limit, n_agents), dtype = torch.float32)
	# 	self.rewards = torch.zeros((self.episode_limit), dtype=torch.float32)
	# 	self.dones = torch.zeros((self.episode_limit), dtype=torch.float32)
		
	# def __len__(self):
	# 	pass

	# def __getitem__(self,idx):
	# 	pass

	def generate_batches(self):
		n_states = len(self.states)
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype=np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]
		# return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, batches
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

class ActorNetwork(nn.Module):
	def __init__(self, n_actions, n_actions_per_agent, input_dims, alpha, hidden_dim=128, chkpt_dir='tmp/ppo'):
		super(ActorNetwork, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
		self.n_actions = n_actions
		self.n_actions_per_agent = n_actions_per_agent
		self.actor = nn.Sequential(
			nn.Linear(input_dims, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, n_actions)
		)
		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = device
		self.to(self.device)

	def forward(self, state):
		dist = self.actor(state).view(-1, self.n_actions)
		# if dist.shape[0] > 1:
		# 	print("multi")
		# print("dist output", dist[0, 0:self.n_actions_per_agent])
		dists = []
		for i in range(0, self.n_actions, self.n_actions_per_agent):
			cur_dist = dist[:, i:i+self.n_actions_per_agent]
			cur_dist = F.softmax(cur_dist, dim=1)
			dists.append(cur_dist)
		dists = [Categorical(dist) for dist in dists]
		# print("categorial", T.exp(dists[0].logits[0]))
		return dists

	def save_checkpoint(self, path: str=None):
		if not path:
			path = self.checkpoint_file
		os.makedirs(os.path.dirname(path), exist_ok=True)
		T.save(self.state_dict(), path)

	def load_checkpoint(self, path: str=None):
		if not path:
			path = self.checkpoint_file
		self.load_state_dict(T.load(path, map_location=device))

class CriticNetwork(nn.Module):
	def __init__(self, input_dims, alpha, hidden_dim=128, chkpt_dir='tmp/ppo'):
		super(CriticNetwork, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
		self.critic = nn.Sequential(
			nn.Linear(input_dims, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, 1)
		)
		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = device
		self.to(self.device)

	def forward(self, state):
		value = self.critic(state)
		return value

	def save_checkpoint(self, path: str=None):
		if not path:
			path = self.checkpoint_file
		os.makedirs(os.path.dirname(path), exist_ok=True)
		T.save(self.state_dict(), path)

	def load_checkpoint(self, path: str=None):
		if not path:
			path = self.checkpoint_file
		self.load_state_dict(T.load(path, map_location=device))

class Agent:
	# N = horizon, steps we take before we perform an update
	def __init__(self, env_name: str, n_actions: int, n_actions_per_agent:int, input_dims: int, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
			policy_clip=0.2, batch_size=64, n_epochs=10, n_agents=2, hidden_dim = 128):
		self.env_name = env_name
		self.plotter_x = []
		self.plotter_y = []
		self.gamma = gamma
		self.policy_clip = policy_clip
		self.n_epochs = n_epochs
		self.gae_lambda = gae_lambda
		self.actor = ActorNetwork(n_actions, n_actions_per_agent, input_dims, alpha, hidden_dim = hidden_dim)
		self.n_agents = n_agents
		self.batch_size = batch_size

		self.critic = CriticNetwork(input_dims, alpha, hidden_dim = hidden_dim)
		self.memory = PpoMemory(batch_size)

	def remember(self, state, action, probs, vals, reward, done):
		self.memory.store_memory(state, action, probs, vals, reward, done)

	def save_models(self, id: str=None):
		self.actor.save_checkpoint(f'./checkpoints/ppo_actor_{id}_{self.env_name}.pth')
		self.critic.save_checkpoint(f'./checkpoints/ppo_critic_{id}_{self.env_name}.pth')

	def load_models(self, id: str=None):
		self.actor.load_checkpoint(f'./checkpoints/ppo_actor_{id}_{self.env_name}.pth')
		self.critic.load_checkpoint(f'./checkpoints/ppo_critic_{id}_{self.env_name}.pth')

	def choose_action(self, observation: np.array):
		state = T.tensor(np.array(observation), dtype=T.float32).to(self.actor.device)
		dists = self.actor(state)
		value = self.critic(state)

		actions = [dist.sample() for dist in dists]

		actions = T.tensor(actions, dtype=int).to(self.actor.device)

		log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]
		log_probs = T.stack(log_probs)

		probs = T.squeeze(log_probs)
		if actions.shape[0] != 1:
			actions = T.squeeze(actions)
		value = T.squeeze(value).item()

		actions = actions.cpu().detach()
		probs = probs.cpu().detach()
		state = state.cpu().detach()

		return actions, probs, value

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

				dists = self.actor(states)
				critic_values = self.critic(states)
				critic_values = T.squeeze(critic_values)

				new_probs = []
				# for i in range(self.batch_size):
				# 	cur_batch = []
				# 	for j in range(self.n_agents):
				# 		cur_batch.append(dists[j].log_prob(actions[i,j])[i])
				# 	new_probs.append(cur_batch)

				for i, dist in enumerate(dists):
					new_probs.append(dist.log_prob(actions[:, i]))
				new_probs = T.stack(new_probs, dim=1)
				
				# old = old_probs[:,0]
				# new_probs = T.tensor(new_probs).to(self.actor.device)
				# new = new_probs[:,0]
				# for i in range(1, old_probs.shape[-1]):
				# 	old += old_probs[:,i]
				# 	new += new_probs[:,i]

				prob_ratio_prod = (new_probs.exp() / old_probs.exp()).prod(dim=1)

				# new_probs = T.tensor(new_probs).to(self.actor.device)
				# prob_ratio = new_probs.exp() / old_probs.exp()
				# prob_ratio_prod = prob_ratio[:,0]
				# for i in range(1, prob_ratio.shape[-1]):
				# 	prob_ratio_prod *= prob_ratio[:,i]

				weighted_probs = advantages[batch] * prob_ratio_prod
				weighted_clipped_probs = T.clamp(prob_ratio_prod, 1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch]
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

				# print('before', self.actor.state_dict()['actor.0.weight'])
				self.actor.optimizer.step()
				self.critic.optimizer.step()

				# print('after', self.actor.state_dict()['actor.0.weight'])
				

				# if len(self.plotter_x) > 10000:
				# 	# print a plot and save it with the self.plotter_x and self.plotter_y
				# 	plt.plot(self.plotter_x, self.plotter_y)
				# 	plt.savefig('/Users/georgye/Documents/repos/ml/backprop/plots/ppo.png')
				# 	plt.close()
				# 	raise Exception('plotted')
		self.memory.clear_memory()
