import numpy as np
import torch


class Buffer:
	def __init__(self, n_agents: int, obs_dim: int, state_dim: int, episode_limit: int, batch_size: int):
		self.n_agents = n_agents
		self.obs_dim = obs_dim
		self.state_dim = state_dim
		self.episode_limit = episode_limit
		self.batch_size = batch_size
		self.episode_num = 0
		self.buffer = None
		self.reset_buffer()
		# create a buffer (dictionary)

	def reset_buffer(self):
		self.buffer = {
			'obs_n': np.zeros([self.batch_size, self.episode_limit, self.n_agents, self.obs_dim]),
			's': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
			'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.n_agents]),
			'a_n': np.zeros([self.batch_size, self.episode_limit, self.n_agents]),
			'a_logprob_n': np.zeros([self.batch_size, self.episode_limit, self.n_agents]),
			'r_n': np.zeros([self.batch_size, self.episode_limit, self.n_agents]),
			'done_n': np.zeros([self.batch_size, self.episode_limit, self.n_agents])
		}
		self.episode_num = 0

	def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
		self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
		self.buffer['s'][self.episode_num][episode_step] = s
		self.buffer['v_n'][self.episode_num][episode_step] = v_n
		self.buffer['a_n'][self.episode_num][episode_step] = a_n
		self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
		self.buffer['r_n'][self.episode_num][episode_step] = r_n
		self.buffer['done_n'][self.episode_num][episode_step] = done_n

	def store_last_value(self, episode_step, v_n):
		self.buffer['v_n'][self.episode_num][episode_step] = v_n
		self.episode_num += 1

	def get_training_data(self):
		batch = {}
		for key, val in self.buffer.items():
			batch[key] = torch.tensor(val, dtype=torch.long if key == 'a_n' else torch.float32)
		return batch
