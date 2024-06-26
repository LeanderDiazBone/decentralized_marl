from typing import List
import numpy as np


# this class is used to store the experiences for each agent's subpolicy
class MultiAgentReplayBuffer:
	def __init__(self, max_size: int, critic_dims: int, actor_dims: int, n_actions: int, n_agents: int, batch_size: int, agent_names: List[str]):
		self.mem_size = max_size
		self.mem_cntr = 0
		self.n_agents = n_agents
		self.actor_dims = actor_dims
		self.batch_size = batch_size
		self.n_actions = n_actions
		self.agent_names = agent_names

		self.state_memory = np.zeros((self.mem_size, critic_dims))
		self.new_state_memory = np.zeros((self.mem_size, critic_dims))
		self.reward_memory = np.zeros((self.mem_size, n_agents))
		self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

		self.init_actor_memory()

	def init_actor_memory(self):
		self.actor_state_memory = []
		self.actor_new_state_memory = []
		self.actor_action_memory = []

		for i in range(self.n_agents):
			self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
			self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
			self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions[i])))

	# Store a new experience in the memory buffer
	def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
		# circular buffer (when buffer is full, replace the old with new)
		index = self.mem_cntr % self.mem_size

		for agent_idx, agent_name in enumerate(self.agent_names):
			self.actor_state_memory[agent_idx][index] = raw_obs[agent_name]
			self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_name]
			self.actor_action_memory[agent_idx][index] = action[agent_name]

		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = [i for i in reward.values()]
		self.terminal_memory[index] = done
		self.mem_cntr += 1


	# Sample a batch of experiences from the memory buffer
	def sample_buffer(self):
		max_mem = min(self.mem_cntr, self.mem_size)

		# Randomly sample indices
		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		states = self.state_memory[batch]
		rewards = self.reward_memory[batch]
		states_ = self.new_state_memory[batch]
		terminal = self.terminal_memory[batch]

		actor_states = []
		actor_new_states = []
		actions = []

		for agent_idx in range(self.n_agents):
			actor_states.append(self.actor_state_memory[agent_idx][batch])
			actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
			actions.append(self.actor_action_memory[agent_idx][batch])

		return actor_states, states, actions, rewards, \
			   actor_new_states, states_, terminal

	# Check if the buffer has enough samples to start training
	def ready(self):
		if self.mem_cntr >= self.batch_size:
			return True
