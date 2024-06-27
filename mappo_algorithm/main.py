import time
import gym
import torch
import numpy as np
import os
from multiprocessing import Process
import matplotlib.pyplot as plt
import pandas as pd
from normalization import Normalization
from agent import Agent
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_discrete import CatMouseMAD



class CatMouse:
	@staticmethod
	def get_action(action):
		action_dict = {
			0: 0,
			1: 0.25,
			2: 0.5,
			3: 0.75
		}
		return action_dict[action]

	@staticmethod
	def trans_obs(obs):
		ret = []
		for agent_obs in obs:
			temp = []
			temp.append(agent_obs['agents']['cur_agent'])
			for agent_pos in agent_obs['agents']['position']:
				temp.append(agent_pos)
			for prey_pos in agent_obs['prey']['position']:
				temp.append(prey_pos)
			temp.append(agent_obs['prey']['caught'])
			ret.append(np.concatenate(temp))
		return np.array(ret)

	@staticmethod
	def trans_state(state):
		ret = []
		for agent_pos in state['agents']['position']:
			ret += agent_pos.tolist()
		for i, prey_pos in enumerate(state['prey']['position']):
			ret += prey_pos.tolist()
			ret.append(state['prey']['caught'][i])
		return np.array(ret)

	def __init__(self, evaluate=False):
		self.env = CatMouseMA(observation_radius=1, n_agents=2, n_prey=2)
		self.state_dim = self.env.n_agents * 2 + self.env.n_prey * 3
		self.obs_dim = self.env.n_agents * 3 + self.env.n_prey * 3
		self.action_dim = 4
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n = np.array(self.trans_obs(obs_n))
		return obs_n, info, self.trans_state(self.env.get_global_obs())

	def step(self, a_n):
		obs_next_n, r_n, done_n, trunc, info = self.env.step([self.get_action(a) for a in a_n])
		obs_next_n = self.trans_obs(obs_next_n)
		done_n = [done_n]
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, done_n, trunc, info, self.trans_state(self.env.get_global_obs())

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


class CatMouseDiscrete:
	@staticmethod
	def get_action_discrete(action):
		ACTION_LIST = np.array([
			[0,0],
			[0,1],
			[1,0],
			[1,1],
			[0,-1],
			[-1,0],
			[-1,-1],
			[1,-1],
			[-1,1]
		])
		return ACTION_LIST[action]

	@staticmethod
	def trans_obs_discrete(obs):
		ret = []
		for ob in obs:
			temp = []
			agent_grid = ob["agent_grid"].flatten()
			prey_grid = ob["prey_grid"].flatten()
			agent_pos = ob["agent_pos"]
			agent_id = np.array([ob["agent_id"]])
			temp.append(agent_grid)
			temp.append(prey_grid)
			temp.append(agent_pos)
			temp.append(agent_id)
			temp = np.concatenate(temp)
			ret.append(temp)
		return np.array(ret)

	@staticmethod
	def trans_state_discrete(state):
		ret = []
		agent_grid = state["grids"]["agents"].flatten()
		prey_grid = state["grids"]["prey"].flatten()
		agent_pos = state["agent_pos"].flatten() # if we have 1 grid per agent then dont need agent_pos anymore
		ret.append(agent_grid)
		ret.append(prey_grid)
		ret.append(agent_pos)
		ret = np.concatenate(ret)
		return ret

	def __init__(self, evaluate=False, n_agents=2, n_prey=4, grid_size=5, observation_radius=1, ma=True):
		self.env = CatMouseMAD(observation_radius=observation_radius, n_agents=n_agents, n_prey=n_prey, grid_size=grid_size)
		self.state_dim = n_agents * (9 * 2) + n_agents * 2 # local lumberjack state
		self.obs_dim = ((observation_radius * 2 + 1) ** 2) * 2 +  3 # local observation, 2 local grids + cur agent position + id
		self.n_actions_per_agent = 9
		self.ma = ma
		if self.ma: # if multi-agent, else use normal ppo
			self.action_dim = self.n_actions_per_agent
		else:
			self.action_dim = self.n_actions_per_agent * n_agents

		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n = np.array(self.trans_obs_discrete(obs_n))
		return obs_n, info, self.trans_state_discrete(self.env.get_global_obs())

	def step(self, a_n):
		obs_next_n, r_n, done_n, trunc, info = self.env.step(self.get_action_discrete(a_n))
		obs_next_n = self.trans_obs_discrete(obs_next_n)
		if not self.ma:
			r_n = sum(r_n)

		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, [done_n for _ in range(self.n_agents)], trunc, info, self.trans_state_discrete(self.env.get_global_obs())

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


class Lumberjacks:
	def __init__(self, evaluate=False, grid_size=5, n_agents=2, n_trees=8, observation_rad=1):
		self.env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees) #n_trees=8,
		self.state_dim = np.sum([self.env.observation_space[agent].shape[0] for agent in range(self.env.n_agents)]) # (grid_size ** 2 ) * 2 
		self.obs_dim = self.env.observation_space[1].shape[0]
		self.action_dim = 5
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate
		self.grid_size = grid_size

	def reset(self):
		obs_n = self.env.reset()
		obs_n = np.array(obs_n)
		return obs_n, None, obs_n.flatten() # self.get_global_obs() #

	def step(self, a_n):
		obs_next_n, r_n, done_n, info = self.env.step(a_n)
		obs_next_n = np.array(obs_next_n)
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, done_n, None, info, obs_next_n.flatten() #  self.get_global_obs() #

	def get_global_obs(self):
		def get_value(x, y):
			return x * self.grid_size + y
		global_env = self.env.get_global_obs()
		glob_state = [0 for _ in range(self.state_dim)]
		agent_pos = global_env[0]
		for i, agent in enumerate(agent_pos):
			glob_state[get_value(agent[0] - glob_state[1] - 1, agent[1] - glob_state[2] - 1) + 3] += 1

		tree_pos = global_env[1]
		for pos, strength in tree_pos:
			temp = get_value(pos[0] - 1, pos[1] - 1)
			glob_state[temp + (self.grid_size**2)] = strength
		return glob_state

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


def train(agent: Agent, env, n_games=10000):
	reward_norm = Normalization(shape=agent.N)
	total_episodes = 0
	total_steps = 0
	log_interval = 200
	score_history = []
	for epi in range(n_games):
		episode_reward = 0
		obs_n, _, s = env.reset()
		for episode_step in range(50):
			a_n, a_logprob_n = agent.choose_action(obs_n, evaluate=False)
			v_n = agent.get_value(s)
			obs_next_n, r_n, done_n, _, _, s_next = env.step(a_n)
			episode_reward += sum(r_n)
			r_n = reward_norm([r_n])
			agent.buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)
			obs_n = np.array(obs_next_n)
			s = s_next
			total_steps += 1
			if all(done_n):
				break
		if epi % log_interval == 0:
			print(f'Episode:{epi}, Reward:{episode_reward}, Total Steps:{total_steps}')
		score_history.append(episode_reward)
		# Store v_n in the last step
		v_n = agent.get_value(s_next)
		agent.buffer.store_last_value(episode_step + 1, v_n)
		if agent.buffer.episode_num == agent.batch_size:
			agent.train(total_steps)
		total_episodes += 1
	return score_history


def evaluate(agent: Agent, env):
	evaluate_reward = 0
	total_steps = 0
	for _ in range(5):
		episode_reward = 0
		obs_n, _, s = env.reset()
		for _ in range(50):
			a_n, a_logprob_n = agent.choose_action(obs_n, evaluate=evaluate)
			obs_next_n, r_n, done_n, _, _, s_next = env.step(a_n)
			episode_reward += sum(r_n)
			obs_n = np.array(obs_next_n)
			s = s_next
			total_steps += 1
			if all(done_n):
				break
		evaluate_reward += episode_reward

	evaluate_reward /= 5
	print(f'evaluate_reward:{evaluate_reward}')


def run_experiment(n_games, exp_dir, exp_name, n_agents, n_prey, grid_size, obs_rad):
	env = CatMouseDiscrete(n_agents=n_agents, n_prey=n_prey, ma=True, grid_size=grid_size, observation_radius=obs_rad)
	agent = Agent(
		env_name='lumberjacks',
		action_dim=env.action_dim,
		obs_dim=env.obs_dim,
		state_dim=env.state_dim,
		lr=0.0003,
		gamma=0.99,
		n_epochs=4,
		batch_size=64,
		continuous=False,
		n_agents=env.n_agents,
		episode_limit=50,
		max_train_steps=10000000,
		mini_batch_size=8
	)
	score_history = train(agent, env, n_games=n_games)
	score_df = pd.DataFrame(score_history ,columns=["score"])
	score_df.to_csv(f"{exp_dir}/scores_{exp_name}.csv")
	agent.save_models(id=f"{exp_name}")


def run_experiments(exp_dir, n_games = 40000, n_runs = 3, single_proc = False):
	exp_names_list = [f"num_agent_exp_{i}" for i in range(2, 5)]  + [f"env_comp_exp_{i}" for i in range(3)] #+ [f"comm_rad_exp_{i}" for i in [-1, 1, 2]]
	n_agents_list = [2, 3, 4]  + [2, 2, 2] #+ [2, 2, 2]
	n_prey_list = [6, 6, 6]  + [6, 10, 14] # + [8, 8, 8]
	grid_sizes_list = [4, 4, 4]  + [4, 6, 8] #+ [5, 5, 5]
	obs_radius_list = [1, 1, 1]  + [1, 1, 1] #+ [1, 1, 2]
	if single_proc:
		for j in range(n_runs):
			for i in range(len(exp_names_list)):
				exp_name = exp_names_list[i]+f"_run_{j}"
				run_experiment(n_games=n_games, exp_dir=exp_dir, exp_name=exp_name, n_agents=n_agents_list[i], n_prey=n_prey_list[i], grid_size= grid_sizes_list[i], obs_rad=obs_radius_list[i])



def init_dir(dir_name):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

if __name__ == '__main__':
	model_dir = "checkpoints/"
	exp_out_dir = "exp_outputs/"
	init_dir(model_dir)
	init_dir(exp_out_dir)
	n_games = 30000
	n_runs = 1
	single_proc = True
	run_experiments(exp_out_dir, n_games=n_games, n_runs=n_runs, single_proc=single_proc)



# if __name__ == '__main__':
# 	# self.env = SimpleSpreadV3()
# 	eval = True
# 	n_games = 10800
# 	# env = CatMouse(evaluate=evaluate)
# 	# env = SimpleSpreadV3(evaluate=evaluate)
# 	# env = Lumberjacks(n_agents=4, n_trees=16, grid_size=8, evaluate=eval)	
# 	# env = Lumberjacks(n_agents=2, n_trees=8, grid_size=5, evaluate=eval)

# 	env = CatMouseDiscrete(n_agents=2, n_prey=8, ma=True, grid_size=5, observation_radius=1, evaluate=eval)
# 	# agent = Agent(
# 	# 	env_name='lumberjacks', continuous=False,
# 	# 	n_agents=env.n_agents, obs_dim=env.obs_dim, action_dim=env.action_dim, state_dim=env.state_dim,
# 	# 	episode_limit=50, batch_size=64, mini_batch_size=8,
# 	# 	max_train_steps=1000000
# 	# )
# 	agent = Agent(
# 		env_name='catmouse',
# 		action_dim=env.action_dim,
# 		obs_dim=env.obs_dim,
# 		state_dim=env.state_dim,
# 		gamma=0.99,
# 		n_epochs=4,
# 		batch_size=64,
# 		continuous=False,
# 		n_agents=env.n_agents,
# 		episode_limit=50,
# 		max_train_steps=n_games * 50,
# 		mini_batch_size=8
# 	)
# 	if eval:
# 		agent.load_models()
# 		evaluate(agent, env)
# 	else:
# 		train(agent, env, n_games=n_games)
# 		agent.save_models()