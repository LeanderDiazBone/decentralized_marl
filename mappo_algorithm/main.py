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
from pettingzoo.mpe import simple_spread_v3
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_discrete import CatMouseMAD


class SimpleSpreadV3:
	def __init__(self, evaluate=False):
		self.env = simple_spread_v3.parallel_env(N=3, max_cycles=25, local_ratio=0.5,
			render_mode='human' if evaluate else None, continuous_actions=False)
		self.env.reset(seed=42)
		self.n_agents = self.env.num_agents
		self.obs_dim = [self.env.observation_spaces[agent].shape[0] for agent in self.env.agents][0]
		self.state_dim = self.obs_dim * self.n_agents
		self.action_dim = 5
		self.evaluate = evaluate
		# env.action_dim_n = [env.action_spaces[agent].n for agent in env.agents][0]

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n = np.array([obs_n[agent] for agent in obs_n.keys()])
		return obs_n, info, obs_n.flatten()

	def step(self, a_n):
		actions = {}
		for i, agent in enumerate(self.env.agents):
			actions[agent] = a_n[i]
		obs_next_n, r_n, done_n, trunc, info = self.env.step(actions)
		obs_next_n = np.array([obs_next_n[agent] for agent in obs_next_n.keys()])
		done_n = np.array([val for val in done_n.values()])
		r_n = list(r_n.values())
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, done_n, trunc, info, np.array(obs_next_n).flatten()

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


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
			prey_grid = ob["agent_grid"].flatten()
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
		agent_pos = state["agent_pos"].flatten()
		ret.append(agent_grid)
		ret.append(prey_grid)
		ret.append(agent_pos)
		ret = np.concatenate(ret)
		return ret

	def __init__(self, evaluate=False):
		self.env = CatMouseMAD(observation_radius=1, n_agents=2, n_prey=4)
		# self.state_dim = self.env.n_agents * 2 + self.env.n_prey * 3
		# self.obs_dim = self.env.n_agents * 3 + self.env.n_prey * 3
		self.state_dim = 54
		self.obs_dim = 53
		self.action_dim = 9
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n = np.array(self.trans_obs_discrete(obs_n))
		return obs_n, info, self.trans_state_discrete(self.env.get_global_obs())

	def step(self, a_n):
		obs_next_n, r_n, done_n, trunc, info = self.env.step([self.get_action_discrete(a) for a in a_n])
		obs_next_n = self.trans_obs_discrete(obs_next_n)
		done_n = [done_n]
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, done_n, trunc, info, self.trans_state_discrete(self.env.get_global_obs())

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


class Lumberjacks:
	def __init__(self, evaluate=False, grid_size=5, n_agents=2, n_trees=8, observation_rad=1):
		self.env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees) #n_trees=8,
		self.state_dim = (grid_size ** 2 ) * 2 # np.sum([self.env.observation_space[agent].shape[0] for agent in range(self.env.n_agents)])
		self.obs_dim = self.env.observation_space[1].shape[0]
		self.action_dim = 5
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate
		self.grid_size = grid_size

	def reset(self):
		obs_n = self.env.reset()
		obs_n = np.array(obs_n)
		return obs_n, None, self.get_global_obs() # obs_n.flatten()

	def step(self, a_n):
		obs_next_n, r_n, done_n, info = self.env.step(a_n)
		obs_next_n = np.array(obs_next_n)
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, done_n, None, info, self.get_global_obs() #obs_next_n.flatten()

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


def run_experiment_lumberjack(n_games, exp_dir, exp_name, n_agents, n_trees, grid_size):
	env = Lumberjacks(n_agents=n_agents, n_trees=n_trees, grid_size=grid_size)
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


def run_experiments_lumberjack(exp_dir, n_games = 40000, n_runs = 3, single_proc = False):
	exp_names_list = [f"num_agent_exp_{i}" for i in range(2, 5)] + [f"comm_rad_exp_{i}" for i in [-1, 1, 2]] + [f"env_comp_exp_{i}" for i in range(3)]
	n_agents_list = [2, 3, 4] + [2, 2, 2] + [2, 2, 2]
	n_trees_list = [6, 6, 6] + [8, 8, 8] + [6, 10, 14]
	grid_sizes_list = [4, 4, 4] + [5, 5, 5] + [4, 6, 8]
	obs_radius_list = [1, 1, 1] + [1, 1, 2] + [1, 1, 1]
	comm_radius_list = [1, 1, 1] + [-1, 1, -2] + [1, 1, 1]
	if single_proc:
		for j in range(n_runs):
			for i in range(len(exp_names_list)):
				exp_name = exp_names_list[i]+f"_run_{j}"
				run_experiment_lumberjack(n_games = n_games, exp_dir=exp_dir, exp_name=exp_name, n_agents=n_agents_list[i], n_trees=n_trees_list[i], grid_size= grid_sizes_list[i])
	else:
		processes = []
		for j in range(n_runs):
			for i in range(len(exp_names_list)):
				exp_name = exp_names_list[i]+f"run_{j}"
				try:
					p = Process(target=run_experiment_lumberjack, args=(n_games, exp_dir, exp_name, n_agents_list[i], n_trees_list[i],  grid_sizes_list[i]))
					processes.append(p)
					p.start()
				except Exception: 
					print(f"{exp_name} failed.")
		for p in processes:
			p.join()


def init_dir(dir_name):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

if __name__ == '__main__':
	model_dir = "checkpoints/"
	exp_out_dir = "exp_outputs/"
	init_dir(model_dir)
	init_dir(exp_out_dir)
	n_games = 1000
	n_runs = 1
	single_proc = False
	run_experiments_lumberjack(exp_out_dir, n_games=n_games, n_runs=n_runs, single_proc=single_proc)



# if __name__ == '__main__':
# 	# self.env = SimpleSpreadV3()
# 	evaluate = False
# 	# env = CatMouse(evaluate=evaluate)
# 	# env = SimpleSpreadV3(evaluate=evaluate)
# 	env = Lumberjacks(n_agents=2, n_trees=6, grid_size=4)
# 	# agent = Agent(
# 	# 	env_name='lumberjacks', continuous=False,
# 	# 	n_agents=env.n_agents, obs_dim=env.obs_dim, action_dim=env.action_dim, state_dim=env.state_dim,
# 	# 	episode_limit=50, batch_size=64, mini_batch_size=8,
# 	# 	max_train_steps=1000000
# 	# )
# 	agent = Agent(
# 		env_name='lumberjacks',
# 		action_dim=env.action_dim,
# 		obs_dim=env.obs_dim,
# 		state_dim=env.state_dim,
# 		lr=0.0003,
# 		gamma=0.99,
# 		n_epochs=4,
# 		batch_size=64,
# 		continuous=False,
# 		n_agents=env.n_agents,
# 		episode_limit=50,
# 		max_train_steps=10000000,
# 		mini_batch_size=8
# 	)
# 	if evaluate:
# 		agent.load_model()
# 		evaluate(agent, env)
# 	else:
# 		train(agent, env, n_games=11800)
# 		agent.save_model()