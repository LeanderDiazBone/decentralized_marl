import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from agent import Agent
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_discrete import CatMouseMAD
from multiprocessing import Process


def generate_action_space(n_actions, n_agents, l):
	if n_agents == 0:
		return l
	len_l = len(l)
	for li in range(len_l):
		temp = l.pop(0)
		for i in range(n_actions):
			l.append(temp.copy() + [i])
	l = generate_action_space(n_actions, n_agents-1, l)
	return l

class CatMouse:
	@staticmethod
	def get_action(action):
		action_dict = {
			0: 0,
			1: 0.25,
			2: 0.5,
			3: 0.75
		}
		ret = []
		for x in range(4):
			for y in range(4):
				ret.append([action_dict[x], action_dict[y]])
		return ret[action]

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
		self.action_dim = 16
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n = np.array(self.trans_obs(obs_n))
		return obs_n, info, self.trans_state(self.env.get_global_obs())

	def step(self, a_n):
		obs_next_n, r_n, done_n, trunc, info = self.env.step(self.get_action(a_n))
		obs_next_n = self.trans_obs(obs_next_n)
		r_n = sum(r_n)
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
		obs_next_n, r_n, done_n, trunc, info = self.env.step(self.get_action_discrete(a_n))
		obs_next_n = self.trans_obs_discrete(obs_next_n)
		r_n = sum(r_n)
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, done_n, trunc, info, self.trans_state_discrete(self.env.get_global_obs())

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


def get_local_observations_lumber(global_observation, n_agents, n_trees, obs_rad = 1, com_rad = 1):
	local_obs = []
	comm = []
	for i in range(n_agents):
		comm.append([])
		offset = 1
		agent_pos = global_observation[2*i+offset:2*(i+1)+offset]
		loc_obs = np.copy(global_observation)
		for j in range(n_agents):
			if abs(global_observation[2*j+offset]-agent_pos[0]) > obs_rad or abs(global_observation[2*j+1+offset]-agent_pos[1]) > obs_rad:
				loc_obs[2*j+offset] = -1
				loc_obs[2*j+1+offset] = -1
			elif abs(global_observation[2*j+offset]-agent_pos[0]) <= com_rad and abs(global_observation[2*j+1+offset]-agent_pos[1]) <= com_rad:
				comm[i].append(j)
		offset = 2*n_agents+1
		for j in range(n_trees):
			if abs(global_observation[offset+3*j]-agent_pos[0]) > obs_rad or abs(global_observation[offset+3*j+1]-agent_pos[1]) > obs_rad:
				loc_obs[offset+3*j] = -1
				loc_obs[offset+3*j+1] = -1
		local_obs.append(loc_obs)
	return local_obs, comm

def state_to_array_lumber(state):
	state_list = []
	for agent in state[0]:
		state_list.append(agent[0]-1)
		state_list.append(agent[1]-1)
	for tree in state[1]:
		state_list.append(tree[0][0]-1)
		state_list.append(tree[0][1]-1)
		state_list.append(tree[1])
	return np.array(state_list)


class Lumberjacks:
	def __init__(self, evaluate=False, grid_size=5, n_agents=2, n_trees=8, observation_rad=1):
		self.env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees)
		self.grid_size = grid_size
		self.state_dim = (grid_size ** 2) * 2
		self.obs_dim = self.env.observation_space[1].shape[0]
		self.action_dim = 5 ** self.env.n_agents
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate
		self.ACTION_SPACE = generate_action_space(5, self.n_agents, [[]])

	def reset(self):
		obs_n = self.env.reset()
		obs_n = np.array(obs_n)
		return obs_n, None, self.get_global_obs()

	def step(self, a_n):
		obs_next_n, r_n, done_n, info = self.env.step(self.ACTION_SPACE[a_n])
		obs_next_n = np.array(obs_next_n)
		done_n = all(done_n)
		r_n = sum(r_n)
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, done_n, None, info, self.get_global_obs()

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


def plot_learning_curve(name, episode_history, score_history):
	plt.figure(figsize=(10, 5))
	episode_history, score_history = episode_history[::200], score_history[::200]
	plt.plot(episode_history, score_history)
	plt.xlabel('Episodes')
	plt.ylabel('Reward')
	plt.title('Reward vs Episodes')
	plt.grid(True)
	plt.savefig(f'{name}.png')
	data = {'Episodes': episode_history, 'Reward': score_history}
	df = pd.DataFrame(data)
	df.to_csv(f'{name}.csv', index=False)


def train(agent: Agent, env, n_games=10000, best_score=-100, learning_step=128):
	episode_history = []
	score_history = []

	learn_iters = 0
	avg_score = 0
	n_steps = 0

	print_interval = 100

	for i in range(n_games):
		done = False
		_, _, state = env.reset()
		score = 0
		steps = 0
		while not done and steps < 50:
			action, prob, val = agent.choose_action(state)
			_, reward, done, _, _, state_ = env.step(action)
			n_steps += 1
			score += reward
			agent.remember(state, action, prob, val, reward, done)
			if n_steps % learning_step == 0:
				agent.learn()
				learn_iters += 1
			state = state_
			steps += 1
		score_history.append(score)
		episode_history.append(n_steps)
		avg_score = np.mean(score_history[-100:])
		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()
		if i % print_interval == 0:
			print(f'episode: {i} | avg score: {avg_score:.1f} | learning_steps: {learn_iters}')
	return score_history


def evaluate(agent: Agent, env):
	_, _, observation = env.reset()
	done = False
	while not done:
		action, prob, val = agent.choose_action(observation)
		_, reward, done, _, _, observation_ = env.step(action)
		env.render()
		time.sleep(0.01)
		observation = observation_


def run_experiment_lumberjack(n_games, exp_dir, exp_name, n_agents, n_trees, grid_size, obs_rad, comm_rad):
	env = Lumberjacks(n_agents = n_agents, n_trees = n_trees, grid_size = grid_size, observation_rad=obs_rad)
	agent = Agent(
		env_name='lumberjacks',
		n_actions=env.action_dim,
		input_dims=env.state_dim,
		alpha= 0.0003,
		gamma=0.99,
		n_epochs=4,
		batch_size=128,
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
			for i in range(6, len(exp_names_list)):
				exp_name = exp_names_list[i]+f"_run_{j}"
				print(exp_name)
				run_experiment_lumberjack(n_games = n_games, exp_dir=exp_dir, exp_name=exp_name, n_agents=n_agents_list[i], n_trees=n_trees_list[i], grid_size= grid_sizes_list[i], obs_rad=obs_radius_list[i], comm_rad = comm_radius_list[i])
	else:
		processes = []
		for j in range(n_runs):
			for i in range(len(exp_names_list)):
				exp_name = exp_names_list[i]+f"run_{j}"
				try:
					p = Process(target=run_experiment_lumberjack, args=(n_games, exp_dir, exp_name, n_agents_list[i], n_trees_list[i],  grid_sizes_list[i], obs_radius_list[i], comm_radius_list[i]))
					processes.append(p)
					p.start()
				except Exception: 
					print(f"{exp_name} failed.")
		for p in processes:
			p.join()

"""
if __name__ == '__main__':
	# env = gym.make('CartPole-v0')
	# env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(5, 5), n_agents=2)
	learning_step = 128
	eval = False
	# env = CatMouseDiscrete(evaluate=eval)
	env = Lumberjacks(evaluate=eval)
	# env = SimpleSpreadV3(evaluate=eval)
	agent = Agent(
		env_name='lumberjacks',
		n_actions=env.action_dim,
		input_dims=env.state_dim,
		alpha= 0.0003,
		gamma=0.99,
		n_epochs=4,
		batch_size=64,
	)
	if eval:
		agent.load_models()
		for i in range(10):
			evaluate(agent, env)
	else:
		agent.load_models()
		train(agent, env)
		agent.save_models()"""


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
	run_experiments_lumberjack(exp_out_dir, n_games=n_games, n_runs=n_runs, single_proc=single_proc)