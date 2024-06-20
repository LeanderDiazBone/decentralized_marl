import copy
import time
import multiprocessing as mp
from typing import List
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pettingzoo.mpe import simple_spread_v3
from agent import Agent
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_discrete import CatMouseMAD


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


class SimpleSpreadV3:
	def __init__(self, evaluate=False):
		N = 2
		self.env = simple_spread_v3.parallel_env(N=N, max_cycles=25, local_ratio=0.5,
			render_mode='human' if evaluate else None, continuous_actions=False)
		self.env.reset(seed=42)
		self.n_agents = self.env.num_agents
		self.obs_dim = [self.env.observation_spaces[agent].shape[0] for agent in self.env.agents][0]
		self.state_dim = self.obs_dim * self.n_agents
		self.action_dim = 5 ** N
		self.evaluate = evaluate
		self.ACTION_SPACE = generate_action_space(5, N, [[]])
		# env.action_dim_n = [env.action_spaces[agent].n for agent in env.agents][0]

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n = np.array([obs_n[agent] for agent in obs_n.keys()])
		return obs_n, info, obs_n.flatten()

	def step(self, a_n):
		a_n = self.ACTION_SPACE[a_n]
		actions = {}
		for i, agent in enumerate(self.env.agents):
			actions[agent] = a_n[i] if len(self.env.agents) != 1 else a_n
		obs_next_n, r_n, done_n, trunc, info = self.env.step(actions)
		obs_next_n = np.array([obs_next_n[agent] for agent in obs_next_n.keys()])
		done_n = np.array([val for val in done_n.values()])
		r_n = list(r_n.values())
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return np.array(obs_next_n), sum(r_n), all(done_n), trunc, info, np.array(obs_next_n).flatten()

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
		print(self.get_action_discrete(a_n))
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


class Lumberjacks:
	def __init__(self, n_agents, evaluate=False):
		self.env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(5, 5), n_agents=n_agents, n_trees=8) #n_trees=8,
		self.state_dim = 75 # [self.env.observation_space[agent].shape[0] for agent in range(self.env.n_agents)][0]
		self.obs_dim = self.env.observation_space[0].shape[0]
		self.action_dim = 5
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate
		self.ACTION_SPACE = generate_action_space(5, self.n_agents, [[]])

	def reset(self):
		obs_n = self.env.reset()
		obs_n = np.array(obs_n)
		obs_n = np.array([self.get_global_obs(), self.get_global_obs()])
		return obs_n, None, obs_n.flatten()

	def step(self, a_n):
		obs_next_n, r_n, done_n, info = self.env.step(a_n)
		obs_next_n = np.array(obs_next_n)
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		obs_next_n = np.array([self.get_global_obs(), self.get_global_obs()])
		return obs_next_n, r_n, done_n, None, info, obs_next_n.flatten()

	def get_global_obs(self):
		global_env = self.env.get_global_obs()
		glob_state = [0 for _ in range(75)]
		tree_pos = global_env[1]
		for pos, strength in tree_pos:
			glob_state[5*(pos[0] - 1) + pos[1]] = strength

		agent_pos = global_env[0]
		for agent in agent_pos:
			glob_state[5*(agent[0] - 1) + agent[1] + 25] = 1
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


def train_single_worker(worker_id: int, agents: List[Agent], env, n_workers: int, n_games=10000):
	score_history = []

	learn_iters = 0
	avg_score = 0
	best_score = -100000

	print_interval = 100
	if worker_id == n_workers - 1:
		n_games = n_games // 100

	for episode in range(n_games):
		start = time.time()
		dones = [False]
		state, _, _ = env.reset()
		score = 0
		steps = 0
		buffer = {
			'state': [[]] * len(agents),
			'action': [[]] * len(agents),
			'probs': [[]] * len(agents),
			'vals': [[]] * len(agents),
			'reward': [[]] * len(agents),
			'done': [[]] * len(agents),
		}
		while not all(dones) and steps < 50:
			actions, probs, vals, dones = [], [], [], []
			for i, agent in enumerate(agents):
				action, prob, val = agent.choose_action(state[i])
				actions.append(action)
				probs.append(prob)
				vals.append(val)
			state_, reward, dones, _, _, _ = env.step(actions)
			score += sum(reward)
			for i, agent in enumerate(agents):
				buffer['state'][i].append(state[i])
				buffer['action'][i].append(actions[i])
				buffer['probs'][i].append(probs[i])
				buffer['vals'][i].append(vals[i])
				buffer['reward'][i].append(reward[i])
				buffer['done'][i].append(dones[i])
			state = state_
			steps += 1
		for i, agent in enumerate(agents):
			buffer['state'][i].append(state[i])
			buffer['action'][i].append(actions[i])
			buffer['probs'][i].append(probs[i])
			buffer['vals'][i].append(vals[i])
			buffer['reward'][i].append(reward[i])
			buffer['done'][i].append(dones[i])
		for i, agent in enumerate(agents):
			agent.remember(
				worker_id,
				buffer['state'][i],
				buffer['action'][i],
				buffer['probs'][i],
				buffer['vals'][i],
				buffer['reward'][i],
				buffer['done'][i]
			)
		score_history.append(score)
		end = time.time()
		# if worker_id == 0:
		# 	print(f'Worker [{worker_id}] time per episode: {end-start}')

		if worker_id == n_workers - 1:
			if episode % 2 == 0:
				for agent in agents:
					agent.learn()
				learn_iters += 1

			avg_score = np.mean(score_history[-100:])
			if avg_score > best_score:
				best_score = avg_score
			if episode % (print_interval // n_workers) == 0:
				print(f'episode: {episode} | avg score: {avg_score:.1f} | learning_steps: {learn_iters}')
	plot_learning_curve('lumberjacks', [i for i in range(len(score_history))], score_history)



def train(agents: List[Agent], envs, n_workers=1, n_games=10000, best_score=-100, learning_step=128):
	processes = []
	for i in range(n_workers):
		p = mp.Process(target=train_single_worker, args=(i, agents, envs[i], n_workers))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()

def evaluate(agents: List[Agent], env): 
	observation, _, _ = env.reset()
	dones = [False]
	while not all(dones):
		actions = []
		for i, agent in enumerate(agents):
			action, _, _ = agent.choose_action(observation[i])
			actions.append(action)
		observation_, reward, dones, _, _, _ = env.step(actions)
		env.render()
		time.sleep(0.01)
		observation = observation_


if __name__ == '__main__':
	# env = gym.make('CartPole-v0')
	# env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(5, 5), n_agents=2)
	n_agents = 1
	n_workers = 8
	n_games = 10000
	eval = False
	# env = CatMouse(evaluate=eval)
	# env = SimpleSpreadV3(evaluate=eval)

	envs = [Lumberjacks(n_agents, evaluate=eval) for _ in range(n_workers)]
	agents = []
	for i in range(n_agents):
		agents.append(Agent(
			env_name='lumberjacks',
			n_actions=envs[0].action_dim,
			input_dims=envs[0].state_dim,
			alpha= 0.0003,
			gamma=0.99,
			n_epochs=4,
			batch_size=2048,
			n_workers=n_workers
		))
	if eval:
		for i, agent in enumerate(agents):
			agent.load_models(id=i)
		for i in range(10):
			evaluate(agents, envs[0])
	else:
		train(agents, envs, n_workers=n_workers, n_games=n_games)
		for i, agent in enumerate(agents):
			agent.save_models(id=i)
