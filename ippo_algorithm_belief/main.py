import time
from typing import List
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pettingzoo.mpe import simple_spread_v3
from agent import Agent
from lumberjack_state_distribution import Lumberjacks_State_Distribution


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

def get_local_observations_lumber(global_observation, n_agents, n_trees, rad = 1):
	local_obs = []
	comm = []
	for i in range(n_agents):
		comm.append([])
		agent_pos = global_observation[2*i+1:2*(i+1)+1]
		loc_obs = np.copy(global_observation)
		for j in range(n_agents):
			if abs(global_observation[2*j+1]-agent_pos[0]) > rad or abs(global_observation[2*j+1+1]-agent_pos[1]) > rad:
				loc_obs[2*j+1] = -1
				loc_obs[2*j+1+1] = -1
			else:
				comm[i].append(j)
		for j in range(n_trees):
			if abs(global_observation[2*n_agents+3*j+1]-agent_pos[0]) > rad or abs(global_observation[2*n_agents+3*j+1+1]-agent_pos[1]) > rad:
				loc_obs[2*n_agents+3*j+1] = -1
				loc_obs[2*n_agents+3*j+1+1] = -1
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
	def __init__(self, n_agents, n_trees = 8, grid_size = 5, evaluate=False):
		self.n_trees = n_trees
		self.grid_size = grid_size
		self.env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees) #n_trees=8,
		self.get_env_step = lambda env: env.get_agent_obs()[0][3]
		self.get_local_obs = lambda env: get_local_observations_lumber(np.append(np.array([self.get_env_step(env)]), state_to_array_lumber(env.get_global_obs())), n_agents, n_trees)
		self.state_dim = (2*n_agents+1)*grid_size**2+1#[self.env.observation_space[agent].shape[0] for agent in range(self.env.n_agents)][0] # 75
		self.state_dim = 53 # (2*n_agents+1)*(2*self.grid_size-1)**2+1
		self.obs_dim = (2*n_agents+1)*grid_size**2+1#self.env.observation_space[0].shape[0]
		self.action_dim = 5
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate
		self.ACTION_SPACE = generate_action_space(5, self.n_agents, [[]])
		self.belief_distr = []
		for i in range(self.n_agents):
			self.belief_distr.append(Lumberjacks_State_Distribution(n_agents, n_trees, grid_size, i))

	def reset(self):
		obs_n = self.env.reset()
		obs_n = np.array(obs_n)
		obs_n, comm_n = self.get_local_obs(self.env)
		belief_obs = []
		for i in range(self.n_agents):
			self.belief_distr[i].reset()
			self.belief_distr[i].update_estimation_local_observation(np.array(obs_n[i]))
			belief_obs.append(np.concatenate((np.array([i]), obs_n[i][1:3], self.belief_distr[i].get_belief_state())))
		# obs_n = np.array([self.get_global_obs(), self.get_global_obs()])
		return np.array(belief_obs), None, np.array(belief_obs)#.flatten()

	def step(self, a_n):
		obs_next_n, r_n, done_n, info = self.env.step(a_n)
		obs_next_n = np.array(obs_next_n)
		obs_next_n, comm_next_n = self.get_local_obs(self.env)
		belief_obs = []
		for i in range(self.n_agents):
			self.belief_distr[i].update_estimation_local_observation(np.array(obs_next_n[i]))
		if comm_next_n[0] == 2:
			dist = Lumberjacks_State_Distribution.update_estimation_communication(self.belief_distr)
			for i in range(self.n_agents):
				self.belief_distr[i] = dist
		for i in range(self.n_agents):
			belief_obs.append(np.concatenate((np.array([i]), obs_next_n[i][1:3], self.belief_distr[i].get_belief_state())))
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		# obs_next_n = np.array([self.get_global_obs(), self.get_global_obs()])
		obs_next_n = self.get_local_obs(self.env)
		return np.array(belief_obs), r_n, done_n, None, info, np.array(belief_obs)#.flatten()

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


def train(agents: List[Agent], env, n_games=10000, best_score=-100, learning_step=128):
	episode_history = []
	score_history = []

	learn_iters = 0
	avg_score = 0
	n_steps = 0

	print_interval = 100
	for episode in range(n_games):
		dones = [False]
		state, _, _ = env.reset()
		score = 0
		steps = 0
		while not all(dones) and steps < 50:
			actions, probs, vals, dones = [], [], [], []
			temp = [0, 0, 0, 0, 0]
			prev_action = 0
			for i, agent in enumerate(agents):
				action, prob, val = agent.choose_action(state[i])
				actions.append(action)
				probs.append(prob)
				vals.append(val)
			state_, reward, dones, _, _, _ = env.step(actions)
			n_steps += 1
			score += sum(reward)
			for i, agent in enumerate(agents):
				agent.remember(state[i], actions[i], probs[i], vals[i], reward[i], dones[i])
			if n_steps % learning_step == 0:
				for agent in agents:
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
		if episode % print_interval == 0:
			print(f'episode: {episode} | avg score: {avg_score:.1f} | learning_steps: {learn_iters}')
	return score_history


def evaluate(agents: List[Agent], env):
	state, _, _ = env.reset()
	dones = [False]
	rewards = 0
	while not all(dones):
		actions = []
		temp = [0, 0, 0, 0, 0]
		prev_action = 0
		for i, agent in enumerate(agents):
			# if i == 0:
			# 	action, prob, val = agent.choose_action(state[i])
			# 	prev_action = action
			# if i == 1:
			# 	temp[prev_action] = 1
			# 	temp = np.concatenate([state[i], np.array(temp)])
			# 	action, prob, val = agent.choose_action(temp)
			action, prob, val = agent.choose_action(state[i])
			actions.append(action)
		state_, reward, dones, _, _, _ = env.step(actions)
		rewards += sum(reward)
		env.render()
		# print(state_[i])
		time.sleep(0.1)
		state = state_
	print(rewards)


if __name__ == '__main__':
	# env = gym.make('CartPole-v0')
	# env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(5, 5), n_agents=2)
	n_agents = 2
	eval = True
	# env = CatMouse(evaluate=eval)
	env = Lumberjacks(n_agents, evaluate=eval)
	# env = SimpleSpreadV3(evaluate=eval)
	agents = []
	for i in range(n_agents):
		agents.append(Agent(
			env_name='lumberjacks',
			n_actions=env.action_dim,
			input_dims=env.state_dim, #  + (5 if i == 1 else 0)
			alpha= 0.0001,
			gamma=0.99,
			n_epochs=4,
			batch_size=128
		))
	if eval:
		for i, agent in enumerate(agents):
			agent.load_models(id=i)
		for i in range(10):
			evaluate(agents, env)
	else:
		for i, agent in enumerate(agents):
			agent.load_models(id=i)
		score_history = train(agents, env, n_games=25000)
		plot_learning_curve('lumberjacks', [i for i in range(len(score_history))], score_history)
		for i, agent in enumerate(agents):
			agent.save_models(id=i)
	