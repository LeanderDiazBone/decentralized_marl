import time
import os
from typing import List
import gym
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from pettingzoo.mpe import simple_spread_v3
from agent import Agent
from final_repo.algorithms.lumberjack_state_distribution import Lumberjacks_State_Distribution
from multiprocessing import Process
from multiprocessing import Pool

def init_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


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

def communication_bubbles(comm):
    all_agents = set(np.linspace(0, len(comm)-1, len(comm)).astype(int))
    bubbles = []
    for i in range(len(comm)):
        if i in all_agents:
            all_agents.remove(i)
            new_bubble = set([])
            queue = [i]
            while len(queue) > 0:
                el = queue.pop(0)
                new_bubble.add(el)
                for sec_el in comm[el]:
                    if sec_el in all_agents:
                        queue.append(sec_el)
                        all_agents.remove(sec_el)
            bubbles.append(list(new_bubble))
    return bubbles

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
	def __init__(self, n_agents = 2, n_trees = 8, grid_size = 5, belief_radius=2, observation_rad = 1, communication_rad = 1, evaluate=False):
		self.n_trees = n_trees
		self.grid_size = grid_size
		self.env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees) #n_trees=8,
		self.get_env_step = lambda env: env.get_agent_obs()[0][3]
		self.get_local_obs = lambda env: get_local_observations_lumber(np.append(np.array([self.get_env_step(env)]), state_to_array_lumber(env.get_global_obs())), n_agents, n_trees, obs_rad = observation_rad, com_rad = communication_rad)
		#self.state_dim = (2*n_agents+1)*grid_size**2+1#[self.env.observation_space[agent].shape[0] for agent in range(self.env.n_agents)][0] # 75
		#self.state_dim = 53 # (2*n_agents+1)*(2*self.grid_size-1)**2+1
		self.state_dim = (2*n_agents-1)*(2*belief_radius+1)**2+3
		#self.obs_dim = (2*n_agents+1)*grid_size**2+1#self.env.observation_space[0].shape[0]
		self.action_dim = 5
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate
		self.ACTION_SPACE = generate_action_space(5, self.n_agents, [[]])
		self.belief_distr = []
		for i in range(self.n_agents):
			self.belief_distr.append(Lumberjacks_State_Distribution(n_agents, n_trees, grid_size, i, belief_radius=belief_radius, obs_rad=observation_rad))

	def reset(self):
		obs_n = self.env.reset()
		obs_n = np.array(obs_n)
		obs_n, comm_n = self.get_local_obs(self.env)
		belief_obs = []
		for i in range(self.n_agents):
			self.belief_distr[i].reset()
			self.belief_distr[i].update_estimation_local_observation(np.array(obs_n[i]))
			belief_obs.append(np.concatenate((np.array([i]), obs_n[i][1:3], self.belief_distr[i].get_belief_state())))
		return np.array(belief_obs), None, np.array(belief_obs)

	def step(self, a_n):
		obs_next_n, r_n, done_n, info = self.env.step(a_n)
		obs_next_n = np.array(obs_next_n)
		obs_next_n, comm_next_n = self.get_local_obs(self.env)
		belief_obs = []
		for i in range(self.n_agents):
			self.belief_distr[i].update_estimation_local_observation(np.array(obs_next_n[i]))
		bubbles = communication_bubbles(comm_next_n)
		for bubble in bubbles:
			if len(bubble) > 1:
				final_distr = Lumberjacks_State_Distribution.update_estimation_communication([self.belief_distr[i] for i in bubble])
				for i in bubble:
					agent_distr = copy.deepcopy(final_distr)
					agent_distr.agent_id = i
					self.belief_distr[i] = agent_distr

		for i in range(self.n_agents):
			belief_obs.append(np.concatenate((np.array([i]), obs_next_n[i][1:3], self.belief_distr[i].get_belief_state())))
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
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

	print_interval = 250
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
	while not all(dones):
		actions = []
		temp = [0, 0, 0, 0, 0]
		prev_action = 0
		for i, agent in enumerate(agents):
			action, prob, val = agent.choose_action(state[i])
			actions.append(action)
		state_, reward, dones, _, _, _ = env.step(actions)
		env.render()
		print(state_[i])
		time.sleep(20)
		state = state_


def run_experiment_lumberjack(n_games, exp_dir, exp_name, n_agents, n_trees, grid_size, belief_radius, obs_rad, comm_rad):
	env = Lumberjacks(n_agents = n_agents, n_trees = n_trees, grid_size = grid_size, belief_radius=belief_radius, observation_rad = obs_rad, communication_rad = comm_rad)
	agents = []
	for i in range(n_agents):
		agents.append(Agent(env_name='lumberjacks', n_actions=env.action_dim, input_dims=env.state_dim, alpha= 0.0001, gamma=0.99, n_epochs=4, batch_size=128))
	score_history = train(agents, env, n_games=n_games)
	score_df = pd.DataFrame(score_history ,columns=["score"])
	score_df.to_csv(f"{exp_dir}/scores_{exp_name}.csv")
	for i, agent in enumerate(agents):
		agent.save_models(id=f"{i}{exp_name}")

def run_experiments_lumberjack(exp_dir, n_games = 40000, n_runs = 3, single_proc = False):
	exp_names_list = [f"num_agent_exp_{i}" for i in range(2, 5)] + [f"comm_rad_exp_{i}" for i in [-1, 1, 2]] + [f"env_comp_exp_{i}" for i in range(3)]
	n_agents_list = [2, 3, 4] + [2, 2, 2] + [2, 2, 2]
	n_trees_list = [6, 6, 6] + [8, 8, 8] + [6, 10, 14]
	grid_sizes_list = [4, 4, 4] + [5, 5, 5] + [4, 6, 8]
	belief_radius_list = [2, 2, 2] + [2, 2, 2] + [2, 2, 2]
	obs_radius_list = [1, 1, 1] + [1, 1, 2] + [1, 1, 1]
	comm_radius_list = [1, 1, 1] + [-1, 1, -2] + [1, 1, 1]
	start = time.time()
	if single_proc:
		for j in range(run_offset, n_runs+run_offset):
			for i in [0, 1, 2, 3, 5, 8]:
			#for i in range(4, len(exp_names_list)):
				exp_name = exp_names_list[i]+f"_run_{j}"
				print(exp_name)
				print("Time in Minutes:")
				print((time.time()-start)/60)
				run_experiment_lumberjack(n_games = n_games, exp_dir=exp_dir, exp_name=exp_name, n_agents=n_agents_list[i], n_trees=n_trees_list[i], grid_size= grid_sizes_list[i], belief_radius=belief_radius_list[i], obs_rad=obs_radius_list[i], comm_rad = comm_radius_list[i])
	else:
		print("Parallel Execution")
		args = []
		for j in range(n_runs):
			for i in range(len(exp_names_list)):
				exp_name = exp_names_list[i]+f"_run_{j}"
				args.append((n_games, exp_dir, exp_name, n_agents_list[i], n_trees_list[i], grid_sizes_list[i], belief_radius_list[i], obs_radius_list[i], comm_radius_list[i]))
		pool = Pool(processes = 9)
		pool.starmap(run_experiment_lumberjack, args, chunksize = 1)



if __name__ == '__main__':
	model_dir = "checkpoints/"
	exp_out_dir = "exp_outputs/"
	init_dir(model_dir)
	init_dir(exp_out_dir)
	n_games = 30000
	n_runs = 1
	single_proc = True
	run_offset = 0
	run_experiments_lumberjack(exp_out_dir, n_games=n_games, n_runs=n_runs, single_proc=single_proc, run_offset = run_offset)
	"""
	# env = gym.make('CartPole-v0')
	# env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(5, 5), n_agents=2)
	
	n_agents = 2
	eval = False
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
		#for i, agent in enumerate(agents):
		#	agent.load_models(id=i)
		for i in range(10):
			evaluate(agents, env)
	else:
		# for i, agent in enumerate(agents):
		# 	agent.load_models(id=i)
		score_history = train(agents, env, n_games=20000)
		plot_learning_curve('lumberjacks', [i for i in range(len(score_history))], score_history)
		for i, agent in enumerate(agents):
			agent.save_models(id=i)"""
	