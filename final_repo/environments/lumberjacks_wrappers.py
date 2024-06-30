import time
import os
from typing import List
import gym
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("../")
from algorithms.lumberjack_state_distribution import Lumberjacks_State_Distribution

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


class Lumberjacks_Wrapper_Belief:
	def __init__(self, n_agents = 2, n_trees = 8, grid_size = 5, belief_radius=2, observation_rad = 1, communication_rad = 1, evaluate=False):
		self.n_trees = n_trees
		self.grid_size = grid_size
		self.env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees)
		self.get_env_step = lambda env: env.get_agent_obs()[0][3]
		self.get_local_obs = lambda env: get_local_observations_lumber(np.append(np.array([self.get_env_step(env)]), state_to_array_lumber(env.get_global_obs())), n_agents, n_trees, obs_rad = observation_rad, com_rad = communication_rad)
		self.state_dim = (2*n_agents-1)*(2*belief_radius+1)**2+3
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


class Lumberjacks_Wrapper_Local:
	def __init__(self, evaluate=False, grid_size=5, n_agents=2, n_trees=8, observation_rad=1, communication_rad=1):
		self.env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees)
		self.state_dim = np.sum([self.env.observation_space[agent].shape[0] for agent in range(self.env.n_agents)])
		self.get_env_step = lambda env: env.get_agent_obs()[0][3]
		self.get_local_obs = lambda env: get_local_observations_lumber(np.append(np.array([self.get_env_step(env)]), state_to_array_lumber(env.get_global_obs())), n_agents, n_trees, obs_rad=observation_rad, com_rad=communication_rad)
		self.obs_dim = self.env.observation_space[1].shape[0]
		self.action_dim = 5 * self.env.n_agents
		self.n_actions_per_agent = 5
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate
		self.ACTION_SPACE = generate_action_space(5, self.n_agents, [[]])

	def reset(self):
		obs_n = self.env.reset()
		obs_n = np.array(obs_n)
		return obs_n, None, self.state_to_array_lumber_2(self.env.get_global_obs(), self.n_agents, 5) # obs_n.flatten()

	def step(self, a_n):
		obs_next_n, r_n, done_n, info = self.env.step(a_n)
		obs_next_n = np.array(obs_next_n)
		done_n = all(done_n)
		r_n = sum(r_n)
		if self.evaluate:
			time.sleep(0.1)
			self.env.render()
		return obs_next_n, r_n, done_n, None, info, self.state_to_array_lumber_2(self.env.get_global_obs(), self.n_agents, 5) # obs_next_n.flatten()

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()