import time
import sys
import gym
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from agent import Agent
import torch as T
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import CatMouse
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_discrete import CatMouseMAD
from multiprocessing import Process
sys.path.append("../algorithms/")
from discrete_cat_mouse_state_distribution import Discrete_CatMouse_State_Distribution
import copy
from typing import List


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

def transpose_obs_local(global_obs, n_agents, n_mice):
	for i in range(n_agents):
		x = global_obs[2*i]
		global_obs[2*i] = global_obs[2*i+1]
		global_obs[2*i+1] = x
	offset = 2*n_agents
	for i in range(n_mice):
		x = global_obs[offset+3*i]
		global_obs[offset+3*i] = global_obs[offset+3*i+1]
		global_obs[offset+3*i+1] = x
	return global_obs

def get_local_observations_catmouse(global_observation, n_agents, n_mice, obs_rad = 1, com_rad = 1):
	global_observation = transpose_obs_local(global_observation, n_agents, n_mice)
	local_obs = []
	comm = []
	for i in range(n_agents):
		comm.append([])
		offset = 0
		agent_pos = global_observation[2*i+offset:2*(i+1)+offset]
		loc_obs = np.copy(global_observation)
		for j in range(n_agents):
			if abs(global_observation[2*j+offset]-agent_pos[0]) > obs_rad or abs(global_observation[2*j+1+offset]-agent_pos[1]) > obs_rad:
				loc_obs[2*j+offset] = -1
				loc_obs[2*j+1+offset] = -1
			elif abs(global_observation[2*j+offset]-agent_pos[0]) <= com_rad and abs(global_observation[2*j+1+offset]-agent_pos[1]) <= com_rad:
				comm[i].append(j)
		offset = 2*n_agents
		for j in range(n_mice):
			if abs(global_observation[offset+3*j]-agent_pos[0]) > obs_rad or abs(global_observation[offset+3*j+1]-agent_pos[1]) > obs_rad:
				loc_obs[offset+3*j] = -1
				loc_obs[offset+3*j+1] = -1
		local_obs.append(loc_obs)
	#print(local_obs)
	return local_obs, comm

class CatMouseDiscrete_Wrapper_Belief:
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
	def trans_belief_obs_discrete(obs):
		return np.append(obs["agents"].flatten(), obs["prey"].flatten())

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

	def __init__(self, evaluate=False, n_agents=2, n_prey=4, grid_size=5, observation_radius = 1, communication_radius = 1, ma = True, belief_radius=2):
		self.env = CatMouseMAD(observation_radius=observation_radius, n_agents=n_agents, n_prey=n_prey, grid_size=grid_size)
		self.state_dim = n_agents * (((observation_radius * 2 + 1) ** 2) * 2) + n_agents * 2
		self.state_dim = n_agents * ((belief_radius * 2 + 1) ** 2) + 1
		self.obs_dim = n_agents * ((belief_radius * 2 + 1) ** 2) + 1
		self.n_actions_per_agent = 9
		self.ma = ma
		if self.ma:
			self.action_dim = self.n_actions_per_agent
		else:
			self.action_dim = self.n_actions_per_agent * n_agents
		self.belief_distr = []
		self.n_agents = self.env.n_agents
		self.env.reset()
		self.evaluate = evaluate
		self.get_local_obs = lambda env: get_local_observations_catmouse(self.trans_belief_obs_discrete(env.get_obs_belief()), n_agents=n_agents,n_mice=n_prey,obs_rad=observation_radius, com_rad=communication_radius)
		for i in range(self.n_agents):
			self.belief_distr.append(Discrete_CatMouse_State_Distribution(n_agents, n_mice = n_prey,grid_size= grid_size, agent_id= i, belief_radius=belief_radius, obs_rad=observation_radius))

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n, comm_n = self.get_local_obs(self.env)
		belief_obs = []
		for i in range(self.n_agents):
			self.belief_distr[i].reset()
			self.belief_distr[i].update_estimation_local_observation(np.array(obs_n[i]))
			belief_obs.append(np.concatenate((np.array([i]), self.belief_distr[i].get_belief_state()))) # only agent id, but other removed
		return np.array(belief_obs), info, self.trans_state_discrete(self.env.get_global_obs())

	def step(self, a_n):
		obs_next_n, r_n, done_n, trunc, info = self.env.step(self.get_action_discrete(a_n))
		obs_next_n, comm_next_n = self.get_local_obs(self.env)
		belief_obs = []
		for i in range(self.n_agents):
			self.belief_distr[i].update_estimation_local_observation(np.array(obs_next_n[i]))
		
		bubbles = communication_bubbles(comm_next_n)
		for bubble in bubbles:
			if len(bubble) > 1:
				final_distr = Discrete_CatMouse_State_Distribution.update_estimation_communication([self.belief_distr[i] for i in bubble])
				for i in bubble:
					agent_distr = copy.deepcopy(final_distr)
					agent_distr.agent_id = i
					self.belief_distr[i] = agent_distr
		for i in range(self.n_agents):
			belief_obs.append(np.concatenate((np.array([i]), self.belief_distr[i].get_belief_state())))

		if not self.ma:
			r_n = sum(r_n)

		if self.evaluate:
			self.env.render()
			time.sleep(0.5)
		return np.array(belief_obs), r_n, [done_n for _ in range(self.n_agents)], trunc, info, self.trans_state_discrete(self.env.get_global_obs())

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()

class CatMouseDiscrete_Wrapper_Local:
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