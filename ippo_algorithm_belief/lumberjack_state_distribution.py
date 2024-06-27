import sys
import torch
import time
import numpy as np
from torch.distributions import Normal, Bernoulli, MultivariateNormal

class Lumberjacks_State_Distribution(torch.distributions.distribution.Distribution):

    def __init__(self, n_agents, n_trees, grid_size, agent_id, belief_radius = 2, obs_rad = 1):
        super().__init__()
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_trees = n_trees
        self.grid_size = grid_size
        self.belief_radius = belief_radius
        self.obs_rad = obs_rad
        self.observed_fields = np.zeros((grid_size, grid_size))
        self.agent_pos_distribution = np.ones((n_agents, grid_size, grid_size))/((grid_size**2)*n_agents)
        self.observed_trees = np.zeros((n_agents+1, grid_size, grid_size))
        
    
    def reset(self):
        self.observed_fields = np.zeros((self.grid_size, self.grid_size))
        self.agent_pos_distribution = np.ones((self.n_agents, self.grid_size, self.grid_size))/((self.grid_size**2)*self.n_agents)
        self.observed_trees = np.zeros((self.n_agents+1, self.grid_size, self.grid_size))

    def get_view(self, agent_pos):
        view = np.ones((self.grid_size, self.grid_size))
        its = [i for i in range(self.obs_rad+1)]+[-i for i in range(1, self.obs_rad+1)]
        its.sort()
        for i in its:
            for j in its:
                if agent_pos[0]+i >= 0 and agent_pos[0]+i < self.grid_size and agent_pos[1]+j >= 0 and agent_pos[1]+j < self.grid_size:
                    view[int(agent_pos[0])+i][int(agent_pos[1])+j] = 0
        return view

    def update_estimation_local_observation(self, loc_obs):
        agents = loc_obs[1:self.n_agents*2+1].astype('int')
        trees = loc_obs[self.n_agents*2+1:].astype('int')
        view = self.get_view(agents[2*self.agent_id:2*(self.agent_id+1)])
        self.observed_fields = self.observed_fields+((-1)*view+1)
        self.observed_fields[self.observed_fields.nonzero()] = np.ones((self.grid_size, self.grid_size))[self.observed_fields.nonzero()]
        for i in range(self.n_agents):
            agent = agents[2*i:2*(i+1)]
            if agent[0] != -1:
                distr = np.zeros((self.grid_size, self.grid_size))
                distr[agent[0]][agent[1]] = 1
                self.agent_pos_distribution[i] = distr
            else:
                if i == self.agent_id:
                    print("error")
                distr = self.agent_pos_distribution[i]+np.ones((self.grid_size, self.grid_size))*0.1
                distr = distr*view
                distr /= np.sum(distr)
                self.agent_pos_distribution[i] = distr
        for i in range(self.n_trees):
            tree = trees[3*i:3*(i+1)]
            if tree[0] != -1:
                self.observed_trees[tree[2]][tree[0]][tree[1]] = 1
                if tree[2] == 0:
                    for i in range(self.n_agents):
                        self.observed_trees[i+1][tree[0]][tree[1]] = 0

    def get_belief_state(self):
        n_seen_trees = np.sum(self.observed_trees)
        number_left_fields = self.grid_size**2-np.sum(self.observed_fields)
        if number_left_fields == 0:
            prob = 0
        else:
            prob = (self.n_trees-n_seen_trees)/(number_left_fields*self.n_agents)
        tree_state = np.concatenate((np.zeros((1, self.grid_size, self.grid_size)), np.ones((self.n_agents, self.grid_size, self.grid_size))*prob), axis=0)
        for i in range(self.n_agents+1):
            tree_state[i] = tree_state[i]*(1-self.observed_fields)
            tree_state[i] += self.observed_trees[i]
        tree_state_centralized = np.zeros((self.n_agents+1, 2*self.grid_size-1, 2*self.grid_size-1))
        agent_state_centralized = np.zeros((self.n_agents, 2*self.grid_size-1, 2*self.grid_size-1))
        ag_pos = np.where(self.agent_pos_distribution[self.agent_id] == 1)
        ag_pos = [ag_pos[0][0], ag_pos[1][0]]
        for i in range(self.n_agents+1):
            x = np.array(list(range(self.grid_size-ag_pos[0]-1,2*self.grid_size-ag_pos[0]-1, 1))) # sagt an der stelle wohin er kopiert wird.
            y = np.array(list(range(self.grid_size-ag_pos[1]-1,2*self.grid_size-ag_pos[1]-1, 1)))
            tree_state_centralized[i][np.ix_(x,y)] = tree_state[i]
            if i < self.n_agents:
                agent_state_centralized[i][np.ix_(x,y)] = self.agent_pos_distribution[i]
        
        start = self.grid_size-1-self.belief_radius
        end = self.grid_size+self.belief_radius
        state_agents = np.delete(agent_state_centralized, self.agent_id, axis=0)[:, start:end, start:end].flatten()
        new_tree_state = tree_state_centralized[1:,start:end, start:end].flatten()
        return np.append(state_agents, new_tree_state)

    @staticmethod
    def update_estimation_communication(distributions):
        grid_size = distributions[0].grid_size
        n_agents = distributions[0].n_agents
        n_trees = distributions[0].n_trees
        belief_radius = distributions[0].belief_radius
        obs_rad = distributions[0].obs_rad
        final_distr = Lumberjacks_State_Distribution(n_trees=n_trees, n_agents=n_agents, grid_size=grid_size, agent_id=-1, belief_radius=belief_radius, obs_rad=obs_rad)
        total_observed = np.zeros((grid_size, grid_size))
        total_observed_trees = np.zeros((n_agents+1, grid_size, grid_size))
        total_agent_pos = np.ones((n_agents, grid_size, grid_size))
        for distr in distributions:
            total_observed += distr.observed_fields
            total_observed_trees += distr.observed_trees
            total_agent_pos = distr.agent_pos_distribution

        for i in range(n_agents):
            for distr in distributions:
                if 1 in distr.agent_pos_distribution[i]:
                    total_agent_pos[i] = distr.agent_pos_distribution[i]
                    
        for i in range(n_agents):
            total_agent_pos[i] = total_agent_pos[i]/np.sum(total_agent_pos[i])

        total_observed[total_observed.nonzero()] = np.ones((grid_size, grid_size))[total_observed.nonzero()]
        total_observed_trees[total_observed_trees.nonzero()] = np.ones((n_agents+1, grid_size, grid_size))[total_observed_trees.nonzero()]
        final_distr.observed_fields = total_observed
        final_distr.observed_trees = total_observed_trees
        final_distr.agent_pos_distribution = total_agent_pos
        return final_distr