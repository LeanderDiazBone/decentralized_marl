import sys
import torch
import numpy as np
from torch.distributions import Normal, Bernoulli, MultivariateNormal

class Lumberjacks_State_Distribution(torch.distributions.distribution.Distribution):

    def __init__(self, n_agents, n_trees, grid_size, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_trees = n_trees
        self.grid_size = grid_size

        self.observed_fields = np.zeros((grid_size, grid_size))
        self.agent_pos_distribution = np.ones((n_agents, grid_size, grid_size))/(grid_size**2*n_agents)
        self.observed_trees = np.zeros((n_agents+1, grid_size, grid_size))
    
    def reset(self):
        self.observed_fields = np.zeros((self.grid_size, self.grid_size))
        self.agent_pos_distribution = np.ones((self.n_agents, self.grid_size, self.grid_size))/((self.grid_size**2)*self.n_agents)
        self.observed_trees = np.zeros((self.n_agents+1, self.grid_size, self.grid_size))

    def get_view(self, agent_pos):
        view = np.ones((self.grid_size, self.grid_size))
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if agent_pos[0]+i >= 0 and agent_pos[0]+i < self.grid_size and agent_pos[1]+j >= 0 and agent_pos[1]+j < self.grid_size:
                    view[agent_pos[0]+i][agent_pos[1]+j] = 0
        return view

    def update_estimation_local_observation(self, loc_obs):
        agents = loc_obs[:self.n_agents*2]
        trees = loc_obs[self.n_agents*2:]
        view = self.get_view(agents[2*self.agent_id:2*(self.agent_id+1)])
        self.observed_fields = self.observed_fields+1-view
        self.observed_fields[self.observed_fields.nonzero()] = np.ones((self.grid_size, self.grid_size))[self.observed_fields.nonzero()]
        for i in range(self.n_agents):
            agent = agents[2*i:2*(i+1)]
            if agent[0] != -1:
                distr = np.zeros((self.grid_size, self.grid_size))
                distr[agent[0]][agent[1]] = 1
                #if i != self.agent_id: # Workaround
                #    distr += np.zeros((self.grid_size, self.grid_size))*0.01
                self.agent_pos_distribution[i] = distr/np.sum(distr)
            else:
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
        state_agents = self.agent_pos_distribution.flatten()
        state_trees = tree_state.flatten()
        return np.append(state_agents, state_trees)

    @staticmethod
    def update_estimation_communication(distributions):
        grid_size = distributions[0].grid_size
        n_agents = distributions[0].n_agents
        n_trees = distributions[0].n_trees
        final_distr = Lumberjacks_State_Distribution(n_trees=n_trees, n_agents=n_agents, grid_size=grid_size, agent_id=-1)
        total_observed = np.zeros((grid_size, grid_size))
        total_observed_trees = np.zeros((n_agents+1, grid_size, grid_size))
        total_agent_pos = np.ones((n_agents, grid_size, grid_size))
        for distr in distributions:
            total_observed += distr.observed_fields
            total_observed_trees += distr.observed_trees
            total_agent_pos *= distr.agent_pos_distribution+np.ones((grid_size, grid_size))*0.01 # Workaround

        for i in range(n_agents):
            total_agent_pos[i] = total_agent_pos[i]/np.sum(total_agent_pos[i])

        total_observed[total_observed.nonzero()] = np.ones((grid_size, grid_size))[total_observed.nonzero()]
        total_observed_trees[total_observed_trees.nonzero()] = np.ones((n_agents+1, grid_size, grid_size))[total_observed_trees.nonzero()]
        final_distr.observed_fields = total_observed
        final_distr.observed_trees = total_observed_trees
        final_distr.agent_pos_distribution = total_agent_pos
        return final_distr

    @staticmethod
    def set_from_belief_state(belief_state, agent_id, n_agents, n_trees, grid_size):
        distr = Lumberjacks_State_Distribution(n_agents=n_agents, n_trees=n_trees, grid_size=grid_size, agent_id=agent_id)
        state_agents = belief_state[:n_agents*grid_size**2].reshape((n_agents, grid_size, grid_size))
        state_trees = belief_state[n_agents*grid_size**2:].reshape((n_agents+1, grid_size, grid_size))
        distr.agent_pos_distribution = state_agents
        distr.observed_fields = np.zeros((grid_size, grid_size))
        distr.observed_fields[state_trees[0] == 0] = np.ones((grid_size, grid_size))[state_trees[0] == 0]
        distr.observed_trees = np.zeros((n_agents+1, grid_size, grid_size))
        distr.observed_trees[state_trees == 1] = np.ones((n_agents+1, grid_size, grid_size))[state_trees == 1]
        distr.observed_fields += np.sum(distr.observed_trees, axis=0)
        return distr

    @staticmethod
    def update_belief_state(belief_states, ids, n_agents, n_trees):
        grid_size = int(np.sqrt(belief_states[0].shape[0]/(2*n_agents+1)))
        distrs = []
        for i in range(len(belief_states)):
            distr = Lumberjacks_State_Distribution.set_from_belief_state(belief_states[i], ids[i], n_agents=n_agents, n_trees=n_trees, grid_size=grid_size)
            distrs.append(distr)
        final_distr = Lumberjacks_State_Distribution.update_estimation_communication(distrs)
        return final_distr.get_belief_state()