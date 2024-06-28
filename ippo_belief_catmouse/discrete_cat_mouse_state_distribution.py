import sys
import torch
import time
import numpy as np
from torch.distributions import Normal, Bernoulli, MultivariateNormal

class Discrete_CatMouse_State_Distribution:

    def __init__(self, n_agents, n_mice, grid_size, agent_id, belief_radius = 2, obs_rad = 1):
        super().__init__()
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_mice = n_mice
        self.belief_radius = belief_radius
        self.obs_rad = obs_rad
        self.grid_size = grid_size
        self.caught_mice = set()
        self.agent_pos_distribution = np.ones((n_agents, grid_size, grid_size))/(grid_size**2)
        self.mice_pos_distribution = np.ones((n_mice, grid_size, grid_size))/(grid_size**2)
    
    def reset(self):
        self.agent_pos_distribution = np.ones((self.n_agents, self.grid_size, self.grid_size))/(self.grid_size**2)
        self.mice_pos_distribution = np.ones((self.n_mice, self.grid_size, self.grid_size))/(self.grid_size**2)
        self.caught_mice = set()

    def get_view(self, agent_pos):
        view = np.ones((self.grid_size, self.grid_size))
        its = [i for i in range(self.obs_rad+1)]+[-i for i in range(1, self.obs_rad+1)]
        its.sort()
        for i in its:
            for j in its:
                if agent_pos[0]+i >= 0 and agent_pos[0]+i < self.grid_size and agent_pos[1]+j >= 0 and agent_pos[1]+j < self.grid_size:
                    view[int(agent_pos[0])+i][int(agent_pos[1])+j] = 0
        #print("View")
        #print(view)
        return view
    
    def random_walk_update(self, distribution):
        # Assumption is 9 actions equally likely
        #print("Walk")
        #print(distribution)
        new_distr = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                trans_prob = 1/9
                if (i == 0 or i == self.grid_size-1) and (j == 0 or j == self.grid_size-1):
                    trans_prob = 1/4
                elif (i == 0 or i == self.grid_size-1) or (j == 0 or j == self.grid_size-1):
                    trans_prob = 1/6
                new_distr[i][j] += trans_prob * distribution[i][j] # stay
                if i > 0:
                    new_distr[i-1][j] += trans_prob * distribution[i][j] # up
                    if j > 0:
                        new_distr[i-1][j-1] += trans_prob * distribution[i][j] # up left
                    if j < self.grid_size-1:
                        new_distr[i-1][j+1] += trans_prob * distribution[i][j] # up right
                if i < self.grid_size-1:
                    new_distr[i+1][j] += trans_prob * distribution[i][j] # down
                    if j > 0:
                        new_distr[i+1][j-1] += trans_prob * distribution[i][j]  # down left
                    if j < self.grid_size-1:
                        new_distr[i+1][j+1] += trans_prob * distribution[i][j] # down right
                if j > 0:
                    new_distr[i][j-1] += trans_prob * distribution[i][j] # left
                if j < self.grid_size-1:
                    new_distr[i][j+1] += trans_prob * distribution[i][j]  # right
        #print(new_distr)
        return new_distr


    def update_estimation_local_observation(self, loc_obs):
        self.env_step = loc_obs[0]
        agents = loc_obs[:self.n_agents*2].astype('int')
        mice = loc_obs[self.n_agents*2:].astype('int')
        view = self.get_view(agents[2*self.agent_id:2*(self.agent_id+1)])

        for i in range(self.n_agents):
            agent = agents[2*i:2*(i+1)]
            if agent[0] != -1:
                distr = np.zeros((self.grid_size, self.grid_size))
                distr[agent[0]][agent[1]] = 1
                self.agent_pos_distribution[i] = distr
            else:
                if i == self.agent_id:
                    print("error")
                distr = self.random_walk_update(self.agent_pos_distribution[i])
                distr = distr*view
                if np.sum(distr) > 0:
                    distr /= np.sum(distr)
                else:
                    print("Issue in Agent division")
                self.agent_pos_distribution[i] = distr

        for i in range(self.n_mice):
            mouse = mice[3*i:3*(i+1)]
            #print(mouse)
            if mouse[2] == 1 or i in self.caught_mice: #(Caught)
                self.caught_mice.add(i)
                distr = np.zeros((self.grid_size, self.grid_size))
                self.mice_pos_distribution[i] = distr
            else:
                if mouse[0] != -1:
                    distr = np.zeros((self.grid_size, self.grid_size))
                    distr[mouse[0]][mouse[1]] = 1
                    self.mice_pos_distribution[i] = distr
                else:
                    distr = self.random_walk_update(self.mice_pos_distribution[i])
                    distr = distr*view
                    if np.sum(distr) > 0:
                        distr /= np.sum(distr)
                    else:
                        print("Issue in Mouse division")
                        #self.caught_mice.add(i)
                    self.mice_pos_distribution[i] = distr

    def get_belief_state(self):
        mice_state = np.sum(self.mice_pos_distribution, axis=0)
        mice_state_centralized = np.zeros((2*self.grid_size-1, 2*self.grid_size-1))
        agent_state_centralized = np.zeros((self.n_agents, 2*self.grid_size-1, 2*self.grid_size-1))
        ag_pos = np.where(self.agent_pos_distribution[self.agent_id] == 1)
        ag_pos = [ag_pos[0][0], ag_pos[1][0]]
        x = np.array(list(range(self.grid_size-ag_pos[0]-1,2*self.grid_size-ag_pos[0]-1, 1))) # sagt an der stelle wohin er kopiert wird.
        y = np.array(list(range(self.grid_size-ag_pos[1]-1,2*self.grid_size-ag_pos[1]-1, 1)))
        mice_state_centralized[np.ix_(x,y)] = mice_state
        for i in range(self.n_agents):
            agent_state_centralized[i][np.ix_(x,y)] = self.agent_pos_distribution[i]
        
        start = self.grid_size-1-self.belief_radius
        end = self.grid_size+self.belief_radius
        state_agents = np.delete(agent_state_centralized, self.agent_id, axis=0)[:, start:end, start:end].flatten()
        state_mice = mice_state_centralized[start:end, start:end].flatten()
        #print(state_agents.shape)
        #print(state_mice.shape)
        #print(np.append(state_agents, state_mice))
        return np.append(state_agents, state_mice)

    @staticmethod
    def update_estimation_communication(distributions):
        #print("Communication")
        grid_size = distributions[0].grid_size
        n_agents = distributions[0].n_agents
        n_mice = distributions[0].n_mice
        belief_radius = distributions[0].belief_radius
        obs_rad = distributions[0].obs_rad
        final_distr = Discrete_CatMouse_State_Distribution(n_agents, n_mice, grid_size, agent_id=-1, belief_radius = belief_radius, obs_rad = obs_rad)
        final_distr.agent_pos_distribution = np.ones((n_agents, grid_size, grid_size))
        final_distr.mice_pos_distribution = np.ones((n_mice, grid_size, grid_size))
        for distr in distributions:
            final_distr.caught_mice = final_distr.caught_mice | distr.caught_mice
            final_distr.agent_pos_distribution *= distr.agent_pos_distribution#+np.ones((grid_size, grid_size))*0.01 # Workaround
            final_distr.mice_pos_distribution *= distr.mice_pos_distribution
            
        for i in range(n_agents):
            for distr in distributions:
                if 1 in distr.agent_pos_distribution[i]:
                    final_distr.agent_pos_distribution[i] = distr.agent_pos_distribution[i]

        for i in range(n_mice):
            for distr in distributions:
                if 1 in distr.mice_pos_distribution[i]:
                    final_distr.mice_pos_distribution[i] = distr.mice_pos_distribution[i]
                
        for i in range(n_agents):
            final_distr.agent_pos_distribution[i] = final_distr.agent_pos_distribution[i]/np.sum(final_distr.agent_pos_distribution[i])
            if np.sum(final_distr.mice_pos_distribution[i]) > 0:
                final_distr.mice_pos_distribution[i] = final_distr.mice_pos_distribution[i]/np.sum(final_distr.mice_pos_distribution[i])

        return final_distr