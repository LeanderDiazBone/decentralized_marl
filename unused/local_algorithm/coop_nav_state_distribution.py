import sys
import torch
from torch.distributions import Normal, Bernoulli, MultivariateNormal

class Coop_Nav_State_Distribution(torch.distributions.distribution.Distribution):

    def __init__(self, num_agents, num_targets, agent_id):
        super().__init__()
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.agent_id = agent_id
        self.ini_var = 10
        self.agent_pos_distribution = []
        self.target_pos_distribution = []
        for i in range(self.num_agents):
            position_mean = torch.Tensor([0.5, 0.5])
            position_var = torch.Tensor([[self.ini_var, 0],[0, self.ini_var]])
            agent_pos_i_distribution = MultivariateNormal(position_mean, position_var)
            self.agent_pos_distribution.append(agent_pos_i_distribution)
        
        for i in range(self.num_targets):
            position_mean = torch.Tensor([0.5, 0.5])
            position_var = torch.Tensor([[self.ini_var, 0],[0, self.ini_var]])
            target_pos_i_distribution = MultivariateNormal(position_mean, position_var)
            self.target_pos_distribution.append(target_pos_i_distribution)

    """def log_prob(self, glob_state):
        lp = 0
        for i in range(self.num_agents):
            pos_agent_i = torch.Tensor(glob_state['agents']['position'][i])
            lp += self.agent_pos_distribution[i].log_prob(pos_agent_i)
        for i in range(self.num_targets):
            pos_target_i = torch.Tensor(glob_state['prey']['position'][i])
            lp += self.mouse_pos_distribution[i].log_prob(pos_target_i)
        return lp"""
    

    def update_estimation_local_observation(self, loc_obs):
        self.agent_pos_distribution[self.agent_id] = MultivariateNormal(torch.Tensor([loc_obs[2], loc_obs[3]]), torch.Tensor([[0.01, 0], [0, 0.01]]))
        for i in range(self.num_targets):
            self.target_pos_distribution[i] = MultivariateNormal(torch.Tensor([loc_obs[4+2*i]+loc_obs[2], loc_obs[5+2*i]+loc_obs[3]]), torch.Tensor([[0.01, 0], [0, 0.01]]))
        cor = 0
        for i in range(self.num_agents):
            if i == self.agent_id:
                cor = 1
                continue
            self.agent_pos_distribution[i] = MultivariateNormal(torch.Tensor([loc_obs[4+2*self.num_targets+2*(i-cor)]+loc_obs[2], loc_obs[5+2*self.num_targets+2*(i-cor)]]+loc_obs[3]), torch.Tensor([[0.01, 0], [0, 0.01]]))
    
    @staticmethod
    def update_estimation_communication(distributions, num_agents, num_targets, ini_var = 10):
        final_distr = Coop_Nav_State_Distribution(num_agents, num_targets, -1)
        num_comm = len(distributions)
        for i in range(num_agents):
            mean = torch.Tensor([0,0])
            cov_sum = 0
            min_cov = ini_var
            for j in range(num_comm):
                cov = distributions[j].agent_pos_distribution[i].covariance_matrix[0][0]
                mean += distributions[j].agent_pos_distribution[i].mean/cov
                cov_sum += 1/cov
                min_cov += min(min_cov, cov)
            final_distr.agent_pos_distribution[i] = MultivariateNormal(mean/cov_sum, torch.Tensor([[min_cov, 0],[0, min_cov]]))

        for i in range(num_targets):
            mean = torch.Tensor([0,0])
            cov_sum = 0
            min_cov = ini_var
            for j in range(num_comm):
                cov = distributions[j].target_pos_distribution[i].covariance_matrix[0][0]
                mean += distributions[j].target_pos_distribution[i].mean/cov
                cov_sum += 1/cov
                min_cov += min(min_cov, cov)
            final_distr.target_pos_distribution[i] = MultivariateNormal(mean/cov_sum, torch.Tensor([[min_cov, 0],[0, min_cov]]))
        return final_distr

    @staticmethod
    def set_from_belief_state(belief_state, agent_id, num_agents, num_targets):
        distr = Coop_Nav_State_Distribution(num_agents, num_targets, agent_id)
        for i in range(num_agents):
            distr.agent_pos_distribution[i].mean[0] = belief_state[4*i]
            distr.agent_pos_distribution[i].mean[1] = belief_state[4*i+1]
            distr.agent_pos_distribution[i].covariance_matrix[0][0] = belief_state[4*i+2]
            distr.agent_pos_distribution[i].covariance_matrix[1][1] = belief_state[4*i+3]    
        for i in range(num_targets):
            distr.target_pos_distribution[i].mean[0] = belief_state[2*i+4*num_agents]
            distr.target_pos_distribution[i].mean[1] = belief_state[4*i+1+4*num_agents]
            distr.target_pos_distribution[i].covariance_matrix[0][0] = belief_state[4*i+2+4*num_agents]
            distr.target_pos_distribution[i].covariance_matrix[1][1] = belief_state[4*i+3+4*num_agents]
        return distr
    
    @staticmethod
    def update_belief_state(belief_states, ids, num_agents, num_targets):
        distrs = []
        for i in range(len(belief_states)):
            distr = Coop_Nav_State_Distribution.set_from_belief_state(belief_states[i], ids[i], num_agents, num_targets)
            distrs.append(distr)
        final_distr = Coop_Nav_State_Distribution.update_estimation_communication(distrs, num_agents, num_targets)
        return final_distr.get_belief_state()

    def get_belief_state(self):
        state = []
        for i in range(self.num_agents):
            state.append(self.agent_pos_distribution[i].mean[0])
            state.append(self.agent_pos_distribution[i].mean[1])
            state.append(self.agent_pos_distribution[i].covariance_matrix[0][0])
            state.append(self.agent_pos_distribution[i].covariance_matrix[1][1])
        for i in range(self.num_targets):
            state.append(self.target_pos_distribution[i].mean[0])
            state.append(self.target_pos_distribution[i].mean[1])
            state.append(self.target_pos_distribution[i].covariance_matrix[0][0])
            state.append(self.target_pos_distribution[i].covariance_matrix[1][1])
        return state
    
    