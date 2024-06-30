import sys
import torch
from torch.distributions import Normal, Bernoulli, MultivariateNormal

class Cat_Mouse_State_Distribution(torch.distributions.distribution.Distribution):

    def __init__(self, num_agents, num_mice, agent_id):
        super().__init__()
        self.num_agents = num_agents
        self.num_mice = num_mice
        self.agent_id = agent_id
        self.ini_var = 10
        self.agent_pos_distribution = []
        self.mouse_pos_distribution = []
        self.mouse_found_distribution = []
        for i in range(num_agents):
            position_mean = torch.Tensor([0.5, 0.5])
            position_var = torch.Tensor([[self.ini_var, 0],[0, self.ini_var]])
            agent_pos_i_distribution = MultivariateNormal(position_mean, position_var)
            self.agent_pos_distribution.append(agent_pos_i_distribution)
        
        for i in range(num_mice):
            position_mean = torch.Tensor([0.5, 0.5])
            position_var = torch.Tensor([[self.ini_var, 0],[0, self.ini_var]])
            mouse_pos_i_distribution = MultivariateNormal(position_mean, position_var)
            self.mouse_pos_distribution.append(mouse_pos_i_distribution)

        for i in range(num_mice):
            mouse_found_i_distribution = Bernoulli(0)
            self.mouse_found_distribution.append(mouse_found_i_distribution)

    """def log_prob(self, glob_state):
        lp = 0
        for i in range(self.num_agents):
            pos_agent_i = torch.Tensor(glob_state['agents']['position'][i])
            lp += self.agent_pos_distribution[i].log_prob(pos_agent_i)
        for i in range(self.num_mice):
            if glob_state['prey']['caught'][i] == 1:
                continue
            pos_mouse_i = torch.Tensor(glob_state['prey']['position'][i])
            lp += self.mouse_pos_distribution[i].log_prob(pos_mouse_i)
        for i in range(self.num_mice):
            found_mouse_i = torch.Tensor([glob_state['prey']['caught'][i]])
            lp +=  self.mouse_found_distribution[i].log_prob(found_mouse_i)[0] 
        return lp"""

    def update_estimation_local_observation(self, loc_obs):
        loc_agent_obs = loc_obs['agents']['position']
        loc_mice_obs = loc_obs['prey']['position']
        loc_mice_caught_obs = loc_obs['prey']['caught']
        for i in range(self.num_agents):
            if loc_agent_obs[i][0] != -1:
                self.agent_pos_distribution[i] = MultivariateNormal(torch.Tensor([loc_agent_obs[i][0], loc_agent_obs[i][1]]), torch.Tensor([[0.01, 0], [0, 0.01]]))
            elif self.agent_pos_distribution[i].covariance_matrix[0][0] < 4:
                self.agent_pos_distribution[i] = MultivariateNormal(self.agent_pos_distribution[i].mean, self.agent_pos_distribution[i].covariance_matrix + torch.Tensor([[1, 0], [0, 1]]))
        for i in range(self.num_mice):
            if loc_mice_caught_obs[i] == 1:
                continue
            if loc_mice_obs[i][0] != -1:
                self.mouse_pos_distribution[i] = MultivariateNormal(torch.Tensor([loc_mice_obs[i][0], loc_mice_obs[i][1]]), torch.Tensor([[0.01, 0], [0, 0.01]]))
            elif self.mouse_pos_distribution[i].covariance_matrix[0][0] < 4:
                self.mouse_pos_distribution[i] = MultivariateNormal(self.mouse_pos_distribution[i].mean, self.mouse_pos_distribution[i].covariance_matrix + torch.Tensor([[1, 0], [0, 1]]))
        for i in range(self.num_mice):
            if loc_mice_caught_obs[i] == 1:
                self.mouse_found_distribution[i] = Bernoulli(1)
            else:
                self.mouse_found_distribution[i] = Bernoulli(self.mouse_found_distribution[i].probs + 0.00)


    def get_belief_state(self):
        state = []
        for i in range(self.num_agents):
            state.append(self.agent_pos_distribution[i].mean[0])
            state.append(self.agent_pos_distribution[i].mean[1])
            state.append(self.agent_pos_distribution[i].covariance_matrix[0][0])
            state.append(self.agent_pos_distribution[i].covariance_matrix[1][1])
        for i in range(self.num_mice):
            state.append(self.mouse_pos_distribution[i].mean[0])
            state.append(self.mouse_pos_distribution[i].mean[1])
            state.append(self.mouse_pos_distribution[i].covariance_matrix[0][0])
            state.append(self.mouse_pos_distribution[i].covariance_matrix[1][1])
            state.append(self.mouse_found_distribution[i].probs)
        return state
        

    @staticmethod
    def update_estimation_communication(distributions, num_agents, num_mice):
        ini_var = 10
        final_distr = Cat_Mouse_State_Distribution(num_agents, num_mice, -1)
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

        for i in range(num_mice):
            mean = torch.Tensor([0,0])
            cov_sum = 0
            min_cov = ini_var
            for j in range(num_comm):
                cov = distributions[j].mouse_pos_distribution[i].covariance_matrix[0][0]
                mean += distributions[j].mouse_pos_distribution[i].mean/cov
                cov_sum += 1/cov
                min_cov += min(min_cov, cov)
            final_distr.mouse_pos_distribution[i] = MultivariateNormal(mean/cov_sum, torch.Tensor([[min_cov, 0],[0, min_cov]]))

        for i in range(num_mice):
            max_prob = 0
            for j in range(num_comm):
                max_prob = max(max_prob, distributions[j].mouse_found_distribution[i].probs)
            final_distr.mouse_found_distribution[i] = Bernoulli(max_prob)
        return final_distr
    
    @staticmethod
    def set_from_belief_state(belief_state, agent_id, num_agents, num_mice):
        distr = Cat_Mouse_State_Distribution(num_agents, num_mice, agent_id)
        for i in range(num_agents):
            distr.agent_pos_distribution[i].mean[0] = belief_state[4*i]
            distr.agent_pos_distribution[i].mean[1] = belief_state[4*i+1]
            distr.agent_pos_distribution[i].covariance_matrix[0][0] = belief_state[4*i+2]
            distr.agent_pos_distribution[i].covariance_matrix[1][1] = belief_state[4*i+3]    
        for i in range(num_mice):
            distr.mouse_pos_distribution[i].mean[0] = belief_state[5*i+4*num_agents]
            distr.mouse_pos_distribution[i].mean[1] = belief_state[5*i+1+4*num_agents]
            distr.mouse_pos_distribution[i].covariance_matrix[0][0] = belief_state[5*i+2+4*num_agents]
            distr.mouse_pos_distribution[i].covariance_matrix[1][1] = belief_state[5*i+3+4*num_agents]
            distr.mouse_found_distribution[i].probs = belief_state[5*i+4+4*num_agents]
        return distr
    
    @staticmethod
    def update_belief_state(belief_states, ids, num_agents, num_mice):
        distrs = []
        for i in range(len(belief_states)):
            distr = Cat_Mouse_State_Distribution.set_from_belief_state(belief_states[i], ids[i], num_agents, num_mice)
            distrs.append(distr)
        final_distr = Cat_Mouse_State_Distribution.update_estimation_communication(distrs, num_agents, num_mice)
        return final_distr.get_belief_state()
