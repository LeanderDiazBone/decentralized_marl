import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import random
from utils import ReplayBuffer, run_episode
import time
import math
from decentralized_marl.marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from decentralized_marl.marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *

class NeuralNet(nn.Module):

    def __init__(self,state_dim: int, action_dim: int):
        super(NeuralNet, self).__init__()
        hidden_dim = 128
        hidden_layers = 1
        layers_fc = []
        layers_fc.append(nn.Linear(state_dim, hidden_dim))
        layers_fc.append(nn.ReLU())
        for _ in range(hidden_layers):
            layers_fc.append(nn.Linear(hidden_dim,hidden_dim))
            layers_fc.append(nn.ReLU())
        layers_fc.append(nn.Linear(hidden_dim,action_dim))

        self.layers_fc = nn.ModuleList(layers_fc)


    def forward(self, s: torch.Tensor) -> torch.Tensor:
        for l in self.layers_fc:
            s = l(s)
        return s
        
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, lr):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.setup_dqn()

    def setup_dqn(self):
        self.network = NeuralNet(self.state_dim, self.action_dim)
        self.optimizer = optim.AdamW(self.network.parameters(), lr = self.lr, amsgrad=True)

class Agent:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = 128
        self.discount = 0.99
        self.dqn_lr = 1e-4
        self.min_buffer_size = 128
        self.max_buffer_size = 10000
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size)
        self.eps_start = 0.9
        self.eps_end  = 0.05
        self.eps_decay = 1000
        self.tau = 0.005
        self.steps_done = 0
        self.setup_agent()

    def setup_agent(self):
        self.dqn = DQN(self.state_dim, self.action_dim, self.dqn_lr)
        self.dqn_target = DQN(self.state_dim, self.action_dim, self.dqn_lr)
        self.dqn_target.network.load_state_dict(self.dqn.network.state_dict())

    def calc_qstar_target(self, mini_batch):
        _, _, r, s_next = mini_batch
        non_final_mask = torch.tensor(tuple(map(lambda s: s[0] != 5,
                                          s_next)), dtype=torch.bool)
        non_final_next_states = torch.cat([s.unsqueeze(0) for s in s_next
                                                if not s[0] != 5], dim=0)
        maxq = torch.zeros((self.batch_size))
        with torch.no_grad():
            # target is r + gamma * max_a Q_old(x_next,a)
            maxq[non_final_mask] = self.dqn_target.network(non_final_next_states).max(1).values
            maxq = maxq.unsqueeze(1)
            target = r + self.discount * maxq
        return target
    
    @staticmethod
    def run_gradient_update_step(object, loss: torch.Tensor):
        object.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(object.network.parameters(), 100)
        object.optimizer.step()

    def train_agent(self):
        # start = time.time()
        mini_batch = self.memory.sample(self.batch_size)
        # end = time.time()
        # print("sample", end-start)

        # start = time.time()
        s_batch, a_batch, _, _ = mini_batch
        qstar_target = self.calc_qstar_target(mini_batch)
        dqn_loss = F.smooth_l1_loss(self.dqn.network(s_batch).gather(1, a_batch), qstar_target)
        self.run_gradient_update_step(self.dqn, dqn_loss)
        # end = time.time()
        # print("update", end-start)
        
        # start = time.time()
        target_state_dict = self.dqn_target.network.state_dict()
        q_state_dict = self.dqn.network.state_dict()
        for key in q_state_dict:
            target_state_dict[key] = q_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.dqn_target.network.load_state_dict(target_state_dict)
        # end = time.time()
        # print("soft dqn update", end-start)
    
    def get_action(self, state: np.ndarray):

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample < eps_threshold:
            action = np.random.randint(self.action_dim)
        else:
            # normalize state dimensions
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state.copy(), dtype=torch.float)
            with torch.no_grad():
                action = self.dqn.network(state).max(1).indices
            action = action.flatten().item()
        return action

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="human")

    n_actions = 2
    state_dim = 4

    agent = Agent(state_dim, n_actions)
    for i in range(600):
        print("ep ", i)
        done_rep = np.array([5,0,0,0])
        run_episode(env, agent, done_rep, train=True)

    for _ in range(1):
        run_episode(env, agent, train=False)

    env.close()