from collections import deque
import numpy as np
import torch
import random
from itertools import count

class ReplayBuffer():

    def __init__(self, min_size, max_size):
        self.buffer = deque([],maxlen = max_size)
        self.min_size = min_size

    def put(self,transition):
        self.buffer.append(transition)

    # gym environment returns np arrays, turn into tensors for q learning
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_next_lst = [], [], [], []

        for t in mini_batch:
            s,a,r,s_next = t
            s_lst.append(s)
            a_lst.append([int(a)])
            r_lst.append([r])
            s_next_lst.append(s_next)

        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float)
        a_batch = torch.tensor(np.array(a_lst), dtype=torch.int64)
        r_batch = torch.tensor(np.array(r_lst), dtype=torch.float)
        s_next_batch = torch.tensor(np.array(s_next_lst), dtype=torch.float)

        # maybe normalize rewards

        return s_batch, a_batch, r_batch, s_next_batch

    def size(self):
        return len(self.buffer)
    
    def start_training(self):
        return self.size() >= self.min_size


def run_episode(env, agent, done_rep, train=True):

    state = env.reset()
    state = state.flatten()

    for t in count():
        action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)
        next_state = state.flatten()

        if done:
            next_state = done_rep

        if train:
            # memory buffer, train_agent
            agent.memory.put((state, action, reward, next_state))
            if agent.memory.start_training():
                agent.train_agent()
        else:
            env.render()

        if done:
            break

        state = next_state

    print(t)
    print(env.score)
    print(env.highest())
