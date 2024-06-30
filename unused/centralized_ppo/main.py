import time
from typing import List
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


d = []
for x in range(5):
	for y in range(5):
		d.append([x, y])

def state_to_array_lumber(state, steps_since_start, grid_size):
	state_list = []
	for agent in state[0]:
		state_list.append(agent[0]/grid_size)
		state_list.append(agent[1]/grid_size)
	for tree in state[1]:
		state_list.append(tree[0][0]/grid_size)
		state_list.append(tree[0][1]/grid_size)
		state_list.append(tree[1]/grid_size)
	state_list = [steps_since_start]+state_list
	return np.array(state_list)

def state_to_array_lumber_2(state, n_agents, grid_size):
	agents = np.zeros((n_agents, grid_size, grid_size))
	trees = np.zeros((grid_size, grid_size))
	num = 0
	for agent in state[0]:
		agents[num][agent[0]-1][agent[1]-1] = 1
		num += 1
	for tree in state[1]:
		trees[tree[0][0]-1][tree[0][1]-1] = tree[1]
	return np.append(agents.flatten(), trees.flatten())	
	
#Hyperparameters
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
learning_rate = 0.0005
gamma		 = 0.98
lmbda		 = 0.95
eps_clip	  = 0.1
K_epoch	   = 3
T_horizon	 = 20


def cross(s1: List[float], s2: List[float]) -> List[float]:
	ret = []
	for sx in s1:
		for sy in s2:
			ret.append(sx + sy)
	return ret


class PPO(nn.Module):
	def __init__(self, state_dim):
		super(PPO, self).__init__()
		self.data = []
		self.fc1   = nn.Linear(state_dim, 256)
		self.fc_pi = nn.Linear(256, 25)
		self.fc_v  = nn.Linear(256, 1)
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

	def pi(self, x, softmax_dim = 0):
		x = F.relu(self.fc1(x))
		x = self.fc_pi(x)
		prob = F.softmax(x, dim=softmax_dim)
		return prob

	def v(self, x):
		x = F.relu(self.fc1(x))
		v = self.fc_v(x)
		return v
	  
	def put_data(self, transition):
		self.data.append(transition)

	def make_batch(self):
		s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
		for transition in self.data:
			s, a, r, s_prime, prob_a, done = transition

			s_lst.append(s)
			a_lst.append([a])
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			prob_a_lst.append([prob_a])
			done_mask = 0 if done else 1
			done_lst.append([done_mask])

		s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
										  torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
										  torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
		self.data = []
		return s, a, r, s_prime, done_mask, prob_a

	def train_net(self):
		s, a, r, s_prime, done_mask, prob_a = self.make_batch()
		for i in range(K_epoch):
			td_target = r + gamma * self.v(s_prime) * done_mask
			delta = td_target - self.v(s)
			delta = delta.detach().numpy()
			advantage_lst = []
			advantage = 0.0
			for delta_t in delta[::-1]:
				advantage = gamma * lmbda * advantage + delta_t[0]
				advantage_lst.append([advantage])
			advantage_lst.reverse()
			advantage = torch.tensor(advantage_lst, dtype=torch.float)

			pi = self.pi(s, softmax_dim=1)
			pi_a = pi.gather(1,a)
			ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
			loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

	def save_weights(self, filepath):
		torch.save(self.state_dict(), filepath)

	def load_weights(self, filepath):
		self.load_state_dict(torch.load(filepath))


def train_lumber(env, n_agents, n_trees, grid_size):
	state_dim = 2*n_agents+3*n_trees+1
	state_dim = (n_agents+1)*grid_size**2
	#state_dim = 44
	model = PPO(state_dim)
	score = 0.0
	print_interval = 50
	for n_epi in range(5000):
		s = env.reset()
		s = np.array(s[0])
		s = state_to_array_lumber_2(env.get_global_obs(), n_agents, grid_size)
		done = [False, False]
		while not all(done):
			for t in range(T_horizon):
				prob = model.pi(torch.from_numpy(s).float())
				m = Categorical(prob)
				a = m.sample().item()
				s_prime, r, done, info = env.step(d[a])
				#s_prime = np.array([s_prime[0], s_prime[1]])
				s_prime = np.array(s_prime[0])
				s_prime = state_to_array_lumber_2(env.get_global_obs(), n_agents, grid_size)
				model.put_data((s, a, sum(r)/100.0, s_prime, prob[a].item(), all(done)))
				s = s_prime

				score += sum(r)
				if all(done):
					break
			model.train_net()
		if n_epi%print_interval==0 and n_epi != 0:
			print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
			score = 0.0
	env.close()
	model.save_weights('weights_1000')
	return model
	
def main():
	n_agents = 1
	n_trees = 6
	#state_dim = 44
	#state_dim = 2*n_agents+3*n_trees+1
	
	grid_size = 3
	state_dim = (n_agents+1)*grid_size**2
	env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees)
	model = train_lumber(env, n_agents, n_trees, grid_size)
	model = PPO(state_dim=state_dim)
	#model.load_weights('weights_1000')
	for i in range(10):
		done = [False, False]
		s = env.reset()
		s = state_to_array_lumber_2(env.get_global_obs(), n_agents, grid_size)
		while not all(done):
			for t in range(T_horizon):
				#env.render()
				#print(s[0:22])
				#print(s[22:44])
				#time.sleep(0.2)
				prob = model.pi(torch.from_numpy(s).float())
				m = Categorical(prob)
				a = m.sample().item()
				s_prime, r, done, info = env.step(d[a])
				s_prime = state_to_array_lumber_2(env.get_global_obs(), n_agents, grid_size)
				# s_prime_comb = np.array(cross(s_prime[0], s_prime[1]))
				#s_prime_comb = np.array(s_prime[0])
				model.put_data((s, a, sum(r)/100.0, s_prime, prob[a].item(), all(done)))
				s = s_prime
				if all(done):
					break
	time.sleep(100)
	env.close()

	




	# obs_n = env.reset()
	# while not all(done_n):
	# 	env.render()
	# 	obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
	# 	print(env.action_space.sample())
	# 	print('obs_n:', obs_n)
	# 	print('reward_n:', reward_n)
	# 	print('done_n:', done_n)
	# 	print(info)
	# 	ep_reward += sum(reward_n)
	# 	time.sleep(0.1)
	# env.close()

if __name__ == '__main__':
	main()
