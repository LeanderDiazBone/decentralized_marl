import time
from typing import List
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
import pandas as pd


#Hyperparameters
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
learning_rate = 0.0005
gamma		 = 0.98
lmbda		 = 0.95
eps_clip	  = 0.1
K_epoch	   = 3
T_horizon	 = 25


def cross(s1: List[float], s2: List[float]) -> List[float]:
	ret = []
	for sx in s1:
		for sy in s2:
			ret.append(sx + sy)
	return ret


class PPO(nn.Module):
	def __init__(self):
		super(PPO, self).__init__()
		self.data = []
		self.fc1   = nn.Linear(54, 512)
		self.fc_pi = nn.Linear(512, 125)
		self.fc_v  = nn.Linear(512, 1)
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


def main():
	d = []
	for x in range(5):
		for y in range(5):
			for z in range(5):
				d.append([x, y, z])

	env = simple_spread_v3.parallel_env(N=3, max_cycles=25, local_ratio=0.5, render_mode='human', continuous_actions=False)

	done_n = [False for _ in range(env.max_num_agents)]
	ep_reward = 0

	model = PPO()
	score = 0.0
	print_interval = 1000
	eps = []
	rewards = []
	# for n_epi in range(60_000):
	# 	re = 0
	# 	s, _ = env.reset()
	# 	s = list(s.values())
	# 	s = np.concatenate((s[0], s[1], s[2]))
	# 	# s = np.array(cross(s[0], s[1]))
	# 	# s = np.array(s[0])
	# 	done = [False, False, False]
	# 	while not all(done):
	# 		for t in range(T_horizon):
	# 			prob = model.pi(torch.from_numpy(s).float())
	# 			m = Categorical(prob)
	# 			a = m.sample().item()
	# 			actions_dict = {}
	# 			for agent_id, agent in enumerate(env.agents):
	# 				actions_dict[agent] = d[a][agent_id]
	# 			s_prime, r, done, _, info = env.step(actions_dict)
	# 			# s_prime_comb = np.array(cross(s_prime[0], s_prime[1]))
	# 			s_prime = list(s_prime.values())
	# 			s_prime = np.concatenate((s_prime[0], s_prime[1], s_prime[2]))
	# 			model.put_data((s, a, sum(list(r.values()))/100.0, s_prime, prob[a].item(), all(done)))
	# 			s = s_prime
	# 			re += sum(list(r.values()))
	# 			score += sum(list(r.values()))
	# 			if all(list(done.values())):
	# 				break
	# 		model.train_net()
	# 	if n_epi % print_interval == 0 and n_epi != 0:
	# 		print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
	# 		# Save the model weights
	# 		model.save_weights(f'ppo_model_weights{n_epi}_{score/print_interval}.pth')
	# 		score = 0.0


	# 	if n_epi % print_interval == 0 and n_epi > 0:
	# 		eps.append(n_epi)
	# 		rewards.append(re)
	for _ in range(20):
		env.reset()
	model.load_weights('./ppo_model_weights59000_-96.15694764512232.pth')
	for x in range(10):
		s, _ = env.reset()
		# s = np.array(cross(s[0], s[1]))
		s = list(s.values())
		s = np.concatenate((s[0], s[1], s[2]))
		# s = np.array(s[0])

		done = [False, False]
		while not all(done):
			for t in range(T_horizon):
				env.render()
				time.sleep(0.125)
				prob = model.pi(torch.from_numpy(s).float())
				m = Categorical(prob)
				a = m.sample().item()
				actions_dict = {}
				for agent_id, agent in enumerate(env.agents):
					actions_dict[agent] = d[a][agent_id]
				s_prime, r, done, _, info = env.step(actions_dict)
				# s_prime, r, done, _, info = env.step(d[a])
				# print(done)
				# s_prime_comb = np.array(cross(s_prime[0], s_prime[1]))
				# s_prime_comb = np.array(s_prime[0])
				s_prime = list(s_prime.values())
				s_prime = np.concatenate((s_prime[0], s_prime[1], s_prime[2]))
				model.put_data((s, a, sum(list(r.values()))/100.0, s_prime, prob[a].item(), all(done)))
				s = s_prime

				score += sum(list(r.values()))
				if all(list(done.values())):
					break

	# plt.figure(figsize=(10, 5))
	# plt.plot(eps, rewards)
	# plt.xlabel('Episodes')
	# plt.ylabel('Reward')
	# plt.title('Reward vs Episodes')
	# plt.grid(True)

	# # Save the plot to a file
	# plt.savefig('reward_vs_episodes.png')

	# # Storing the data in a .csv file
	# data = {'Episodes': eps, 'Reward': rewards}
	# df = pd.DataFrame(data)
	# df.to_csv('reward_vs_episodes.csv', index=False)

	time.sleep(10)
	env.close()

	# if n_epi%print_interval==0 and n_epi!=0:
	# 	print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
	# 	score = 0.0

	# env.close()

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
