import time
import local_agent
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pettingzoo.mpe import simple_spread_v3
import random
sys.path.append("../")
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
# from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *
from agent import Agent
from cat_mouse_state_distribution import Cat_Mouse_State_Distribution
from coop_nav_state_distribution import Coop_Nav_State_Distribution
from lumberjack_state_distribution import Lumberjacks_State_Distribution
from local_agent import Local_Agent
import pandas as pd

def get_local_observations_lumber(global_observation, n_agents, n_trees):
    local_obs = []
    comm = []
    for i in range(n_agents):
        comm.append([])
        agent_pos = global_observation[2*i:2*(i+1)]
        loc_obs = global_observation
        for j in range(n_agents):
            if abs(global_observation[2*j]-agent_pos[0]) > 1 or abs(global_observation[2*j+1]-agent_pos[1]) > 1:
                loc_obs[2*j] = -1
                loc_obs[2*j+1] = -1
            else:
                 comm[i].append(j)
        for j in range(n_trees):
            if abs(global_observation[2*n_agents+3*j]-agent_pos[0]) > 1 or abs(global_observation[2*n_agents+3*j+1]-agent_pos[1]) > 1:
                loc_obs[2*n_agents+3*j] = -1
                loc_obs[2*n_agents+3*j+1] = -1
        local_obs.append(loc_obs)
    return local_obs, comm

def state_to_array_lumber(state):
	state_list = []
	for agent in state[0]:
		state_list.append(agent[0]-1)
		state_list.append(agent[1]-1)
	for tree in state[1]:
		state_list.append(tree[0][0]-1)
		state_list.append(tree[0][1]-1)
		state_list.append(tree[1])
	return np.array(state_list)

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

def train_lumberjacks_local(n_agents = 2, n_trees = 5, grid_size = 4, num_episodes = 10000, eval = False, csv_loc = "rewards_experiments/episodes_rewards.csv", exp_name = ""):
    get_local_obs = lambda env: get_local_observations_lumber(state_to_array_lumber(env.get_global_obs()), n_agents, n_trees)
    env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(grid_size, grid_size), n_agents=n_agents, n_trees=n_trees)
    state_dim = (2*n_agents+1)*grid_size**2
    train_when_full = False
    agents = []
    for i in range(n_agents):
        state_distr = Lumberjacks_State_Distribution(n_agents, n_trees, grid_size, i)
        local_agent = Local_Agent((state_dim, ), n_agents, n_trees, 5, i, state_distr, distr=Lumberjacks_State_Distribution, alpha=10e-6, layer_size=128, memory_max_size=50, train_when_full=train_when_full)
        if eval:
             local_agent.load_models()
        agents.append(local_agent)
    trans_ind = 0
    eps_rewards = []
    avg_rewards = []
    best_reward = -20
    for ep in range(num_episodes):
        if ep % 100 == 0 and ep > 0 and not eval:
            print(f"Episode: {ep}")
            av_rew = np.mean(np.array(eps_rewards[ep-100:]))
            print(av_rew)
            avg_rewards.append(av_rew)
            if av_rew > best_reward:
                best_reward = av_rew
                for i in range(n_agents):
                    agents[i].save_models()
        state = env.reset()
        for i in range(n_agents):
            agents[i].state_distr.reset()
        obs, comm = get_local_obs(env)
        ep_rew = 0
        discount = 1
        for _ in range(25):
            actions_all = []
            probs_all = []
            vals_all = []
            for i in range(n_agents):
                action, probs, value, _ = agents[i].choose_action(obs[i])
                actions_all.append(action) 
                probs_all.append(probs)
                vals_all.append(value)
            _, reward, done, _ = env.step(actions_all)
            obs_n, comm = get_local_obs(env)
            for i in range(n_agents):
                ep_rew += discount*reward[i]
                agents[i].observe(actions_all[i], probs_all[i], vals_all[i], reward[i], done[i], trans_ind)
            discount *= 0.99
            if eval:
                env.render()
                print(ep_rew)
            trans_ind += 1
            obs = obs_n
            if np.all(done):
                 break
            bubbles = communication_bubbles(comm)
            for bubble in bubbles:
                if len(bubble) > 1:
                    agent_list = []
                    for el in bubble:
                        agent_list.append(agents[el])
                    Local_Agent.communicate(agents, n_agents, n_trees, Lumberjacks_State_Distribution)
            if eval:
                time.sleep(60)
        if not eval and not train_when_full:
            for agent in agents:
                agent.learn()
        eps_rewards.append(ep_rew)
    if not eval:
        agent.save_models()
    reward_out = pd.DataFrame(eps_rewards, columns=['ep_rew'])
    reward_out.to_csv(csv_loc)
    plt.plot(avg_rewards)
    plt.savefig(f'Plots/{exp_name}_eps_reward.png')

def run_experiments():
    rew_dir = "rewards_experiments/"
    model_dir = 'tmp2'
    if not os.path.exists(rew_dir):
        os.makedirs(rew_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    n_agents_list = [2, 2, 2, 3, 4]
    n_trees_list = [4, 5, 6, 5, 5]
    grid_size_list = [3, 4, 5, 4, 4]
    for i in range(len(n_agents_list)):
        n_agents = n_agents_list[i]
        n_trees = n_trees_list[i]
        grid_size = grid_size_list[i]
        exp_name = f"ag_{n_agents}_tr_{n_trees}_gs_{grid_size}"
        print(exp_name)
        train_lumberjacks_local(n_agents = n_agents, n_trees = n_trees, grid_size = grid_size, num_episodes = 30000, eval = False, csv_loc=rew_dir+f"episodes_rewards_{exp_name}.csv", exp_name=exp_name)


def __main__():
    # -5.922 after 5000 eps equall step size
    train_lumberjacks_local(n_agents = 3, n_trees = 3, grid_size = 3, num_episodes = 30000, eval = False)

__main__()