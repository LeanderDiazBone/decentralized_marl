import time
import local_agent
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
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
# def train_cat_mouse(num_episodes = 5000):
#     n_agents = 1
#     n_prey = 2
#     env = CatMouseMA(n_agents=n_agents, n_prey=n_prey) #render_mode='human'_agents+3*n_mice, action_dim=1, num_agents=num_agents, env=env)
#     state_dim = 4*n_agents+5*n_prey 
#     agent = Agent(5, (state_dim,), alpha=0.00003, n_epochs=4, batch_size=16)
#     state_distr = Cat_Mouse_State_Distribution(n_agents, n_prey, 0)
#     eps_rewards = []
#     for ep in range(num_episodes):
#         env.reset()
#         belief_state = state_distr.get_belief_state()
#         observations, communication = env._get_obs()
#         state_distr.update_estimation_local_observation(observations[0])
#         ep_rew = 0
#         discount = 1
#         for i in range(100):
#             action, probs, value, _ = agent.choose_action(belief_state)
#             state_n, reward, done, _, _ = env.step([action/5])
#             ep_rew += discount*reward[0]
#             discount *= 0.99
#             eps_rewards.append(ep_rew)
#             observations_n, communication_n = env._get_obs()
#             state_distr.update_estimation_local_observation(observations_n[0])
#             belief_state_n = state_distr.get_belief_state()
#             agent.remember(belief_state_n, action, probs, value, reward, done)
#             if np.all(done):
#                 break
#         print(ep_rew)
#         eps_rewards.append(ep_rew)
#         agent.learn()

def global_state_navigation(state, n_agents):
    loc_obs = state['agent_0']
    global_state = []
    global_state.append(loc_obs[2])
    global_state.append(loc_obs[3])

    for i in range(n_agents-1):
        global_state.append(loc_obs[4+2*n_agents]+loc_obs[2])
        global_state.append(loc_obs[5+2*n_agents]+loc_obs[3])

    for i in range(n_agents):
        global_state.append(loc_obs[4+2*i]+loc_obs[2])
        global_state.append(loc_obs[5+2*i]+loc_obs[3])
    return global_state

def get_actions(action, n_actions, n_agents):
    actions = []
    for _ in range(n_agents):
        ac = action % n_actions
        actions.append(ac)
        action -= ac
        action = int(action/n_actions)
    return actions

def train_cooperative_nav(num_episodes, train = True):
    n_agents = 3
    n_actions = 5
    render_mode = 'human'
    if train:
        render_mode = None
    env = simple_spread_v3.parallel_env(N=n_agents, max_cycles=50,local_ratio=0.5, continuous_actions=False, render_mode=render_mode) #Tr/Te
    state_dim = 4*n_agents
    entropy_scaling = lambda learning_step: 5* (num_episodes-learning_step)/num_episodes 
    agent = Agent(n_actions**n_agents, (state_dim,), alpha=0.0003, n_epochs=10, batch_size = 64, n_steps = 100, entropy_scaling=entropy_scaling)
    if not train:
        agent.load_models()
    eps_rewards = []
    best_avg_rew = -100000
    for ep in range(num_episodes):
        if ep % 50 == 0 and ep > 0 and train:
            print(f"Episode: {ep}")
            ep_avg_rew = np.mean(np.array(eps_rewards[ep-50:]))
            print(ep_avg_rew)  
            if ep_avg_rew >= best_avg_rew:
                best_avg_rew = ep_avg_rew
                agent.save_models()
        state, _ = env.reset()
        state = global_state_navigation(state, n_agents)
        ep_rew = 0
        discount = 1
        for i in range(50):
            action, probs, value, _ = agent.choose_action([state])
            actions = get_actions(action, n_actions, n_agents)
            all_actions = {}
            for i in range(n_agents):
                all_actions[f"agent_{i}"] = actions[i]
            state_n, reward, done, _, _ = env.step(all_actions)
            if not train:
                env.render()
                print(probs)
                time.sleep(0.25)
            state_n = global_state_navigation(state_n, n_agents)
            tot_reward = 0
            tot_done = True
            for i in range(n_agents):
                tot_reward += reward[f"agent_{i}"]
                tot_done =  tot_done and done[f"agent_{i}"]
            ep_rew += discount*tot_reward
            discount *= 0.99
            eps_rewards.append(ep_rew)
            agent.remember(state, action, probs, value, tot_reward, tot_done)
            if tot_done:
                break
            state = state_n
            # print(state_n['agent_0'][2].item() + state_n['agent_0'][4].item(), state_n['agent_0'][3].item() + state_n['agent_0'][5].item())
            # print(belief_state_n)
        if train:
            agent.learn()   #Training
    # # Training

def distance_agents(state):
    return state[4]**2+state[5]**2

def train_coop_navigation_local(num_episodes, eval = False):
    episode_history = []
    score_history = []
    n_agents = 3
    n_targets = n_agents
    render_mode = None
    if eval:
        render_mode = "human"
    env = simple_spread_v3.parallel_env(N=n_agents, max_cycles=50,local_ratio=0.5, continuous_actions=False, render_mode=render_mode) #Tr/Te
    state_dim = 4*(n_agents+n_targets)
    agents = []
    for i in range(n_agents):
        state_distr = Coop_Nav_State_Distribution(n_agents, n_targets, i)
        local_agent = Local_Agent((state_dim, ), n_agents, 5, i, state_distr, alpha=10e-6)
        agents.append(local_agent)
    trans_ind = 0
    eps_rewards = []
    for ep in range(num_episodes):
        if ep % 25 == 0 and ep > 0:
            print(f"Episode: {ep}")
            print(np.mean(np.array(eps_rewards[ep-25:])))
        state, _ = env.reset()
        ep_rew = 0
        discount = 1
        for _ in range(50):
            actions_all = {}
            probs_all = {}
            vals_all = {}
            for i in range(n_agents):
                action, probs, value, _ = agents[i].choose_action(state[f"agent_{i}"])
                actions_all[f"agent_{i}"] = action
                probs_all[f"agent_{i}"] = probs
                vals_all[f"agent_{i}"] = value
            state_n, reward, done, _, _ = env.step(actions_all)
            
            for i in range(n_agents):
                ep_rew +=  reward[f"agent_{i}"]
                agents[i].observe(actions_all[f"agent_{i}"], probs_all[f"agent_{i}"], vals_all[f"agent_{i}"], reward[f"agent_{i}"], done[f"agent_{i}"], trans_ind)
            if eval:
                env.render()
                time.sleep(0.5)
                print(ep_rew)
            trans_ind += 1
            state = state_n
            if distance_agents(state["agent_0"]) < 0.75:
                Local_Agent.communicate(agents, n_agents, Coop_Nav_State_Distribution)
        if not eval:
            for agent in agents:
                agent.learn()
        eps_rewards.append(ep_rew)
        if ep % 500 == 0:
            score_history.append(ep_rew)
            episode_history.append(ep)
    if not eval:
        agent.save_models()

    plt.figure(figsize=(10, 5))
    # episode_history, score_history = episode_history[::1000], score_history[::1000]
    plt.plot(episode_history, score_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward vs Episodes')
    plt.grid(True)
    plt.savefig('reward_vs_episodes_simple_spread.png')
    data = {'Episodes': episode_history, 'Reward': score_history}
    df = pd.DataFrame(data)
    df.to_csv('reward_vs_episodes_simple_spread.csv', index=False)

def train_cat_mouse_local(num_episodes, eval = False):
    episode_history = []
    score_history = []
    n_agents = 2
    n_mice = n_agents
    render_mode = None
    if eval:
        render_mode = "human"
    env = CatMouseMA(n_agents=n_agents, n_prey=n_mice) #Tr/Te
    state_dim = 4*n_agents+5*n_mice
    agents = []
    for i in range(n_agents):
        state_distr = Cat_Mouse_State_Distribution(n_agents, n_mice, i)
        local_agent = Local_Agent((state_dim, ), n_agents, 5, i, state_distr, alpha=10e-6)
        agents.append(local_agent)
    trans_ind = 0
    eps_rewards = []
    for ep in range(num_episodes):
        if ep % 25 == 0 and ep > 0:
            print(f"Episode: {ep}")
            print(np.mean(np.array(eps_rewards[ep-25:])))
        state, _ = env.reset()
        obs, com = env._get_obs()
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
            state_n, reward, done, _, _ = env.step(actions_all)
            obs_n, com_n = env._get_obs()
            for i in range(n_agents):
                ep_rew +=  reward[i]
                agents[i].observe(actions_all[i], probs_all[i], vals_all[i], reward[i], done, trans_ind)
            if eval:
                env.render()
                time.sleep(0.5)
                print(ep_rew)
            trans_ind += 1
            state = state_n
            obs = obs_n
            Local_Agent.communicate(agents, n_agents, Cat_Mouse_State_Distribution)
        if not eval:
            for agent in agents:
                agent.learn()
        eps_rewards.append(ep_rew)
        if ep % 500 == 0:
            score_history.append(ep_rew)
            episode_history.append(ep)
    if not eval:
        agent.save_models()

    plt.figure(figsize=(10, 5))
    # episode_history, score_history = episode_history[::1000], score_history[::1000]
    plt.plot(episode_history, score_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward vs Episodes')
    plt.grid(True)
    plt.savefig('reward_vs_episodes_simple_spread.png')
    data = {'Episodes': episode_history, 'Reward': score_history}
    df = pd.DataFrame(data)
    df.to_csv('reward_vs_episodes_simple_spread.csv', index=False)


def train_lumberjacks_local(num_episodes, eval = False):

    n_agents = 1
    n_trees = 1
    render_mode = None
    if eval:
        render_mode = "human"
    env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(4, 4), n_agents=n_agents, n_trees=n_trees) #Tr/Te
    state_dim = 10
    agents = []
    for i in range(n_agents):
        state_distr = Lumberjacks_State_Distribution(n_agents, n_trees, i)
        local_agent = Local_Agent((state_dim, ), n_agents, 5, i, state_distr, alpha=10e-6)
        agents.append(local_agent)
    trans_ind = 0
    eps_rewards = []
    for ep in range(num_episodes):
        if ep % 25 == 0 and ep > 0:
            print(f"Episode: {ep}")
            print(np.mean(np.array(eps_rewards[ep-25:])))
        obs = env.reset()
        ep_rew = 0
        discount = 1
        for _ in range(50):
            actions_all = []
            probs_all = []
            vals_all = []
            for i in range(n_agents):
                action, probs, value, _ = agents[i].choose_action(obs[i])
                actions_all.append(action) 
                probs_all.append(probs)
                vals_all.append(value)
            state_n, reward, done, _, _ = env.step(actions_all)
            obs_n, com_n = env._get_obs()
            for i in range(n_agents):
                ep_rew +=  reward[i]
                agents[i].observe(actions_all[i], probs_all[i], vals_all[i], reward[i], done, trans_ind)
            if eval:
                env.render()
                time.sleep(0.5)
                print(ep_rew)
            trans_ind += 1
            state = state_n
            obs = obs_n
            Local_Agent.communicate(agents, n_agents, Lumberjacks_State_Distribution)
        if not eval:
            for agent in agents:
                agent.learn()
        eps_rewards.append(ep_rew)
    if not eval:
        agent.save_models()

def __main__():
    # -5.922 after 5000 eps equall step size
    train_cooperative_nav(5000, True)

__main__()