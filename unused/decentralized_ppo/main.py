import gym
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import time
from pettingzoo.mpe import simple_spread_v3
from agent import Centralized_PPO_Agent


def state_to_array_lumber(state):
        state_list = []
        for agent in state[0]:
            state_list.append(agent[0])
            state_list.append(agent[1])
        for tree in state[1]:
            state_list.append(tree[0][0])
            state_list.append(tree[0][1])
            state_list.append(tree[1])
        return state_list

def train_lumberjacks(episodes, eval=False):
    n_trees = 4
    n_agents = 2
    n_actions = 5
    state_dim = 2*n_agents+3*n_trees
    grid_dim = 5
    agent = Centralized_PPO_Agent(n_actions, n_agents, (state_dim,), alpha=0.00003, n_epochs=4, batch_size=16) #22*n_agents
    env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(grid_dim, grid_dim), n_trees=n_trees, n_agents=n_agents)
    scores = []
    for ep in range(episodes):
        if ep % 25 == 0 and ep > 0 and not eval:
            print(f"Episode: {ep}")
            print(np.mean(np.array(scores[ep-25:])))
        state = env.reset()
        state = state_to_array_lumber(env.get_global_obs())
        score = 0
        discount = 1
        steps = 0
        for i in range(100):
            action, probs, value, _ = agent.choose_action(state)
            state_n, reward, done, info = env.step(action)
            state_n = state_to_array_lumber(env.get_global_obs())
            rewards = 0
            done_all = True
            probs_all = 0
            for i in range(n_agents):
                done_all = done_all and done[i]
                rewards += reward[i]
                probs_all += probs[i]
            agent.remember(state_n, action, probs_all, value, rewards, done_all)
            score = score + discount*reward
            discount = 0.99*discount
            if np.all(done):
                break
            steps += 1
        scores.append(score)
        agent.learn()
    agent.save_models()
    """state = env.reset()
    for i in range(25):
        action, prob, val, _ = agent.choose_action(state)
        observation_, reward, done, info = env.step([action])
        if np.all(done):
            break
        env.render()
        time.sleep(0.25)
    env.close()"""

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


def train_navigation(num_episodes, eval = False):
    n_agents = 1
    n_targets = n_agents
    render_mode = None
    if eval:
        render_mode = "human"
    env = simple_spread_v3.parallel_env(N=n_agents, max_cycles=50,local_ratio=0.5, continuous_actions=False, render_mode=render_mode) #Tr/Te
    state_dim = 2*(n_agents+n_targets)
    agent = Centralized_PPO_Agent(n_actions=5, n_agents=n_agents, input_dims=(state_dim,), train = not eval)
    trans_ind = 0
    eps_rewards = []
    for ep in range(num_episodes):
        if ep % 25 == 0 and ep > 0 and not eval:
            print(f"Episode: {ep}")
            print(np.mean(np.array(eps_rewards[ep-25:])))
        state, _ = env.reset()
        state = global_state_navigation(state, n_agents)
        ep_rew = 0
        discount = 1
        for _ in range(50):
            actions_all = {}
            actions, probs, values, _ = agent.choose_action(state)
            done_all = True
            for i in range(n_agents):
                actions_all[f"agent_{i}"] = actions[i]
            state_n, reward, done, _, _ = env.step(actions_all)
            rewards = 0
            probs_all = 0
            for i in range(n_agents):
                done_all = done_all and done[f"agent_{i}"]
                rewards += reward[f"agent_{i}"]
                probs_all += probs[i]
            state_n = global_state_navigation(state_n, n_agents)
            ep_rew += discount*rewards
            discount *= 0.99
            agent.remember(state, actions, probs_all, values, rewards, done_all)
            if eval:
                env.render()
                time.sleep(1)
                print(probs)
            trans_ind += 1
            state = state_n
        eps_rewards.append(ep_rew)
    if not eval:
        agent.save_models()


def __main__():
   train_lumberjacks(5000, eval=False)
            


            
      
__main__()