from agent import Agent
import torch as T
import gym

def state_to_torch_lumber(state):
        state_list = []
        for agent in state[0]:
            state_list.append(agent[0])
            state_list.append(agent[1])
        for tree in state[1]:
            state_list.append(tree[0][0])
            state_list.append(tree[0][1])
            state_list.append(tree[1])
        return T.Tensor(state_list)

def __main__():
    n_agents = 1
    n_trees = 1
    env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(3, 3), n_agents=n_agents, n_trees=n_trees)
    agent = Agent(state_dim=2*n_agents+3*n_trees, action_dim=5)
    num_episodes = 100
    max_steps = 25
    for ep in range(num_episodes):
        env.reset()
        state = state_to_torch_lumber(env.get_global_obs())
        action = agent.choose_action(state)
        for st in range(max_steps):
             env.
             