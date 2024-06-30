import decentralized_agent
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
import sys
from tqdm import tqdm
import time
import warnings
import torch
warnings.filterwarnings("ignore")
sys.path.append("../")
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *
import gym

class Decentralized:

    def __init__(self, state_dim, action_dim, num_agents, env, discount, env_name = None):
        self.num_agents = num_agents
        self.agents = []
        self.env = env
        self.env_name = env_name
        self.discount = discount
        for i in range(num_agents):
            agent_i = decentralized_agent.Agent(state_dim=state_dim, action_dim=action_dim, num_agents=num_agents, agent_id=i, mu_0=0, beta=1e-2, discount=self.discount)
            self.agents.append(agent_i)

    def get_actions(self, s):
        actions = []
        for ag in self.agents:
            a_i = ag.get_action(s)[0]
            actions.append(a_i)
        return actions
    
    def get_critic_values(self, s_a):
        critic_values = []
        for ag in self.agents:
            c_i = ag.critic(s_a)
            critic_values.append(c_i)
        return critic_values
    
    def state_to_array(self, state):
        state_list = []
        for tuple in state:
            for i in range(len(tuple)):
                state_list.append(tuple[i])
        return state_list
    
    def state_to_array_lumber(self, state):
        state_list = []
        for agent in state[0]:
            state_list.append(agent[0])
            state_list.append(agent[1])
        for tree in state[1]:
            state_list.append(tree[0][0])
            state_list.append(tree[0][1])
            state_list.append(tree[1])
        return state_list

    def save(self, path):
        for i in range(self.num_agents):
            torch.save(self.agents[i].actor.network.state_dict(), path+f"agent_{i}_actor")
            torch.save(self.agents[i].critic.network.state_dict(), path+f"agent_{i}_critic")

    def load(self, path):
        for i in range(self.num_agents):
            self.agents[i].actor.network.load_state_dict(torch.load(path+f"agent_{i}_actor"))
            self.agents[i].critic.network.load_state_dict(torch.load(path+f"agent_{i}_critic"))

    def train(self, num_episodes = 100, num_iterations = 25):
        average_reward = []
        for ep in tqdm(range(num_episodes)):
            reward_ep = 0
            dicount = self.discount
            self.env.reset()
            s_t = self.state_to_array_lumber(self.env.get_global_obs())
            a_t = self.get_actions(s_t)
            #for i in range(self.num_agents):
            #    self.agents[i].mu = 0
            
            for t in range(num_iterations):
                next_state, reward, terminated, info = self.env.step(a_t)
                next_state = self.state_to_array_lumber(self.env.get_global_obs())

                if np.all(np.array(terminated)):
                    break
                for i in range(self.num_agents):
                    self.agents[i].update_mu(reward[i])
                next_actions = self.get_actions(next_state)
                for i in range(self.num_agents):
                    self.agents[i].update_actor(next_state, next_actions)
                    self.agents[i].update_critic(s_t, a_t, next_state, next_actions, reward[i])
                s_t = next_state
                a_t = next_actions
                # Communication update
                """con = self.env._get_connections()
                for i in range(self.num_agents):
                    omega_i = []
                    for j in con[i]:
                        omega_i.append(self.agents[j].critic)
                    self.agents[i].set_omega(omega_i)
                for i in range(self.num_agents):
                    self.agents[i].update_omega()"""
                # Store commulative error
                reward_all = 0
                for i in range(self.num_agents):
                    reward_all += reward[i]
                reward_ep += dicount*reward_all
                dicount *= self.discount
            average_reward.append(reward_ep)
        return average_reward
        
def train_cat_mouse():
    n_agents = 2
    n_mice = 4
    env = CatMouse(n_agents=n_agents, n_mice=n_mice) #render_mode='human'
    decentralized = Decentralized(state_dim=2*n_agents+3*n_mice, action_dim=1, num_agents=n_agents, env=env)
    decentralized.train(num_iterations=1000, num_episodes=100)

def train_coorperative_navigation():
    env = simple_spread_v3.parallel_env(N=3, max_cycles=100, local_ratio=0.5,  continuous_actions=False)
	#env.reset(seed=42)

def train_lumberjack():
    n_agents = 1
    n_trees = 1
    env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(3, 3), n_agents=n_agents, n_trees=n_trees)
    env.reset()
    decentralized = Decentralized(state_dim=2*n_agents+3*n_trees, action_dim=5, num_agents=n_agents, env=env, discount = 0.5)
    #average_reward = decentralized.train(num_iterations=10, num_episodes=10000)
    #decentralized.save('ckp/')
    #plt.plot(average_reward)
    #plt.savefig("Plots/average_reward.png")
    decentralized.load('ckp/')
    
    """state_test = [[(0,0)], [((0,1), 1)]]
    for ac in range(5):
        print(decentralized.get_critic_values(decentralized.state_to_array_lumber(state_test)+[ac]))"""
    for ep in range(100): 
        env.reset()
        state = env.get_global_obs()
        for i in range(10):
            action = decentralized.get_actions(decentralized.state_to_array_lumber(state))
            print('')
            for j in range(5):
                print(decentralized.get_critic_values(decentralized.state_to_array_lumber(state)+[j]))
            agent_obs, rewards, terminated, info = env.step(action)
            state = env.get_global_obs()
            env.render()
            time.sleep(1)
            if terminated[0]:
                break
            

def __main__():
    train_lumberjack()

__main__()