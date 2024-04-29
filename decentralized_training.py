import decentralized_agent
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *

class Decentralized:

    def __init__(self, state_dim, action_dim, num_agents):
        self.num_agents = num_agents
        self.agents = []
        for i in range(num_agents):
            agent_i = decentralized_agent.Agent(state_dim=state_dim, action_dim=action_dim, num_agents=num_agents, agent_id=i, mu_0=None, beta=None)
            self.agents.append(agent_i)

    def get_actions(self, s):
        actions = []
        for ag in self.agents:
            a_i = ag.get_action(s)[0]
            print(a_i)
            actions.append(a_i)
        return actions
    
    def get_critic_values(self, s_a):
        critic_values = []
        for ag in self.agents:
            c_i = ag.critic(s_a)
            critic_values.append(c_i)
        return critic_values
        
        
def state_to_array(state):
    state_list = []
    for tuple in state:
        for i in range(len(tuple)):
            state_list.append(tuple[i])
    return state_list

def __main__():

    num_iterations = 25
    n_agents = 2
    n_mice = 4
    env = CatMouse(n_agents=n_agents, n_mice=n_mice)
    decentralized = Decentralized(state_dim=2*n_agents+3*n_mice, action_dim=1, num_agents=n_agents)

    # Start Training
    s_t = env._get_obs()
    s_t = state_to_array(s_t[0]+s_t[1])
    a_t = decentralized.get_actions(s_t)
    for t in range(num_iterations):
        print(t)
        next_state, reward, terminated, info = env.step(a_t)
        next_state = state_to_array(next_state[0]+next_state[1])
        """
        for i in range(n_agents):
            decentralized.agents[i].update_mu(rs[i])
        """
        next_actions = []
        next_actions = decentralized.get_actions(next_state)
        #print(next_actions)
        #s_a = next_state+next_actions
        #critic_values = decentralized.get_critic_values(s_a)
        #print(critic_values)
        for i in range(n_agents):
            # Change to local reward
            decentralized.agents[i].update_actor(next_state, next_actions)
            #decentralized.agents[i].update_critic(s_tn, a_tn, s_tn, a_tn, reward)
        """
        # Communication update
        con = env.get_connections(s_tn)

        for i in range(n_agents):
            omega_i = decentralized.agents[i].get_omega()
            for j in range(con[i]):
                omega_i += decentralized.agents[j].get_omega()
            decentralized.agents[i].set_omega(omega_i/(len(con[i])+1))
        """
        s_t = next_state
        a_t = next_actions
        

__main__()