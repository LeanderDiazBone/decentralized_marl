import gym
from gym import spaces
from marl_gym.marl_gym.envs.utils.utils import *  # maybe need to change if made to package
import numpy as np
import math
import pygame
import copy

class CatMouseMA(gym.Env):

    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, max_iter: int = None, n_agents: int = 2, n_prey: int = 4, step_size: float = 0.05,
                entity_size: float = 0.05, observation_radius: float = 0.2, communication_radius: float = 0.2,
                step_cost: float = -0.1, window_size: int = 250):
        """
        Initialize the environment.

        Args:
        max_iter (int, optional): Maximum number of steps per episode. Defaults to None (no limit).
        n_agents (int): Number of agents in the environment. Defaults to 2.
        n_prey (int): Number of prey in the environment. Defaults to 4.
        step_size (float): Step size for agent movement. Defaults to 0.05.
        entity_size (float): Size of agents and prey (radius). Defaults to 0.05.
        observation_radius (float): Observation radius for agents. Defaults to 0.2.
        communication_radius (float): Communication radius for agents. Defaults to 0.2.
        step_cost (float): Reward penalty for each step. Defaults to -0.1.
        window_size (int): Size of the rendering window. Defaults to 250.
        """

        self.max_iter = max_iter
        self.n_agents = n_agents
        self.n_prey = n_prey
        self.step_size = step_size
        self.entity_size = entity_size
        self.catch_range = 2 * self.entity_size
        self.observation_radius = observation_radius
        self.communication_radius = communication_radius
        self.step_cost = step_cost
        self.window_size = window_size
        self.window = None
        self.clock = None
        self.steps = 0

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_agents,), dtype=np.float32)
        self.observation_space = MultiAgentObservationSpace([spaces.Dict({
            "agents": spaces.Dict({
            "position": spaces.Box(low=0, high=1, shape=(self.n_agents, 2)),
            "cur_agent": spaces.MultiBinary(self.n_agents),
            }),
            "prey": spaces.Dict({
            "position": spaces.Box(low=0, high=1, shape=(self.n_prey, 2)),
            "caught": spaces.MultiBinary(self.n_prey),
            })
        }) for _ in range(self.n_agents)])

        # List to store prey each agent observed getting caught
        self.agent_prey_list = [[] for _ in range(self.n_agents)]

        self.reset()

    def get_global_obs(self) -> dict:
        """
        Returns the global state of the environment.

        :return: Dictionary containing agent positions and prey positions/caught status.
        """
        return {"agents": self.agents, "prey": self.prey}

    def _get_obs(self):
        """
        Turns environment state into local observation state. Each agent's observation contains positions of other agent's and prey.
        If either are outside the observation radius, their position is set to (-1,-1). The cur_agent flag is set to True, if the
        agent matches the observation number.
        :return: Observation space according to self.observation_space
        """
        info = {}
        agent_obs = []
        communication = []
        for i in range(self.n_agents):

            cur_agent_obs = {}
            cur_in_comm_range = []
            
            cur_agent_agent_obs = copy.deepcopy(self.agents)
            cur_agent_agent_obs["cur_agent"] = np.zeros(self.n_agents)
            for j in range(self.n_agents):

                if i == j:
                    cur_agent_agent_obs["cur_agent"][j] = 1
                if not self.agent_agent_obs_matrix[i][j]:
                    cur_agent_agent_obs["position"][j][0] = -1
                    cur_agent_agent_obs["position"][j][1] = -1
                if self.agent_agent_comm_matrix[i][j]:
                    cur_in_comm_range.append(j)

            cur_agent_prey_obs = copy.deepcopy(self.prey)
            for j in range(self.n_prey):
                # set mouse caught flag to 1 to indicate it observed mouse getting caught
                obs_caught = int(j in self.agent_prey_list[i])

                if not self.agent_mouse_obs_matrix[i][j]:
                    cur_agent_prey_obs["position"][j][0] = -1
                    cur_agent_prey_obs["position"][j][1] = -1

                cur_agent_prey_obs["caught"][j] = obs_caught

            cur_agent_obs["agents"] = cur_agent_agent_obs
            cur_agent_obs["prey"] = cur_agent_prey_obs
            agent_obs.append(cur_agent_obs)
            communication.append(cur_in_comm_range)
        
        info["comm_partners"] = communication

        return agent_obs, info

    def reset(self, seed: int = None):
        np.random.seed(seed)
        self.agents = {"position": np.random.rand(self.n_agents,2)}
        self.prey = {"position": np.random.rand(self.n_prey,2), "caught": np.zeros(self.n_prey)}
        # need to calculate matrices for get_obs
        agent_agent_dists = self._calc_dists(self.agents["position"], self.agents["position"])
        self.agent_prey_dists = self._calc_dists(self.agents["position"], self.prey["position"])
        self.agent_agent_obs_matrix = self._calc_in_range_matrix(agent_agent_dists, self.observation_radius)
        self.agent_mouse_obs_matrix = self._calc_in_range_matrix(self.agent_prey_dists, self.observation_radius)
        self.agent_agent_comm_matrix = self._calc_in_range_matrix(agent_agent_dists, self.communication_radius)
        return self._get_obs()

    def step(self, action: list) -> tuple:
        assert len(action) == self.n_agents, "action length should be number of agents"

        next_state = None
        reward = []
        terminated = False

        # move each agent according to action array
        self._move_agents(action)
        # assume uniform random movement of mouse rn, move each mouse
        self._move_prey()

        # calculate boolean matrices indicating whether in catch/observation/communication range
        agent_agent_dists = self._calc_dists(self.agents["position"], self.agents["position"])
        self.agent_prey_dists = self._calc_dists(self.agents["position"], self.prey["position"])
        self.agent_prey_caught = self._calc_in_range_matrix(self.agent_prey_dists, self.catch_range)
        self.agent_agent_obs_matrix = self._calc_in_range_matrix(agent_agent_dists, self.observation_radius)
        self.agent_mouse_obs_matrix = self._calc_in_range_matrix(self.agent_prey_dists, self.observation_radius)
        self.agent_agent_comm_matrix = self._calc_in_range_matrix(agent_agent_dists, self.communication_radius)


        next_state, info = self._get_obs()

        reward = self._calc_reward()
        reward += (np.count_nonzero(self.prey["caught"] == 1) ** 4) * 100 * self._check_caught()
        
        terminated = np.all(self.prey["caught"])

        truncated = False

        self.steps =+ 1
        if self.max_iter:
            truncated = self.steps < self.max_iter

        return next_state, reward, terminated, truncated, info
    
    def _calc_in_range_matrix(self, dists, range):
        """
        Calculates a matrix containing boolean values indicating wether the entities of the two given
        lists (length N and M respecitvely) are in range (euclidean distance) of each other or not.
        :param dists: numpy array of shape (N,M), containing euclidean distances between entities
        :param range: int denoting range
        :return: numpy array of shape (N,M), each entry indicates whether entity of list1 is in range
                 of entity of list2
        """
        x = dists < range
        return x
    
    def _calc_dists(self, list1, list2):
        """
        Calculates euclidean distances between two lists of (x,y) positions.
        :param list1: numpy array of shape (N,2), each entry corresponds to (x,y) position of entity
        :param list2: numpy array of shape (M,2), each entry corresponds to (x,y) position of entity
        :return: numpy array of shape (N,M), each entry indicates the euclidean distance between entity 
                 of list1 and entity of list2
        """
        t1 = list1[:,np.newaxis,:]
        t2 = list2[np.newaxis,:]
        x = t1 - t2
        x = x**2
        x = x[:,:,0] + x[:,:,1]
        x = np.sqrt(x)
        return x
    
    def _move_agents(self, action):
        """
        Moves agents' positions according to action.
        :param action: np array of shape (self.n_agent,)
        """
        for i,a in enumerate(action):
            direction = 2* np.pi * a
            move_x = self.step_size * math.cos(direction)
            move_y = self.step_size * math.sin(direction)
            cur_x, cur_y = self.agents["position"][i][0], self.agents["position"][i][1]
            self.agents["position"][i][0] = min(max(0,cur_x + move_x),1)
            self.agents["position"][i][1] = max(min(1,cur_y + move_y),0)
    
    def _move_prey(self):
        """
        Moves prey's positions according to their specified behavior
        """
        # assume uniform random movement of prey
        pass
        # for i in range(self.n_prey):
        #     cur_x, cur_y = self.prey["position"][i][0], self.prey["position"][i][1]
        #     if self.prey["caught"][i]:
        #         continue
        #     direction = 2 * np.pi * np.random.uniform()
        #     move_x = self.step_size * math.cos(direction)
        #     move_y = self.step_size * math.sin(direction)
            
        #     self.prey["position"][i][0] = min(max(0,cur_x + move_x),1)
        #     self.prey["position"][i][1] = max(min(1,cur_y + move_y),0)

    def _check_caught(self):
        """
        Check if in current environment state an agent can catch a prey and update accordingly.
        """
        mice_caught = [0 for _ in range(self.n_agents)]
        for i in range(self.n_prey):
            if not self.prey["caught"][i]:
                for j in range(self.n_agents):
                    if self.agent_prey_caught[j][i]:
                        for k in range(self.n_agents):
                            # if prey caught, check observation radius to mark all agents that saw it getting caught
                            if self.agent_mouse_obs_matrix[k][i]:
                                self.agent_prey_list[k].append(i)
                        self.prey["caught"][i] = 1
                        mice_caught[j] += 1
                        break
        return np.array(mice_caught)
    
    def _calc_reward(self):
        """
        Calculates reward based on current environment state.
        :return: reward score
        """
        reward = np.full(self.n_agents,self.step_cost)
        for i in range(self.n_agents):
            min_dist = self.observation_radius
            for j in range(self.n_prey):
                if not self.prey["caught"][j] and self.agent_mouse_obs_matrix[i][j]:
                    min_dist = (min(min_dist, self.agent_prey_dists[i][j]) * (np.count_nonzero(self.prey["caught"] == 0) ** 4))
            reward[i] -= min_dist
        return reward

    def render(self):
        self._render_frame()
        
    def _render_frame(self):
        """
        Render each frame using pygame.
        """   
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        canvas.fill((255, 255, 255))

        for a in self.agents["position"]:
            x,y = a[0],a[1]
            x *= self.window_size
            y *= self.window_size
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (x,y),
                self.entity_size*self.window_size,
            )
            pygame.draw.circle(
                canvas,
                (0,0,0),
                (x,y),
                self.observation_radius*self.window_size,
                width=1
            )
        
        for i,p in enumerate(self.prey["position"]):
            if self.prey["caught"][i]:
                continue
            x,y = p[0],p[1]
            x *= self.window_size
            y *= self.window_size
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (x,y),
                self.entity_size*self.window_size,
            )
            
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
    
    # close rendering
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()