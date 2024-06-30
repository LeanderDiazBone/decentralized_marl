import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame

class CatMouse(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, max_iter: int = None, n_agents: int = 2, n_prey: int = 4, step_size: float = 0.05, 
                 entity_size: float = 0.05, step_cost: float = -0.1, window_size: int = 250):

        self.max_iter = max_iter
        self.n_agents = n_agents
        self.n_prey = n_prey
        self.step_size = step_size
        self.entity_size = entity_size
        self.catch_range = 2 * self.entity_size
        self.step_cost = step_cost
        self.window_size = window_size
        self.window = None
        self.clock = None
        self.steps = 0
        # standard gymnasium specifications
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_agents,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "agents": spaces.Dict({
                "position": spaces.Box(low=0,high=1,shape=(self.n_agents,2)),
            }),
            "prey": spaces.Dict({
                "position": spaces.Box(low=0,high=1,shape=(self.n_prey,2)),
                "caught": spaces.MultiBinary(self.n_prey),
            })
        })
        self.reward_range = (-np.inf, 0)
        self.spec = None
        self.reset()

    def _get_obs(self):
        """
        Turns environment state into observation state.
            
        """
        info = {}
        return {"agents": self.agents, "prey": self.prey}, info

    def reset(self, seed=None):
        np.random.seed(seed)
        self.agents = {"position": np.random.rand(self.n_agents,2)}
        self.prey = {"position": np.random.rand(self.n_prey,2), "caught": np.zeros(self.n_prey)}
        return self._get_obs()

    def step(self, action):
        assert len(action) == self.n_agents, "action length should be number of agents"

        next_state = None
        reward = 0
        terminated = False

        # move each agent according to action array
        self._move_agents(action)
        # assume uniform random movement of mouse rn, move each mouse
        self._move_prey()

        self.agent_prey_dists = self._calc_dists(self.agents["position"], self.prey["position"])
        self.agent_prey_caught =  self._calc_in_range_matrix(self.agent_prey_dists, self.catch_range)

        # check which prey got caught, update the env state
        self._check_caught()
        # turn environment state to observation state
        next_state, info = self._get_obs()
        # calculate reward
        reward = self._calc_reward()
        # check if all prey are caught
        terminated = np.all(self.prey["caught"])
        # check if max_iter specified, is True if step limit is reached
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
    
    def _check_caught(self):
        """
        Check if in current environment state an agent can catch a prey and update accordingly.
        """
        for i in range(self.n_prey):
            if not self.prey["caught"][i]:
                for j in range(self.n_agents):
                    if self.agent_prey_caught[j][i]:
                        self.prey["caught"][i] = 1
                        self.caught_in_step += 1
                        break
    
    def _calc_reward(self):
        """
        Calculates reward based on current environment state.
        :return: reward score
        """
        reward = self.step_cost
        for i in range(self.n_agents):
            min_dist = 2
            for j in range(self.n_prey):
                if not self.prey["caught"][j]:
                    min_dist = min(min_dist, self.agent_prey_dists[i][j])
            reward -= min_dist

        reward += self.caught_in_step * 2
        return reward

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
        for i in range(self.n_prey):
            cur_x, cur_y = self.prey["position"][i][0], self.prey["position"][i][1]
            if self.prey["caught"][i]:
                continue
            direction = 2 * np.pi * np.random.uniform()
            move_x = self.step_size * math.cos(direction)
            move_y = self.step_size * math.sin(direction)
            
            self.prey["position"][i][0] = min(max(0,cur_x + move_x),1)
            self.prey["position"][i][1] = max(min(1,cur_y + move_y),0)

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
            x,y = a[0], a[1]
            x *= self.window_size
            y *= self.window_size
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (x,y),
                self.entity_size*self.window_size,
            )
        
        for i,p in enumerate(self.prey["position"]):
            if self.prey["caught"][i]:
                continue
            x,y = p[0], p[1]
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
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
