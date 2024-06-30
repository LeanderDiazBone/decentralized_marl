import gym
from gym import spaces
from marl_gym.marl_gym.envs.utils.utils import *  # maybe need to change if made to package
import numpy as np
import math
import pygame
import copy
import random

class RewardNormalizer:

    def __init__(self):
        self.min_reward = float('inf')
        self.max_reward = float('-inf')

    def update(self, reward):
        # Update min and max rewards
        if reward < self.min_reward:
            self.min_reward = reward
        if reward > self.max_reward:
            self.max_reward = reward

    def normalize(self, reward):
        # Normalize the reward to [-1, 1]
        if self.max_reward > self.min_reward:  # Ensure there's a range to normalize
            normalized_reward = 2 * (reward - self.min_reward) / (self.max_reward - self.min_reward) - 1
        else:
            normalized_reward = 0  # If all rewards are the same, normalize to 0
        return normalized_reward

class CatMouseMAD(gym.Env):

    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, grid_size: int = 5,max_iter: int = None, n_agents: int = 2, n_prey: int = 4,
                observation_radius: int = 1, communication_radius: int = 1,
                step_cost: float = -1, window_size: int = 250):
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
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.n_agents = n_agents
        self.n_prey = n_prey
        self.observation_radius = observation_radius
        self.communication_radius = communication_radius
        self.step_cost = step_cost
        self.window_size = window_size
        self.window = None
        self.clock = None
        self.steps = 0
        self.normalizer = RewardNormalizer()

        self.action_space = MultiAgentActionSpace([spaces.Discrete(9)] * self.n_agents)
        obs_size = 2*self.observation_radius + 1
        self.observation_space = MultiAgentObservationSpace([spaces.Dict({
            "agent_grid": spaces.MultiDiscrete([obs_size, obs_size]),
            "prey_grid": spaces.MultiDiscrete([obs_size, obs_size]),
            "agent_pos": spaces.MultiDiscrete(2),
            "agent_id": spaces.Discrete(self.n_agents)
        }) for _ in range(self.n_agents)])

        self.reset()

    def get_obs_belief(self):
        # state for local belief algo
        agent_pos_norm = np.copy(self.agent_pos)
        """for i in range(self.n_agents):
            agent_pos_norm[i][0] /= self.grid_size
            agent_pos_norm[i][1] /= self.grid_size"""
        prey_pos_norm = np.copy(self.prey_pos)
        """for i in range(self.n_prey):
            prey_pos_norm[i][0] /= self.grid_size
            prey_pos_norm[i][1] /= self.grid_size"""
        return {"agents": agent_pos_norm, "prey": prey_pos_norm}

    def get_global_obs(self) -> dict:
        """
        Returns the global state of the environment.

        :return: Dictionary containing agent positions and prey positions/caught status.
        """
        # lumberjack state
        observation_range = self.observation_radius
        agent_grid = np.zeros((self.n_agents, 2 * observation_range + 1, 2 * observation_range + 1))
        prey_grid = np.zeros((self.n_agents, 2 * observation_range + 1, 2 * observation_range + 1))
        for agent_id in range(self.n_agents):
            cur_pos = self.agent_pos[agent_id]
            for i in range(-observation_range, observation_range+1):
                for j in range(-observation_range, observation_range+1):
                    if 0 <= cur_pos[0] + i < self.grid_size and 0 <= cur_pos[1] + j < self.grid_size:
                        agent_grid[agent_id, i+observation_range, j+observation_range] = self.agents[cur_pos[0] + i, cur_pos[1] + j] / self.n_agents
                        prey_grid[agent_id, i+observation_range, j+observation_range] = self.prey[cur_pos[0] + i, cur_pos[1] + j] / self.n_agents

        agent_pos_norm = np.copy(self.agent_pos)
        for i in range(self.n_agents):
            agent_pos_norm[i][0] /= self.grid_size
            agent_pos_norm[i][1] /= self.grid_size

        return {"grids": {"agents": agent_grid, "prey": prey_grid}, "agent_pos": agent_pos_norm}

        

        # one grid per agent (one-hot) and prey grid as global obs
        agent_grids = np.zeros((self.n_agents, self.grid_size,self.grid_size))
        for i,pos in enumerate(self.agent_pos):
            agent_grids[i][pos[0]][pos[1]] = 1
        return {"grids": {"agents": agent_grids, "prey": self.prey}, "agent_pos": self.agent_pos}
    
        # env state as global obs
        agents_norm = np.copy(self.agents)
        prey_norm = np.copy(self.prey)
        agent_pos_norm = np.copy(self.agent_pos)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                agents_norm[i][j] /= self.n_agents
                prey_norm[i][j] /= self.n_agents

        for i in range(self.n_agents):
            agent_pos_norm[i][0] /= self.grid_size
            agent_pos_norm[i][1] /= self.grid_size
                
        return {"grids":{"agents": agents_norm, "prey": prey_norm}, "agent_pos": agent_pos_norm}

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

            cur_agent_obs["agent_pos"] = self.agent_pos[i]
            cur_agent_obs["agent_id"] = i
            cur_agent_obs["agent_grid"], cur_agent_obs["prey_grid"] = self._get_window(cur_agent_obs["agent_pos"])

            for j in range(self.n_agents):
                if np.all(np.abs(self.agent_pos[j] - self.agent_pos[i]) <= 1):
                    cur_in_comm_range.append(j)

            agent_obs.append(cur_agent_obs)
            communication.append(cur_in_comm_range)
        
        info["comm_partners"] = communication

        return agent_obs, info
    
    def _get_window(self, cur_pos):
        window_size = 2 * self.observation_radius + 1
        window_agents = np.zeros((window_size,window_size), dtype=int)
        window_prey = np.zeros((window_size,window_size), dtype=int)
        start = cur_pos - self.observation_radius
        end = cur_pos + self.observation_radius
        for wi, i in enumerate(range(start[0],end[0]+1)):
            for wj, j in enumerate(range(start[1],end[1]+1)):
                if i < 0 or i >= self.grid_size or j < 0 or j >= self.grid_size:
                    continue
                window_agents[wi][wj] = self.agents[i][j]
                window_prey[wi][wj] = self.prey[i][j]
        
        return window_agents, window_prey

    def reset(self, seed: int = None):
        np.random.seed(seed)
        self.agent_pos = np.zeros((self.n_agents,2), dtype=int)
        self.agents = np.zeros((self.grid_size,self.grid_size), dtype=int)
        for i in range(self.n_agents):
            pos_x = np.random.randint(low=0, high=self.grid_size)
            pos_y = np.random.randint(low=0, high=self.grid_size)
            self.agents[pos_x][pos_y] += 1
            self.agent_pos[i] = np.array([pos_x, pos_y])

        self.prey = np.zeros((self.grid_size,self.grid_size), dtype=int)
        self.prey_pos = np.zeros((self.n_prey,3), dtype=int)
        for i in range(self.n_prey):
            pos_x = np.random.randint(low=0, high=self.grid_size)
            pos_y = np.random.randint(low=0, high=self.grid_size)
            self.prey[pos_x][pos_y] += 1
            self.prey_pos[i] = np.array([pos_x, pos_y, 0])
        return self._get_obs()

    def step(self, action: list) -> tuple:
        if len(action.shape) == 1:
            action = np.expand_dims(action, axis=0)
        assert len(action) == self.n_agents, "action length {} should be number of agents {}".format(len(action), self.n_agents)

        next_state = None
        reward = []
        terminated = False
        
        # assume uniform random movement of mouse rn, move each mouse
        self._move_prey()
        # move each agent according to action array
        collision = self._move_agents(action)
        
        caught = self._check_caught()

        next_state, info = self._get_obs()

        reward = self._calc_reward(caught, collision)
        
        terminated = not np.any(self.prey)

        truncated = False

        self.steps =+ 1
        if self.max_iter:
            truncated = self.steps < self.max_iter

        return next_state, reward, terminated, truncated, info
    
    def _move_agents(self, action):
        """
        Moves agents' positions according to action.
        :param action: np array of shape (self.n_agent,)
        """
        collision = np.zeros((self.n_agents,),dtype=int)
        for i,a in enumerate(action):
            # cur_action = ACTION_LIST[a]
            cur_action = a
            self.agents[self.agent_pos[i][0],self.agent_pos[i][1]] -= 1
            clipped = np.clip(self.agent_pos[i] + cur_action, 0, self.grid_size-1)
            if ~np.array_equal(clipped,self.agent_pos[i] + cur_action):
                collision[i] = 1
            self.agent_pos[i] = clipped
            self.agents[self.agent_pos[i][0],self.agent_pos[i][1]] += 1
        return collision
    
    def _move_prey(self):
        """
        Moves prey's positions according to their specified behavior
        """
        # prey moves away from agents
        prey_new = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.prey[i][j] > 0:
                    found = False
                    for x in range(-self.observation_radius,self.observation_radius+1):
                        for y in range(-self.observation_radius,self.observation_radius+1):
                            if x == 0 and y == 0:
                                continue
                            if 0 <= i + x < self.grid_size and 0 <= j + y < self.grid_size:
                                if self.agents[i+x][j+y] > 0:
                                    # run away
                                    new_pos = [i,j]
                                    if 0 <= i - x < self.grid_size and 0 <= j - y < self.grid_size:
                                        prey_new[i-x][j-y] += self.prey[i][j]
                                        new_pos = [i-x,j-y]
                                    else:
                                        # corner case
                                        if (i - x >= self.grid_size or i - x < 0) and (j-y >= self.grid_size or j-y < 0):
                                            prey_new[i][j] += self.prey[i][j]
                                        elif i - x >= self.grid_size or i - x < 0:
                                            if y == 0:
                                                if 0 <= j - 1 < self.grid_size:
                                                    prey_new[i][j-1] += self.prey[i][j]
                                                    new_pos = [i,j-1]
                                                else:
                                                    prey_new[i][j+1] += self.prey[i][j]
                                                    new_pos = [i,j+1]
                                            else:
                                                prey_new[i][j-y] += self.prey[i][j]
                                                new_pos = [i,j-y]
                                        elif j - y >= self.grid_size or j - y < 0:
                                            if x == 0:
                                                if 0 <= i - 1 < self.grid_size:
                                                    prey_new[i-1][j] += self.prey[i-1][j]
                                                    new_pos = [i-1,j]
                                                else:
                                                    prey_new[i+1][j] += self.prey[i+1][j]
                                                    new_pos = [i+1,j]
                                            else:
                                                prey_new[i-x][j] += self.prey[i][j]
                                                new_pos = [i-x,j]

                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    # if not in vicinity of cat, either random movement or stays still
                    if not found:
                        sample = random.randint(0,8)
                        dir = ACTION_LIST[sample]
                        if 0 <= i + dir[0] < self.grid_size and 0 <= j + dir[1] < self.grid_size:
                            prey_new[i+dir[0]][j+dir[1]] += self.prey[i][j]
                            new_pos = [i+dir[0],j+dir[1]]
                        else:
                            prey_new[i][j] += self.prey[i][j]
                            new_pos = [i,j]

                    for prey_p in self.prey_pos:
                        if i == prey_p[0] and j == prey_p[1]:
                            prey_p[0] = new_pos[0]
                            prey_p[1] = new_pos[1]

        self.prey = prey_new
        

    def _check_caught(self):
        """
        Check if in current environment state an agent can catch a prey and update accordingly.
        """
        caught = np.zeros(self.n_agents, dtype=int)
        for i in range(self.grid_size):
            for j  in range(self.grid_size):
                if self.agents[i][j] > 0 and self.prey[i][j] > 0:
                    # if self.agents[i][j] >= self.prey[i][j]:
                    for a,p in enumerate(self.agent_pos):
                        if p[0] == i and p[1] == j:
                            caught[a] += self.prey[i][j]
                    for prey_p in self.prey_pos:
                        if prey_p[0] == i and prey_p[1] == j:
                            prey_p[2] = 1
                    self.prey[i][j] = 0
        return caught
    
    def _calc_reward(self, caught, collision):
        """
        Calculates reward based on current environment state.
        :return: reward score
        """
        reward = np.full((self.n_agents,),self.step_cost)
        reward += caught * 10 * self.n_agents
        # reward -= collision
        # for r in reward:
        #     self.normalizer.update(r) 
        # normalized_reward = [self.normalizer.normalize(r) for r in reward]

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

        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels
        # First we draw the target
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = ((i+0.5)*pix_square_size, (j+0.5)*pix_square_size)
                if self.prey[i][j]:
                    pygame.draw.circle(
                        canvas,
                        (255, 0, 0),
                        pos,
                        pix_square_size / 3
                    )
                    num = my_font.render(str(self.prey[i][j]), True, (0,0,0))
                    canvas.blit(num, pos)

                if self.agents[i][j]:
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 255),
                        pos,
                        pix_square_size / 3,
                    )
                    num = my_font.render(str(self.agents[i][j]), True, 0)
                    canvas.blit(num, pos)
        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # The following line copies our drawings from `canvas` to the visible window
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

ACTION_LIST = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
    [0,-1],
    [-1,0],
    [-1,-1],
    [1,-1],
    [-1,1]
])