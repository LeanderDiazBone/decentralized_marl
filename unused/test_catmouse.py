from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_discrete import *

env = CatMouseMAD()

state, info = env.reset()
for _ in range(100):
    
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    env.render()

env.close()
    
    