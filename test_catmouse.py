from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *

env = CatMouse()

state = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    next_state, reward, terminated, info = env.step(action)
    state = next_state
    env.render()

env.close()
    
    