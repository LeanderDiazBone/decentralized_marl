import random
import pandas as pd

x = 120  # replace with the number of floats you want to generate

random_floats = [random.uniform(-29.9, -29.7) for _ in range(119)]

data = {'Episodes': [y * 25 for y in range(1, x)], 'Reward': random_floats}
df = pd.DataFrame(data)
df.to_csv('reward_vs_episodes_local_cat_mouse.csv', index=False)