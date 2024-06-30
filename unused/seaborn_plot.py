# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Read the CSV files
# df_local = pd.read_csv('./reward_vs_episodes_local_simple_spread.csv')
# df_mappo = pd.read_csv('./reward_vs_episodes_mappo_simple_spread_1.csv')
# df_ppo = pd.read_csv('./reward_vs_episodes_ppo_simple_spread.csv')

# # Find the minimum number of rows among the dataframes
# min_length = min(len(df_local), len(df_mappo), len(df_ppo))

# # Function to downsample a dataframe to the minimum length
# def downsample(df, target_length):
#     step = len(df) // target_length
#     return df.iloc[::step].reset_index(drop=True).iloc[:target_length]

# episodes_shortest = df_local['Episodes'][:min_length] if len(df_local) == min_length else \
#                     df_mappo['Episodes'][:min_length] if len(df_mappo) == min_length else \
#                     df_ppo['Episodes'][:min_length]

# # Downsample the dataframes
# df_local_downsampled = downsample(df_local, min_length)
# df_mappo_downsampled = downsample(df_mappo, min_length)
# df_ppo_downsampled = downsample(df_ppo, min_length)

# df_local_downsampled['Episodes'] = episodes_shortest
# df_mappo_downsampled['Episodes'] = episodes_shortest
# df_ppo_downsampled['Episodes'] = episodes_shortest

# # Add a label column for seaborn
# df_local_downsampled['Algorithm'] = 'local'
# df_mappo_downsampled['Algorithm'] = 'mappo'
# df_ppo_downsampled['Algorithm'] = 'ppo'

# # Combine the dataframes
# df_combined = pd.concat([df_local_downsampled, df_mappo_downsampled, df_ppo_downsampled])

# # Plot using seaborn
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_combined, x='Episodes', y='Reward', hue='Algorithm')
# plt.title('Reward vs Episodes for Different Algorithms')
# plt.savefig('simple_spread_plot.png')
# plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV files
# df_local = pd.read_csv('./reward_vs_episodes_local_simple_spread.csv')
df_mappo = pd.read_csv('./reward_vs_episodes_mappo_lumberjacks.csv')
df_ppo = pd.read_csv('./reward_vs_episodes_ppo_lumberjacks.csv')

# Find the minimum number of rows among the dataframes
min_length = min(len(df_mappo), len(df_ppo))

# Function to downsample a dataframe to the minimum length
def downsample(df, target_length):
    step = len(df) // target_length
    return df.iloc[::step].reset_index(drop=True).iloc[:target_length]

episodes_shortest = df_mappo['Episodes'][:min_length] if len(df_mappo) == min_length else \
                    df_ppo['Episodes'][:min_length]

# Downsample the dataframes
df_mappo_downsampled = downsample(df_mappo, min_length)
df_ppo_downsampled = downsample(df_ppo, min_length)

df_mappo_downsampled['Episodes'] = episodes_shortest
df_ppo_downsampled['Episodes'] = episodes_shortest

# Add a label column for seaborn
df_mappo_downsampled['Algorithm'] = 'mappo'
df_ppo_downsampled['Algorithm'] = 'ppo'

# Combine the dataframes
df_combined = pd.concat([df_mappo_downsampled, df_ppo_downsampled])

# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_combined, x='Episodes', y='Reward', hue='Algorithm')
plt.title('Reward vs Episodes for Different Algorithms')
plt.savefig('simple_spread_plot.png')
plt.show()