import pandas as pd
import matplotlib.pyplot as plt

# Define the file names and their corresponding colors and labels
files = [
	'ppo_lumberjacks/reward_vs_episodes.csv',
	'mappo_lumberjacks/reward_vs_episodes.csv',
	'qmix_lumberjacks/reward_vs_episodes.csv']
colors = ['red', 'green', 'blue', 'purple']
labels = ['PPO', 'MAPPO', 'MADDPG', 'QMIX']

# Create a new plot
plt.figure()

# Loop over the files
for file, color, label in zip(files, colors, labels):
	# Read the data from the file
	data = pd.read_csv(file)

	# Plot the data with the specified label
	plt.plot(data['Episodes'], data['Reward'], color=color, label=label)

# Add a legend
plt.legend()

# Add labels
plt.xlabel('Episodes')
plt.ylabel('Reward')

# Save the plot as a PNG image
plt.savefig('plot.png')
