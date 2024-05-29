import numpy as np
import pandas as pd
from minisom import MiniSom

# Load the data
data_path = r"C:\Users\pc\Downloads\tp final khadir\sorted_train.csv"
df = pd.read_csv(data_path)

features = df[['t-1','t-2','t-168','t-3','t-24','output','d_0','d_1','d_2','d_3','d_4','d_5','d_6','SD']]

grid_size = (20, 20) 
input_len = features.shape[1] 
sigma = 0.4 
learning_rate = 0.7 
epochs = 200 

som = MiniSom(grid_size[0], grid_size[1], input_len, sigma=sigma, learning_rate=learning_rate)

som.train_random(features.values, epochs)

# Get the cluster assignments for each data point
cluster_labels = np.array([som.winner(x) for x in features.values])

# Get unique winning neuron positions
unique_positions = np.unique(cluster_labels, axis=0)

# Assign cluster indices based on the unique positions
cluster_indices = {tuple(pos): idx for idx, pos in enumerate(unique_positions)}

# Map the winning neuron positions to cluster indices
df['cluster'] = [cluster_indices[tuple(pos)] for pos in cluster_labels]

# Print the number of clusters
num_clusters = len(unique_positions)
print(f"Number of clusters: {num_clusters}")

# Print the DataFrame with the added 'cluster' column
print(df)

# Visualize the SOM grid with data point markers
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Plot the SOM grid

# Add markers for data points
for i, x in enumerate(features.values):
    winner = som.winner(x)
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, 'o', markerfacecolor=plt.cm.rainbow(i / len(features)),
             markeredgecolor='None', markersize=5, alpha=0.5)

# Add cluster boundaries
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        if j < grid_size[1] - 1:
            plt.plot([j + 0.5, j + 1.5], [i + 0.5, i + 0.5], 'k-', linewidth=0.5)
        if i < grid_size[0] - 1:
            plt.plot([j + 0.5, j + 0.5], [i + 0.5, i + 1.5], 'k-', linewidth=0.5)

# Add color legend
plt.colorbar(label='Distance to Neighbors')

plt.title('Self-Organizing Map Clustering')
plt.grid(False)
plt.show()
