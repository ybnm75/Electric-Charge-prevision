import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
data_path = r"C:\Users\pc\Downloads\tp final khadir\sorted_train.csv"  
# Replace "your_data_path.csv" with your actual data path
df = pd.read_csv(data_path)

# Select features for clustering
hour_features = ['t-1', 't-168', 't-2', 't-24', 't-3']
X = df[hour_features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the range of K values to try
k_values = range(1, 11)  # Try K values from 1 to 10

# Initialize a list to store the quantization error for different values of K
quantization_errors = []

# Iterate over each value of K
for k in k_values:
    # Initialize SOM with the current value of K
    som = MiniSom(k, k, X_scaled.shape[1], sigma=0.3, learning_rate=0.5)
    # Train the SOM
    som.train_batch(X_scaled, 1000)  # Adjust the number of epochs as needed
    # Calculate the quantization error
    quantization_error = np.mean(np.linalg.norm(X_scaled - som.quantization(X_scaled), axis=1))
    # Append the quantization error to the list
    quantization_errors.append(quantization_error)

# Plot the elbow curve
plt.plot(k_values, quantization_errors, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for SOM')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Quantization Error')
plt.xticks(k_values)
plt.show()
