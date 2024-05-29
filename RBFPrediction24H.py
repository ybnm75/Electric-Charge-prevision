from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load train and validation datasets
train_data = pd.read_csv(r"C:\Users\pc\Downloads\tp final khadir\sorted_train.csv")
validation_data = pd.read_csv(r"C:\Users\pc\Downloads\tp final khadir\sorted_validation.csv")

# Set the 'index' column as the index
train_data.set_index('index', inplace=True)
validation_data.set_index('index', inplace=True)

# Select relevant columns for modeling
X_train = train_data[['t-24']].values  # Using t-24 as input
y_train = train_data['output'].values  # Using national electricity load as output

X_val = validation_data[['t-24']].values  # Using t-24 as input for validation data
y_val = validation_data['output'].values

# Use KMeans to define RBF centers
n_centers = 35
kmeans = KMeans(n_clusters=n_centers, random_state=42).fit(X_train)

# Create radial basis functions (RBFs)
def RBF(X, centers, gamma):
    return np.exp(-gamma * ((X[:, None] - centers) ** 2).sum(axis=2))

# Calculate RBF activations for training and validation data
X_train_RBF = RBF(X_train, kmeans.cluster_centers_, gamma=0.1)
X_val_RBF = RBF(X_val, kmeans.cluster_centers_, gamma=0.1)

# Create the linear regression model
lin_reg = LinearRegression()

# Train the linear regression model on RBF activations
lin_reg.fit(X_train_RBF, y_train)

# Define the timestamp to predict
timestamp_to_predict = ['2013-06-24 18:00:00']

# Find the closest index in the validation data
index_to_predict = [validation_data.index.get_loc(timestamp) for timestamp in timestamp_to_predict]
y_valHour = y_val[index_to_predict]

# Select data for prediction
X_to_predict = X_val_RBF[index_to_predict]

# Make predictions on the validation data
y_pred_RBF = lin_reg.predict(X_to_predict)

# Denormalize the predicted consumption
predicted_consumption = y_pred_RBF

print("Predicted consumption for the timestamp {}: {}".format(timestamp_to_predict, predicted_consumption))

# Calculate MSE and R2 score for the RBF model
mse_RBF = mean_squared_error(y_val, lin_reg.predict(X_val_RBF))
r2_RBF = r2_score(y_val, lin_reg.predict(X_val_RBF))

print("RBF Model Performance:")
print("MSE:", mse_RBF)
print("R2 Score:", r2_RBF)
# Calculate accuracy
try:
    threshold = 0.5  # Define a threshold for classification
    y_pred_class = (lin_reg.predict(X_val_RBF) > threshold).astype(int)
    y_val_class = (y_val > threshold).astype(int)
    accuracy = (y_pred_class == y_val_class).mean()
    print("Accuracy:", accuracy)
except ValueError as e:
    print("Accuracy: NaN -", e)


# Plot the predictions
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_valHour)), y_valHour, label='Actual Values', color='blue', zorder=5)
plt.scatter(range(len(y_pred_RBF)), y_pred_RBF, label='Predicted Value (RBF)', color='green', zorder=5)
plt.xlabel('Index')
plt.ylabel('Electricity Consumption')
plt.title('Comparison of Actual and Predicted Values (RBF)')
plt.legend()
plt.show()
