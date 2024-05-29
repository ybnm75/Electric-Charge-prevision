from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load train and validation datasets
train_data = pd.read_csv(r"C:\Users\pc\Downloads\tp final khadir\sorted_train.csv")
validation_data = pd.read_csv(r"C:\Users\pc\Downloads\tp final khadir\sorted_validation.csv")

# Set the 'index' column as the index
train_data.set_index('index', inplace=True)
validation_data.set_index('index', inplace=True)

# Drop other columns except the target and features
X_train = train_data.drop(columns=['output','d_0','d_1','d_2','d_3','d_4','d_5','d_6','SD']).values
y_train = train_data['output'].values               

X_val = validation_data.drop(columns=['output','d_0','d_1','d_2','d_3','d_4','d_5','d_6','SD']).values  
y_val = validation_data['output'].values               

# Initialize MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(25, 30, 20), max_iter=700, activation='relu', 
                   alpha=0.0001, learning_rate_init=0.001, batch_size=100, 
                   early_stopping=True, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Define timestamps to predict
timestamps_to_predict = ['2013-01-11 19:00:00']  # Example timestamps

# Convert timestamps to row indices
indices_to_predict = [validation_data.index.get_loc(timestamp) for timestamp in timestamps_to_predict]

# Select data for prediction
X_to_predict = X_val[indices_to_predict]

# Predict on selected data
y_predVal = mlp.predict(X_to_predict)

y_pred = mlp.predict(X_val)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_val[indices_to_predict], y_predVal)
print("Mean Squared Error (MSE):", mse)

# Print predictions
print("Predictions for selected timestamps:")
for timestamp, pred in zip(timestamps_to_predict, y_predVal):
    print(f"Timestamp {timestamp}: Predicted value = {pred}")

# Calculate R^2 score for the entire validation set
try:
    r2 = r2_score(y_val, mlp.predict(X_val))
    msePMC = mean_squared_error(y_val,y_pred)
    print("R-squared (R^2) score:", r2)
    print("MSE:", msePMC)
except ValueError as e:
    print("R-squared (R^2) score: NaN -", e)

# Calculate accuracy
try:
    threshold = 0.5  # Define a threshold for classification
    y_pred_class = (mlp.predict(X_val) > threshold).astype(int)
    y_val_class = (y_val > threshold).astype(int)
    accuracy = (y_pred_class == y_val_class).mean()
    print("Accuracy:", accuracy)
except ValueError as e:
    print("Accuracy: NaN -", e)

plt.figure(figsize=(10, 6))
plt.plot(y_val, label='Valeurs Réelles', color='blue')
plt.plot(y_pred, label='Valeurs Prédites (PMC)', color='green')
plt.xlabel('Index')
plt.ylabel('Charge Électrique')
plt.title('Comparaison des Valeurs Réelles et Prédites (PMC)')
plt.legend()
plt.show()