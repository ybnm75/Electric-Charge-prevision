from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Charger les données d'entraînement et de validation
train_data = pd.read_csv(r"C:\Users\pc\Downloads\tp final khadir\sorted_train.csv")
validation_data = pd.read_csv(r'C:\Users\pc\Downloads\tp final khadir\sorted_validation.csv')

# Sélectionner les colonnes pertinentes pour la modélisation
X_train = train_data.iloc[:, 2:26].values  # Utiliser t-1 à t-24 comme entrée
y_train = train_data['output'].values  # Utiliser la charge électrique nationale comme sortie

X_val = validation_data.iloc[:, 2:26].values
y_val = validation_data['output'].values

# Utiliser KMeans pour définir les centres des RBF
n_centers = 35
kmeans = KMeans(n_clusters=n_centers, random_state=42).fit(X_train)

# Créer les fonctions de base radiales (RBF)
def RBF(X, centers, gamma):
    return np.exp(-gamma * ((X[:, None] - centers) ** 2).sum(axis=2))

# Calculer les activations RBF pour les données d'entraînement et de validation
X_train_RBF = RBF(X_train, kmeans.cluster_centers_, gamma=0.1)
X_val_RBF = RBF(X_val, kmeans.cluster_centers_, gamma=0.1)

# Créer le modèle de régression linéaire
lin_reg = LinearRegression()

# Entraîner le modèle de régression linéaire sur les activations RBF
lin_reg.fit(X_train_RBF, y_train)

# Faire des prédictions sur les données de validation
y_pred_RBF = lin_reg.predict(X_val_RBF)

# Évaluer les performances du modèle RBF
mse_RBF = mean_squared_error(y_val, y_pred_RBF)
r2_RBF = r2_score(y_val, y_pred_RBF)

print("Performances du modèle RBF :")
print("MSE :", mse_RBF)
print("R2 Score :", r2_RBF)

# Calculate accuracy
try:
    threshold = 0.5  # Define a threshold for classification
    y_pred_class = (lin_reg.predict(X_val_RBF) > threshold).astype(int)
    y_val_class = (y_val > threshold).astype(int)
    accuracy = (y_pred_class == y_val_class).mean()
    print("Accuracy:", accuracy)
except ValueError as e:
    print("Accuracy: NaN -", e)


plt.figure(figsize=(10, 6))
plt.plot(y_val, label='Valeurs Réelles', color='blue')
plt.plot(y_pred_RBF, label='Valeurs Prédites (RBF)', color='red')
plt.xlabel('Index')
plt.ylabel('Charge Électrique')
plt.title('Comparaison des Valeurs Réelles et Prédites (RBF)')
plt.legend()
plt.show()