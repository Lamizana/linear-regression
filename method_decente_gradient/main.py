import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Données d'entrée
# -----------------------
data = pd.DataFrame({
    "Heures": [1, 2, 3, 4],
    "Notes": [6, 6.5, 7.5, 8]
})
X = data["Heures"].values
y = data["Notes"].values

# -----------------------
# Initialisation
# -----------------------
beta_0 = 0
beta_1 = 0

learning_rate = 0.01
epochs = 1000
n = len(X)
mse_history = []

# -----------------------
# Descente de gradient
# -----------------------
for epoch in range(epochs):
    y_pred = beta_0 + beta_1 * X
    error = y_pred - y

    gradient_b0 = (1/n) * error.sum()
    gradient_b1 = (1/n) * (error * X).sum()

    beta_0 -= learning_rate * gradient_b0
    beta_1 -= learning_rate * gradient_b1

    mse = (error ** 2).mean()
    mse_history.append(mse)

# -----------------------
# Résultats finaux
# -----------------------
print(f"Modèle appris : y = {beta_0:.2f} + {beta_1:.2f}x")

# Prédiction
x_test = 2.30
y_test = beta_0 + beta_1 * x_test
print(f"Pour {x_test} heures, la note prédite est : {y_test:.2f}")

# -----------------------
# Visualisation 1 : Données et droite de régression
# -----------------------
plt.figure(figsize=(10, 5))

# Sous-graphique 1 : Données
plt.subplot(1, 2, 1)
plt.scatter(X, y, color="blue", label="Données réelles")
plt.plot(X, beta_0 + beta_1 * X, color="red", label="Régression")
plt.xlabel("Heures d'étude")
plt.ylabel("Notes")
plt.title("Régression linéaire")
plt.legend()
plt.grid(True)

# Sous-graphique 2 : Courbe d'erreur
plt.subplot(1, 2, 2)
plt.plot(range(epochs), mse_history, color="green")
plt.xlabel("Itérations")
plt.ylabel("Erreur quadratique moyenne (MSE)")
plt.title("Convergence de la descente de gradient")
plt.grid(True)

plt.tight_layout()
plt.savefig("test.png")
