# ================================ IMPORT =====================================
import sys
import json
import pandas as pd                                 # type: ignore #ignore
import matplotlib.pyplot as plt                     # type: ignore #ignore
import numpy as np

from logger import setup_logger, GREEN_B

# =============================== CONSTANTES ===================================
LOGGER = setup_logger()
THETA_0 = 0
THETA_1 = 0
MSE_HISTORY = []
ITERATIONS = 10000
LEARNING_RATE = 0.01

# =============================== FONCTIONS ====================================
def recup_data(file: str=""):

    try:
        data = pd.read_csv(file)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='raise')

        LOGGER.info(f"Données du fichier '{file}' récupérer avec succès :\n{data.head()}")
    except Exception as e:
        LOGGER.error(f"Lors de la récuperation des données : {e}")
        return sys.exit(1)

    return data


# -------------------------------------------------------------------------------
def normalize(x):
    x_min = min(x)
    x_max = max(x)
    x_norm = [(xi - x_min) / (x_max - x_min) for xi in x]

    return np.array(x_norm, dtype=float), x_min, x_max


#------------------------------------------------------------------------------
def descente_gradiant(data: pd.DataFrame, theta0=0.0, theta1=0.0):

    try:
        # Données d'origine :
        x = np.array(data["km"].values, dtype=float)
        y = np.array(data["price"].values, dtype=float)
        n = len(x)


        # Normalisation pour stabilité
        x_norm, x_min, x_max = normalize(x)
        y_norm, y_min, y_max = normalize(y)
        n = len(x)
        MSE_HISTORY = []

        # Descente de gradient sur données normalisées :
        for _ in range(ITERATIONS):
            # Prédiction :
            y_pred = theta0 + theta1 * x_norm
            error = y_pred - y_norm

            # Gradients :
            gradient_b0 = (1/n) * error.sum()
            gradient_b1 = (1/n) * (error * x_norm).sum()

            # Mise à jour des paramètres :
            theta0 -= LEARNING_RATE * gradient_b0
            theta1 -= LEARNING_RATE * gradient_b1

            # MSE :
            mse = (error ** 2).mean()
            MSE_HISTORY.append(mse)

        # Dénormalisation des paramètres :
        theta1_denorm = theta1 * (y_max - y_min) / (x_max - x_min)
        theta0_denorm = y_min + (y_max - y_min) * (theta0 - theta1 * x_min / (x_max - x_min))

        return theta0_denorm, theta1_denorm

    except Exception as e:
        LOGGER.error(f"Erreur pendant la descente de gradient : {e}")
        sys.exit(1)

#------------------------------------------------------------------------------
def save_values(theta0: float=0.0, theta1: float=0.0) -> None:

    params = {
        "theta0": theta0,
        "theta1": theta1,
    }

    with open("thetas.json", "w") as f:
        json.dump(params, f)


#------------------------------------------------------------------------------
def main() -> int:
    """

    """

    # [1]. Récupération des données :
    data = recup_data("data.csv")

    # [2]. Descente de gradiant:
    theta_0, theta_1 = descente_gradiant(data, THETA_0, THETA_1)

    # [3]. Sauvegarde des variables theta0 et theta1:
    save_values(theta_0, theta_1)

    return 0


# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    sys.exit(main())