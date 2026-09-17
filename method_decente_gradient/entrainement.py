# ================================ IMPORT =====================================
import sys
import json
import pandas as pd                                 # type: ignore #ignore
import numpy as np

from logger import setup_logger
from utils import recup_data

# =============================== CONSTANTES ===================================
LOG = setup_logger()
THETA_0 = 0
THETA_1 = 0
ITERATIONS = 10000
LEARNING_RATE = 0.01


# =============================== FONCTIONS ====================================
def normalize(x: list[float] | np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normalise une séquence numérique dans l'intervalle [0, 1].

    Args:
        x (list[float] | numpy.ndarray):
            Séquence de valeurs numériques à normaliser.

    Returns:
        tuple:
            - numpy.ndarray: Tableau des valeurs normalisées entre 0 et 1.
            - float: Valeur minimale de la séquence d'origine.
            - float: Valeur maximale de la séquence d'origine.

    Raises:
        ValueError: Si la séquence est vide.
        ZeroDivisionError: Si toutes les valeurs de la séquence sont identiques.
    """

    x_min = min(x)
    x_max = max(x)
    x_norm = [(xi - x_min) / (x_max - x_min) for xi in x]

    return np.array(x_norm, dtype=float), x_min, x_max


#------------------------------------------------------------------------------
def descente_gradiant(data: pd.DataFrame, theta0: float, theta1: float):
    """
    Effectue une régression linéaire simple (une variable) en utilisant la descente de gradient
    sur des données normalisées, puis renormalise les paramètres.

    Args:
        data (pd.DataFrame): DataFrame contenant au moins deux colonnes :
            - "km" : les valeurs d'entrée (x)
            - "price" : les valeurs de sortie (y)
        theta0 (float, optional): Paramètre initial de l'ordonnée à l'origine. Par défaut 0.0.
        theta1 (float, optional): Paramètre initial de la pente. Par défaut 0.0.

    Returns:
        tuple:
            - float: theta0 dénormalisé (ordonnée à l'origine)
            - float: theta1 dénormalisé (pente)

    Raises:
        KeyError: Si les colonnes "km" ou "price" ne sont pas présentes dans `data`.
        ValueError: Si les colonnes "km" ou "price" sont vides.
        Exception: Pour toute erreur inattendue lors de la descente de gradient.

    Notes:
        - La fonction utilise des constantes globales `ITERATIONS`, `LEARNING_RATE`, et `MSE_HISTORY`.
        - La normalisation est effectuée pour améliorer la stabilité de la descente de gradient.
        - Les paramètres sont renormalisés à la fin pour correspondre à l'échelle d'origine des données.
    """

    try:
        # Données d'origine :
        x = np.array(data["km"].values, dtype=float)
        y = np.array(data["price"].values, dtype=float)
        n = len(x)

        # Normalisation pour stabilité :
        x_norm, x_min, x_max = normalize(x)
        y_norm, y_min, y_max = normalize(y)

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

        # Dénormalisation des paramètres :
        theta1_denorm = theta1 * (y_max - y_min) / (x_max - x_min)
        theta0_denorm = y_min + (y_max - y_min) * (theta0 - theta1 * x_min / (x_max - x_min))

        return theta0_denorm, theta1_denorm
    except KeyError:
        LOG.error(f"les colonnes 'km' ou 'price' ne sont pas présentes dans `data`")
        raise
    except Exception as e:
        LOG.error(f"Erreur pendant la descente de gradient : {e}")
        raise

#------------------------------------------------------------------------------
def save_values(theta0: float, theta1: float, filepath: str = "thetas.json") -> None:
    """
    Sauvegarde les paramètres theta0 et theta1 dans un fichier JSON.

    Args:
        theta0 (float, optional): Valeur du paramètre theta0 à sauvegarder. Par défaut 0.0.
        theta1 (float, optional): Valeur du paramètre theta1 à sauvegarder. Par défaut 0.0.

    Returns:
        None

    Raises:
        IOError: Si le fichier "thetas.json" ne peut pas être ouvert ou écrit.

    Notes:
        - Le fichier JSON sera créé ou écrasé dans le répertoire courant.
        - Le format du fichier sera :
          {
              "theta0": <valeur de theta0>,
              "theta1": <valeur de theta1>
          }
    """


    params = {
        "theta0": theta0,
        "theta1": theta1,
    }
    try:
        with open(filepath, "w") as f:
            json.dump(params, f)
    except PermissionError as e:
        LOG.warning(f"Fonction save_value(): {e}")
        raise

    except Exception as e:
        LOG.error(f"Lors de la sauvegarde des valeurs : {e}")
        raise

    LOG.info(f"Création d'un fichier thetas.json avec les valeurs:")
    LOG.info(f"- theta0 = {theta0:.4f}\t- theta1 = {theta1:.4f}")
    return

#------------------------------------------------------------------------------
def main() -> int:
    """
    Fonction programme principal>
    """

    print("Bienvenue dans le programme d'entrainement de linar regression.")

    try:
        # [1]. Récupération des données :
        data = recup_data("data.csv")

        # [2]. Descente de gradiant:
        theta_0, theta_1 = descente_gradiant(data, THETA_0, THETA_1)

        # [3]. Sauvegarde des variables theta0 et theta1:
        save_values(theta_0, theta_1)

    except Exception:
        LOG.critical(f"Fermeture forcé du programme !")
        return 1

    LOG.info(f"Fermeture du programme !")
    return 0


# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    sys.exit(main())