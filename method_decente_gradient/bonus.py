# ================================ IMPORT =====================================
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logger import setup_logger, GREEN_B
from utils import recup_data, recup_theta

# =============================== CONSTANTES ===================================
LOGGER = setup_logger()
FILE_DATA = "data.csv"
FILE_THETA = "thetas.json"
GRAPH = "regression_lineaire.png"

# =============================== FONCTIONS ====================================
def save_regression_graph(data: pd.DataFrame, theta_0: float, theta_1: float):
    """
    Trace et sauvegarde le graphique de la régression linéaire.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame contenant les colonnes 'km' et 'price'.
    theta_0 : float
        Ordonnée à l'origine de la droite de régression.
    theta_1 : float
        Pente de la droite de régression.

    Raises
    ------
    KeyError
        Si 'km' ou 'price' sont absents de `data`.
    OSError
        Si le fichier ne peut pas être sauvegardé.
    """


    try:
        if "km" not in data.columns or "price" not in data.columns:
            raise KeyError("Le DataFrame doit contenir les colonnes 'km' et 'price'.")

        # Trace un graphique simple (nuage de points) :
        plt.scatter(
            data["km"].values,
            data["price"].values,
            color='blue',
            label='prix')

        # Droite de regresion :
        x = np.array(data["km"].values, dtype=float)

        x_vals = list(x)
        y_vals = [theta_0 + theta_1 * km for km in x]

        # Ajoute le point à l'origine :
        x_vals.insert(0, 0)
        y_vals.insert(0, theta_0)

        plt.plot(x_vals,
                 y_vals,
                 color='red',
                 label='Régression')

        # Titrages et labels :
        plt.title("Évolution du prix en fonction des kilomètres parcourus")
        plt.xlabel("Kilomètres parcourus")
        plt.ylabel("Prix du véhicule (€)")
        plt.legend()

        # Reglage de la fenetre :
        plt.grid(True)
        plt.xlim(0, data["km"].max() + 20_000)
        plt.ylim(0, data["price"].max() + 2_000)

        # Sauvegarde du graphique :
        plt.savefig(GRAPH)
        LOGGER.info(f"Fichier {GRAPH} enregistré")

    except Exception as e:
        LOGGER.error(f"Erreur lors de la création du graphique : {e}")
        raise
    finally:
        plt.close()
    return


#------------------------------------------------------------------------------
def mean_squared_error(y, y_pred) -> float:
    """
    Calcule l'erreur quadratique moyenne (Mean Squared Error, MSE) entre les valeurs réelles et les valeurs prédites.

    Parameters
    ----------
    y : array-like
        Les valeurs réelles.
    y_pred : array-like
        Les valeurs prédites par le modèle.

    Returns
    -------
    float
        La valeur du MSE, c'est-à-dire la moyenne des carrés des écarts entre y et y_pred.
    """

    try:
        mse = 0
        mse = ((y - y_pred) ** 2).mean()

    except (KeyError, ValueError, TypeError) as e:
        LOGGER.error(f"Valeurs de theta invalides dans 'MSE' : {e}")
        raise

    return float(mse)

#------------------------------------------------------------------------------
def mean_absolute_error(y, y_pred) -> float:
    """
    Calcule l'erreur absolue moyenne (Mean Absolute Error, MAE) entre les valeurs réelles et les valeurs prédites.

    Parameters
    ----------
    y : array-like
        Les valeurs réelles.
    y_pred : array-like
        Les valeurs prédites par le modèle.

    Returns
    -------
    float
        La valeur du MAE, c'est-à-dire la moyenne des valeurs absolues des écarts entre y et y_pred.
    """


    try:

        mae = np.abs(y - y_pred).mean()

    except (KeyError, ValueError, TypeError) as e:
        LOGGER.error(f"Valeurs de theta invalides dans 'MAE' : {e}")
        raise

    return float(mae)

#------------------------------------------------------------------------------
def r2_score(y, y_pred) -> float:
    """
    Calcule le coefficient de détermination R² pour mesurer la qualité de la régression.

    R² indique la proportion de la variance des valeurs réelles expliquée par le modèle.
    Une valeur proche de 1 indique un bon ajustement.

    Parameters
    ----------
    y : array-like
        Les valeurs réelles.
    y_pred : array-like
        Les valeurs prédites par le modèle.

    Returns
    -------
    float
        Le coefficient R² de la régression.
    """

    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)

#------------------------------------------------------------------------------
def calcul_precision(data: pd.DataFrame, theta0: float, theta1: float) -> None:
    """
    Calcule et affiche les métriques de précision pour une régression linéaire.

    Les métriques calculées sont :
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Coefficient de détermination (R²)

    Parameters
    ----------
    data : pandas.DataFrame
        Un DataFrame contenant les colonnes :
        - "km" : nombre de kilomètres parcourus par la voiture.
        - "price" : prix de la voiture.
    theta0 : float
        L'ordonnée à l'origine de la droite de régression.
    theta1 : float
        La pente de la droite de régression.

    Returns
    -------
    None
        La fonction ne renvoie rien. Elle affiche simplement les métriques dans la console.

    Raises
    ------
    KeyError
        Si les colonnes "km" ou "price" ne sont pas présentes dans le DataFrame.
    """


    x = np.array(data["km"].values, dtype=float)
    y = np.array(data["price"].values, dtype=float)

    # Prédictions :
    y_pred = theta0 + theta1 * x

    # Calcul des métriques :
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"MSE (Mean Squared Error) = {mse:.2f}")
    print(f"MAE (Mean Absolute Error) = {mae:.2f}")
    print(f"R² (coefficient de détermination) = {r2:.4f}")
    return

#------------------------------------------------------------------------------
def main() -> int:
    """
    Fonction programme principal
    """

    print("Bienvenue dans les BONUS de linar regression.")
    try:
        # [1]. Récupération des données :
        data = recup_data(FILE_DATA)
        theta0, theta1 = recup_theta(FILE_THETA)

        # [2]. Creation du graphique :
        save_regression_graph(data, theta0, theta1)

        # [3]. Precision de l'algorithme :
        calcul_precision(data, theta0, theta1)

    except FileNotFoundError:
        return 2
    except Exception:
        LOGGER.critical(f"Fermeture du programme !")
        return 1

    return 0

# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    sys.exit(main())