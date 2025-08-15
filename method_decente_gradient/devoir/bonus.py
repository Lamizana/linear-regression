# ================================ IMPORT =====================================
import sys
import json
import numpy as np
import pandas as pd                                 # type: ignore #ignore
import matplotlib.pyplot as plt                     # type: ignore #ignore

from logger import setup_logger, GREEN_B

# =============================== CONSTANTES ===================================
LOGGER = setup_logger()
FILE_DATA = "data.csv"
FILE_THETA = "thetas.json"
GRAH = "regression_lineaire.png"

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


#------------------------------------------------------------------------------
def save_regression_graph(data: pd.DataFrame, theta_0: float, theta_1: float):

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
    plt.savefig(GRAH)    
    LOGGER.info(f"Fichier {GRAH} enregistre avec succees")

    return

#------------------------------------------------------------------------------
def recup_theta(file : str="") -> list[float]:

    theta0 = theta1 = 0.0
    try:
        with open(file, "r") as f:
            data  = json.load(f)
            theta0 = float(data.get("theta0"))
            theta1 = float(data.get("theta1"))
        LOGGER.info(f"- Theta0 = {theta0}\t- Theta1 = {theta1}")
    except Exception as e:
        LOGGER.error(f"Erreur lors de la recuperation des valeurs Theta : {e}")
        sys.exit(1)
    
    return theta0, theta1

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

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> mean_squared_error(y, y_pred)
    0.375
    """

    mse = 0
    mse = ((y - y_pred) ** 2).mean()

    return mse

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

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> mean_absolute_error(y, y_pred)
    0.5
    """

    mae = 0
    mae = np.abs(y - y_pred).mean()
    return mae

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

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> r2_score(y, y_pred)
    0.9486
    """

    r2 = 0
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2

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

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     "km": [50000, 120000, 80000],
    ...     "price": [15000, 9000, 12000]
    ... })
    >>> calcul_precision(data, theta0=2000, theta1=-0.05)
    MSE (Mean Squared Error) = 1234567.89
    MAE (Mean Absolute Error) = 789.12
    R² (coefficient de détermination) = 0.9123
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
    Fonction programme principal>
    """

    # [1]. Récupération des données :
    data = recup_data(FILE_DATA)
    theta0, theta1 = recup_theta(FILE_THETA)

    # [2]. Creation du graphique :
    save_regression_graph(data, theta0, theta1)

    # [3]. Precision de l'algorithme :
    calcul_precision(data, theta0, theta1)

    return 0

# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    sys.exit(main())