# ================================ IMPORT =====================================
import sys
import json
import pandas as pd                                 # type: ignore #ignore
import matplotlib.pyplot as plt                     # type: ignore #ignore
from logger import setup_logger, GREEN_B

# =============================== CONSTANTES ===================================
logger = setup_logger()
THETA_0 = 0
THETA_1 = 0

# =============================== FONCTIONS ====================================
def recup_data(file: str=""):

    try:
        data = pd.read_csv(file)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='raise')

        logger.info(f"Données du fichier '{file}' recupérer avec succès : \n{data.head()}")
    except Exception as e:
        logger.error(f"Lors de la récuperation des données : {e}")
        return sys.exit(1)

    return data


#------------------------------------------------------------------------------
def save_graph(data: pd.DataFrame) -> None:
    """
    Génère et enregistre un graphique en nuage de points (scatter plot)
    illustrant la relation entre les heures d'étude et les notes obtenues.

    Le graphique est sauvegardé sous le nom "relation_notes_heures.png".

    Parameters
    ----------
    data : pandas.DataFrame
        Un DataFrame contenant deux colonnes :
        - "km" : les valeurs sur l'axe des X (km parcourues)
        - "price"  : les valeurs sur l'axe des Y (prix obtenues)

    Returns
    -------
    None

    Raises
    ------
    KeyError
        Si les colonnes "km" ou "price" sont absentes du DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"km": [1, 2, 3], "price": [6, 6.5, 7.5]})
    >>> save_graph(df)
    # Enregistre le fichier 'relation_km_price.png'
    """


    try:
        # [1]. Trace un graphique simple (nuage de points) :
        plt.plot(
            data["km"].values,
            data["price"].values,
            marker='o',
            linestyle = ' ',
            color='blue'
        )
        
        # [2]. Titrage et labels :
        plt.title("Évolution des prix en fonction des kilometres parcourues")
        plt.xlabel("Kilomètres parcourus")
        plt.ylabel("Prix du véhicule (€)")
        plt.grid(True)
        plt.xlim(0, data["km"].max() + 20_000)
        plt.ylim(0, data["price"].max() + 2_000)

        # [3].Sauvegarde le graphique :
        file = "relation_km_price.png"
        plt.savefig(file)
        logger.info(f"Fichier '{file}' enregistrer avec succès !")

    except Exception as e:
        logger.error(f"Lors de la création du graphique : {e}")
        return sys.exit(1)
    return



#------------------------------------------------------------------------------
def mean_squarred_error(data: pd.DataFrame, theta0=0.0, theta1=0.0)-> float:

    try:
        # Valeurs réelles :
        y = data["price"]
        x = data["km"].values

        # Valeurs prédites :
        y_pred = theta0 + theta1 * x

        # MSE :
        mse = ((y_pred - y) ** 2).mean()
        
        logger.info(f"MSE : {mse:.2f}")
        return mse
    
    except Exception as e:
        logger.error(f"Lors du calcul du MSE : {e}")
        return sys.exit(1)


#------------------------------------------------------------------------------
def main() -> int:
    """

    """

    # [1]. Récupération des données :
    data = recup_data("data.csv")

    # [2]. Chargement et affichage des données + nuage de points :
    save_graph(data)

    # [3]. Calcul de la fonction de cout (MSE) :
    mean_squarred_error(data, THETA_0, THETA_1)
    return 0

# =============================== PROGRAMME ===================================
if __name__ == "__main__":
    sys.exit(main())



