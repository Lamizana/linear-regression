# ================================ IMPORT =====================================
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt

# ============================== CONSTANTES ===================================
# Couleurs :
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# Gestion des kilometres :
KM_MIN = 0
KM_MAX = 400_000
KM = 20_000

# ================================ CLASSE =====================================
class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Change dynamiquement le format selon le niveau du log :
        if record.levelno >= logging.ERROR:
            self._style._fmt = f'{RED}[%(levelname)s] %(message)s (%(filename)s:%(lineno)d){RESET}'
        else:
            self._style._fmt = '[%(levelname)s] %(message)s'
        
        return super().format(record)
    
# ================================ HANDLER =====================================
# Configuration du logging :
handler = logging.StreamHandler()
formatter = ColorFormatter('[%(levelname)s] %(message)s (%(filename)s:%(lineno)d)')
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(handler)


# ================================ FONCTIONS ==================================
def save_graph(data: pd.DataFrame):
    
    # Trace un graphique simple (nuage de points) :
    plt.plot(
        data["km"], data["price"],
        marker='o',
        linestyle = ' ',
        color='blue'
    )
    
    # Titrage et labels :
    plt.title("Évolution du prix en fonction des kilomètres parcourus")
    plt.xlabel("Kilomètres parcourus")
    plt.ylabel("Prix du véhicule (€)")
    plt.grid(True)
    plt.xlim(0, data["km"].max() + 10_000)
    plt.ylim(0, data["price"].max() + 1_000)

    # Sauvegarde le graphique :
    plt.savefig("relation_prix_km.png")    
    return


#------------------------------------------------------------------------------
def save_regression_graph(data: pd.DataFrame, beta_0: float, beta_1: float):

    # Trace un graphique simple (nuage de points) :
    plt.plot(
        data["km"],
        data["price"],
        marker='o',
        linestyle = ' ',
        color='blue'
    )
    
    # Droite de regresion :
    x_vals = list(data["km"])
    y_vals = list(linear_regression(data["km"], beta_0, beta_1))

    # Ajoute le point à l'origine :
    x_vals.insert(0, 0)
    y_vals.insert(0, beta_0)

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
    plt.xlim(right=data["km"].max() + 10_000)
    plt.ylim(0, data["price"].max() + 1_000)

    # Sauvegarde du graphique :
    plt.savefig("regression_lineaire_voiture.png")    
    return


#------------------------------------------------------------------------------
def pente(data: pd.DataFrame) -> float:

    # Moyenne des notes et des heures :
    mean_prix = data["price"].mean()
    mean_km = data["km"].mean()
    
    logging.debug(f"Moyenne des prix : {mean_prix}")
    logging.debug(f"Moyenne des kilomètres : {mean_km}")
    
    # Calculs préalables :
    data["Xi - mean_km"] = [(x - mean_km) for x in data["km"]]
    data["Yi - mean_prix"] = [(y - mean_prix) for y in data["price"]]
    logging.debug(f"Rajout de colonnes :\n{data.head()}")
    
    # Calcule la pente (coefficient directeur) beta_1 :
    try:
        a = b =  0.0
        for x, y in zip(data["Xi - mean_km"], data["Yi - mean_prix"]):
            a += x * y
        
        for x in data["Xi - mean_km"]:
            b += x * x
            
        beta_1 = a / b
    except Exception as e:
        logging.error(f"Fonction pente(): {e}")
        raise

    logging.info(f"Valeur de la pente (Beta_1) : {beta_1:.2f}")
    return(beta_1)


#------------------------------------------------------------------------------
def origin(data: pd.DataFrame, beta_1: float) -> float:
    
    y = data["price"].mean()
    x = data["km"].mean()
    
    try:
        beta_0 = y - beta_1 * x
    except TypeError as e:
        logging.error(f"{RED}Fonction origin(): {e}{RESET}")
        raise
    
    logging.info(f"Valeur du point à l'origine (beta_0) : {beta_0:.2f}")
    return beta_0


#------------------------------------------------------------------------------
def linear_regression(heure: float, beta_0: float, beta_1: float) -> float:
    
    try:
        predict = beta_0 + beta_1 * heure
    except TypeError:
        logging.error("Fonction linear_regression() :")
        raise
    return predict


#------------------------------------------------------------------------------
def main():
    """
    Fonction principale exécutant les étapes d'une régression linéaire simple :

    Étapes :
    --------
    1. Chargement du fichier CSV (`predict_prix.csv`)
    2. Affichage et sauvegarde d’un graphique initial (nuage de points)
    3. Calcul de la pente (beta_1) via la méthode des moindres carrés
    4. Calcul de l’ordonnée à l’origine (beta_0)
    5. Prédiction pour une valeur donnée (valeur à changer)
    6. Affichage et sauvegarde du graphique avec la droite de régression

    Returns
    -------
    int
        0 : succès
        1 : erreur de chargement du fichier
        2 : erreur de calcul (pente, origine ou prédiction)
    """

    # ---------
    # Étape 1 : Gestion et traitement des donnéés
    try:
        # [1] Chargement et affichage des données + nuage de points :
        file = "predict_prix.csv"
        data = pd.read_csv(file)
        logging.info(f"Fichier '{file}' exécuté avec succès : \n{data.head()}")
    except Exception as e:
        logging.error(e)
        return 1

    # ---------
    # Étape 2 : Regression lineaire simple avec la methode des moindres carres
    try:
        # [1] Calcul de la pente :
        beta_1 = pente(data)

        # [2] Calcul de l’ordonnée à l’origine :
        beta_0 = origin(data, beta_1)

        # [3] Ne pas vendre a perte (depassement du ratio km/prix) :
        km = KM
        if km > KM_MAX:
            logging.warning(f"{YELLOW}Tu dépasse la limite de kilomètres établie, tu va devoir payer pour qu'on prenne ta caisse !{RESET}")
            
        # [4] Prédiction pour X kilometres :
        predict = linear_regression(km, beta_0, beta_1)
        print(GREEN)
        logging.info(f"Le prix estimé pour une voiture qui a {km:_} km au compteur : {predict:.2f} €{RESET}")
        
    except Exception as e:
        logging.error(f"--- RAGE QUIT !! ---")
        return 2
    
    # ---------
    # Étape 3 : Sauvegarde du graphique avec la droite de régression :
    save_regression_graph(data, beta_0, beta_1)

    return 0


# ================================= PROGRAMME ==================================
if __name__ == "__main__":
    sys.exit(main())