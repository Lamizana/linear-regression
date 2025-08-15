# ================================ IMPORT =====================================
import sys
import json
import pandas as pd                                 # type: ignore #ignore
import matplotlib.pyplot as plt                     # type: ignore #ignore
from logger import setup_logger, GREEN_B, YELLOW_B

LOGGER = setup_logger()

# =============================== FONCTIONS ====================================
def recup_km() -> float:

    msg = "Donner un valeur kilometrique pour predire le prix de la voiture : "

    while True:
        try:
            km  = float(input(msg))
            return km
        except KeyboardInterrupt as e:
            LOGGER.warning(f"\nInterruption volontaire du programme !")
            sys.exit(0)
        except Exception as e:
            LOGGER.warning(f"Lors de la recuperation du kilometrage : {e}")


#------------------------------------------------------------------------------
def recup_theta(file : str="") -> list[float]:

    theta0 = theta1 = 0.0
    try:
        with open(file, "r") as f:
            data  = json.load(f)
            theta0 = float(data.get("theta0"))
            theta1 = float(data.get("theta1"))
        LOGGER.info(f"- Theta0 = {theta0}\t- Theta1 = {theta1}\n")
    except Exception as e:
        LOGGER.error(f"Erreur lors de la recuperation des valeurs Theta : {e}")
        sys.exit(1)
    
    return theta0, theta1


#------------------------------------------------------------------------------
def main() -> int:
    """

    """

    # [1]. Recuperer un kilometrage donnee :
    km = recup_km()

    # [2]. Recuperation des donnees theta-0 et theta-1 :
    theta0, theta1 = recup_theta("thetas.json")

    # [3]. Application de la regression lineaire simple :
    estimation_prix = theta0 + (theta1 * km)
    if estimation_prix < 0:
        LOGGER.warning(f"{YELLOW_B}La voiture à trop de kilometre, il faudrait payer pour la vendre !!")
    elif estimation_prix == 0:
        LOGGER.warning(f"{YELLOW_B}Vous ne gagnerez rien à la revendre")
    else:
        LOGGER.info(f"{GREEN_B}Le prix estimé pour une voiture ayant {km} km est de : {estimation_prix:.2f}€")

    return 0


# =============================== PROGRAMME ===================================
if __name__ == "__main__":
    sys.exit(main())