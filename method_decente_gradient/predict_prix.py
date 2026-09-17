# ================================ IMPORT =====================================
import sys
from logger import setup_logger, GREEN_B, YELLOW_B
from utils import recup_theta

LOG = setup_logger()

# =============================== FONCTIONS ====================================
def recup_km() -> float:
    """
    Demande à l'utilisateur une valeur kilométrique pour prédire le prix d'une voiture.

    Returns:
        float: La valeur kilométrique saisie par l'utilisateur (toujours positive).

    Raises:
        raise: Si l'utilisateur interrompt le programme (Ctrl+C).

    Notes:
        - Si une valeur négative ou invalide est saisie, l'utilisateur doit réessayer.
    """
    msg = "Donner un valeur kilometrique pour predire le prix de la voiture : "

    while True:

        try:
            km  = float(input(msg).strip())
            if km < 0:
                LOG.warning(f"La valeur doit être positive. Réessayez:")
                continue
            return km

        except KeyboardInterrupt:
            raise
        except ValueError as e:
            LOG.warning(f"Lors de la recuperation du kilometrage : {e}")


#------------------------------------------------------------------------------
def main() -> int:
    """
    Point d'entrée principal du programme de prédiction de prix de voiture.

    Étapes :
        1. Demande à l'utilisateur un kilométrage.
        2. Récupère les paramètres de la régression (theta0, theta1).
        3. Estime le prix en appliquant la régression linéaire simple.
        4. Affiche le résultat ou un avertissement selon le cas.

    Returns:
        int: Code de sortie du programme
            - 0 : succès
            - 1 : erreur générale
            - 2 : fichier manquant
    """


    try:
        # [1]. Recuperation des donnees theta-0 et theta-1 :
        theta0, theta1 = recup_theta("thetas.json")

        # [2]. Recuperer un kilometrage donnee :
        km = recup_km()

        # [3]. Application de la regression lineaire simple :
        estimation_prix = theta0 + (theta1 * km)
        if estimation_prix < 0:
            LOG.warning(f"{YELLOW_B}Prix estimé : {estimation_prix:.2f}€.\nLa voiture à trop de kilometre, il faudrait payer pour la vendre !!")
        elif estimation_prix == 0:
            LOG.warning(f"{YELLOW_B}Prix estimé : {estimation_prix:.2f}€.\nVous ne gagnerez rien à la revendre.")
        else:
            LOG.info(f"{GREEN_B}Le prix estimé pour une voiture ayant {km} km est de : {estimation_prix:.2f}€")

    except KeyboardInterrupt:
        LOG.warning(f"\nInterruption volontaire du programme !")

    except FileNotFoundError:
        return 2

    except Exception:
        LOG.critical(f"Fermeture du programme !")
        return 1

    LOG.info(f"Fermeture du programme !")
    return 0


# =============================== PROGRAMME ===================================
if __name__ == "__main__":
    sys.exit(main())