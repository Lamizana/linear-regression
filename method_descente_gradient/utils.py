# ================================ IMPORT =====================================
import json
import pandas as pd

from logger import setup_logger

# =============================== CONSTANTES ===================================
LOG = setup_logger()


# =============================== FONCTIONS ====================================
def recup_data(file: str) -> pd.DataFrame:
    """
    Charge un fichier CSV et tente de convertir ses colonnes en numériques.

    Args:
        file (str): Chemin vers le fichier CSV à charger.

    Returns:
        pandas.DataFrame:
            DataFrame contenant les données du fichier CSV.
            - Les colonnes numériques sont converties en `float64` ou `int64`.
            - Les colonnes non convertibles restent inchangées.

    Raises:
        FileNotFoundError: Si le fichier spécifié est introuvable.
        ValueError: Si le fichier est vide.
        pandas.errors.ParserError: Si une erreur de parsing survient lors de la lecture du CSV.
        Exception: Pour toute autre erreur inattendue lors de la récupération des données.

    Notes:
        - Les colonnes non convertibles en numérique sont conservées telles quelles.
        - Les logs indiquent les colonnes qui n'ont pas pu être converties.
        - Un aperçu (`head`) des données est affiché dans les logs en cas de succès.
    """

    try:
        data = pd.read_csv(file)

        # Verification fichier vide :
        if data.empty:
            raise ValueError("Le fichier est vide.")

        # Conversion des colonnes numerique :
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col])
            except (ValueError, TypeError):
                LOG.warning(f"Impossible de convertir la colonne '{col}' en numérique.")

        LOG.info(f"Données du fichier '{file}' récupérer avec succès :\n{data.head()}")

    except FileNotFoundError:
        LOG.error(f"Fichier '{file}' introuvable.")
        raise
    except pd.errors.ParserError as e:
        LOG.error(f"Erreur lors du parsing du fichier '{file}' : {e}")
        raise
    except Exception as e:
        LOG.error(f"Lors de la récuperation des données : {e}")
        raise

    return data


# -------------------------------------------------------------------------------
def recup_theta(file: str) -> tuple[float, float]:
    """
    Récupère les valeurs de theta0 et theta1 depuis un fichier JSON.

    Args:
        file (str): Chemin vers le fichier JSON contenant les paramètres.

    Returns:
        tuple[float, float]:
            - theta0 : Ordonnée à l'origine du modèle.
            - theta1 : Pente du modèle.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        json.JSONDecodeError: Si le contenu du fichier n'est pas un JSON valide.
        KeyError: Si l'une des clés 'theta0' ou 'theta1' est manquante.
        ValueError: Si les valeurs extraites ne sont pas convertibles en float.
    """

    try:
        with open(file, "r") as f:
            data = json.load(f)
            theta0 = float(data.get("theta0"))
            theta1 = float(data.get("theta1"))

    except FileNotFoundError:
        LOG.error(f"Fichier introuvable : {file}")
        raise

    except json.JSONDecodeError as e:
        LOG.error(f"Erreur de décodage JSON dans '{file}' : {e}")
        raise

    except (KeyError, ValueError, TypeError) as e:
        LOG.error(f"Valeurs de theta invalides dans '{file}' : {e}")
        raise

    LOG.info(f"- Theta0 = {theta0}\t- Theta1 = {theta1}\n")
    return theta0, theta1