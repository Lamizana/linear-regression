"""
===============================================================================
Script de régression linéaire simple : prédiction de notes en fonction du temps d'étude

Ce programme lit un fichier CSV contenant des paires (heures d’étude, notes obtenues),
puis applique une régression linéaire simple pour :
    - Visualiser les données sous forme de nuage de points
    - Calculer la droite de régression (pente et ordonnée à l’origine)
    - Afficher et enregistrer un graphique avec la droite de régression
    - Prédire la note attendue pour une durée d’étude donnée

Structure du programme :
------------------------
- `save_graph` : trace le nuage de points initial
- `pente` : calcule la pente (beta_1) de la droite de régression
- `origin` : calcule l’ordonnée à l’origine (beta_0)
- `linear_regression` : prédit une valeur cible à partir d’une entrée X
- `save_regression_graph` : trace la droite de régression sur le graphique
- `main` : enchaîne les étapes et gère les erreurs

Fichier de données attendu :
----------------------------
Le fichier `predict_note.csv` doit contenir au moins deux colonnes :
    - "Heures" : les heures d’étude (valeurs explicatives X)
    - "Notes"  : les notes obtenues (valeurs à prédire Y)

Sorties du programme :
----------------------
- Un fichier image `relation_notes_heures.png` (nuage de points)
- Un fichier image `regression_lineaire.png` (avec droite de régression)
- Une prédiction de note pour une valeur d’heure fixée dans `main`

Auteurs :
---------
- Nom : Alex Lamizana
- Date : 26/07/2025
===============================================================================
"""

# ================================ IMPORT =====================================
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ================================ FONCTIONS ==================================
def save_graph(data: pd.DataFrame):
    """
    Génère et enregistre un graphique en nuage de points (scatter plot)
    illustrant la relation entre les heures d'étude et les notes obtenues.

    Le graphique est sauvegardé sous le nom "relation_notes_heures.png".

    Parameters
    ----------
    data : pandas.DataFrame
        Un DataFrame contenant deux colonnes :
        - "Heures" : les valeurs sur l'axe des X (heures d'étude)
        - "Notes"  : les valeurs sur l'axe des Y (notes obtenues)

    Returns
    -------
    None

    Raises
    ------
    KeyError
        Si les colonnes "Heures" ou "Notes" sont absentes du DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Heures": [1, 2, 3], "Notes": [6, 6.5, 7.5]})
    >>> save_graph(df)
    # Enregistre le fichier 'relation_notes_heures.png'
    """

    # Trace un graphique simple (nuage de points) :
    plt.plot(
        data["Heures"],
        data["Notes"],
        marker='o',
        linestyle = ' ',
        color='blue'
    )
    
    # Titrage et labels :
    plt.title("Évolution des notes en fonction des heures d'étude")
    plt.xlabel("Heures d'étude")
    plt.ylabel("Notes obtenues")
    plt.grid(True)
    plt.xlim(0, data["Heures"].max() + 1)
    plt.ylim(0, data["Notes"].max() + 1)

    # Sauvegarde le graphique :
    plt.savefig("relation_notes_heures.png")    
    return


#------------------------------------------------------------------------------
def save_regression_graph(data: pd.DataFrame, beta_0: float, beta_1: float):
    """
    Génère et enregistre un graphique illustrant la relation entre les heures d'étude
    et les notes obtenues, avec en plus la droite de régression linéaire.

    Le graphique inclut :
        - Un nuage de points représentant les données brutes
        - Une droite de régression calculée à partir des coefficients beta_0 et beta_1

    Le graphique est sauvegardé sous le nom "regression_lineaire.png".

    Parameters
    ----------
    data : pandas.DataFrame
        Un DataFrame contenant au moins deux colonnes nommées :
        - "Heures" : les valeurs explicatives (X)
        - "Notes"  : les valeurs à prédire (Y)

    beta_0 : float
        L'ordonnée à l'origine (interception) de la droite de régression.

    beta_1 : float
        La pente (coefficient directeur) de la droite de régression.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        Si les colonnes "Heures" ou "Notes" sont absentes du DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Heures": [1, 2, 3], "Notes": [6, 6.5, 7.5]})
    >>> save_regression_graph(df, beta_0=5.25, beta_1=0.75)
    # Enregistre le graphique avec la droite de régression
    """


    # Trace un graphique simple (nuage de points) :
    plt.plot(
        data["Heures"],
        data["Notes"],
        marker='o',
        linestyle = ' ',
        color='blue'
    )
    
    # Droite de regresion :
    x_vals = data["Heures"]
    y_vals = linear_regression(x_vals, beta_0, beta_1)
    plt.plot(x_vals,
             y_vals,
             color='red',
             label='Régression')
    
    # Titrage et labels :
    plt.title("Évolution des notes en fonction des heures d'étude")
    plt.xlabel("Heures d'étude")
    plt.ylabel("Notes obtenues")
    plt.grid(True)
    plt.xlim(0, data["Heures"].max() + 1)
    plt.ylim(0, data["Notes"].max() + 1)

    # Sauvegarde le graphique :
    plt.savefig("regression_lineaire.png")    
    return


#------------------------------------------------------------------------------
def pente(data: pd.DataFrame) -> float:
    """
    Calcule la pente (coefficient directeur) beta_1 d'une régression linéaire simple à l'aide de la méthode des moindres carrés.

    La formule utilisée est :
        beta_1 = Σ[(Xi - mean(X)) * (Yi - mean(Y))] / Σ[(Xi - mean(X))²]

    Parameters
    ----------
    data : pandas.DataFrame
        Un DataFrame contenant au moins deux colonnes nommées :
        - **"Heures"** : les variables explicatives (X)
        - **"Notes"**  : les variables à prédire (Y)

    Returns
    -------
    float
        La valeur de la pente (beta_1) du modèle de régression linéaire.

    Raises
    ------
    KeyError
        Si les colonnes "Heures" ou "Notes" sont absentes du DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Heures": [1, 2, 3], "Notes": [6, 6.5, 7.5]})
    >>> pente(df)
    0.75
    """

    # Moyenne des notes :
    mean_note = data["Notes"].mean()
    print("\n- Moyenne des notes : ", mean_note)
    
    # Moyenne de heures :
    mean_heure = data["Heures"].mean()
    print("- Moyenne des heure : ", mean_heure, "\n")
    
    # Calcul préalable :
    data["Xi - mean_heure"] = [(x - mean_heure) for x in data["Heures"]]
    data["Yi - mean_note"] = [(y - mean_note) for y in data["Notes"]]
    print("Rajout de colonnes :\n", data)
    
    # Calcule la pente (coefficient directeur) beta_1 :
    try:
        a = 0.0
        b =0.0
        for x, y in zip(data["Xi - mean_heure"], data["Yi - mean_note"]):
            a += x * y
        
        for x in data["Xi - mean_heure"]:
            b += x * x
            
        beta_1 = a / b
    except Exception as e:
        print("Erreur de calcul : ", e)
        raise

    print(f"\n- Beta_1 = a / b soit {a} / {b}")
    print("- Valeur de la pente (Beta_1) : ", beta_1)
    return(beta_1)


#------------------------------------------------------------------------------
def origin(data: pd.DataFrame, beta_1: float) -> float:
    """
    Calcule l'ordonnée à l'origine (beta_0) d'une régression linéaire simple.

    Utilise la formule :
        beta_0 = mean(Y) - beta_1 * mean(X)

    où :
        - Y est la variable cible (ici "Notes")
        - X est la variable explicative (ici "Heures")
        - beta_1 est la pente déjà calculée

    Parameters
    ----------
    data : pandas.DataFrame
        Un DataFrame contenant au moins deux colonnes nommées :
        - "Heures" : les variables explicatives (X)
        - "Notes"  : les variables à prédire (Y)

    beta_1 : float
        La pente (coefficient directeur) calculée précédemment via la méthode des moindres carrés.

    Returns
    -------
    float
        L'ordonnée à l'origine (beta_0) du modèle de régression linéaire.

    Raises
    ------
    KeyError
        Si les colonnes "Heures" ou "Notes" sont absentes du DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Heures": [1, 2, 3], "Notes": [6, 6.5, 7.5]})
    >>> beta_1 = 0.75
    >>> origin(df, beta_1)
    5.25
    """
    
    y = data["Notes"].mean()
    x = data["Heures"].mean()
    
    beta_0 = y - beta_1 * x
    print("- Valeur du point a l'origine : ", beta_0)
    
    return beta_0


#------------------------------------------------------------------------------
def linear_regression(heure: float, beta_0: float, beta_1: float) -> float:
    """
    Calcule la prédiction d'une note à partir d'un modèle de régression linéaire simple.

    Utilise la formule :
        y = beta_0 + beta_1 * x

    où :
        - **y** est la note prédite,
        - **x** (ici `heure`) est le nombre d'heures d'étude,
        - **beta_0** est l'ordonnée à l'origine,
        - **beta_1** est la pente de la droite de régression.

    Parameters
    ----------
    heure : float
        Le nombre d'heures d'étude pour lequel on souhaite estimer une note.
    beta_0 : float
        L'ordonnée à l'origine du modèle de régression linéaire.
    beta_1 : float
        La pente (coefficient directeur) du modèle.

    Returns
    -------
    float
        La note prédite selon le modèle de régression.

    Examples
    --------
    >>> linear_regression(2.5, 5.0, 0.8)
    La note statistique pour 2.5 travaillée : 7.0
    7.0
    """
    
    predict = beta_0 + beta_1 * heure
    return predict


#------------------------------------------------------------------------------
def main():
    """
    Fonction principale exécutant les étapes d'une régression linéaire simple :

    Étapes :
    --------
    1. Chargement du fichier CSV (`predict_note.csv`)
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

    # Étape 1 : Chargement et affichage des données + nuage de points :
    try:
        data = pd.read_csv("predict_note.csv")
        save_graph(data)
        print(data)
    except Exception as e:
        print("Erreur de chargement : ", e)
        return 1

    try:
        # Étape 2 : Calcul de la pente :
        beta_1 = pente(data)

        # Étape 3 : Calcul de l’ordonnée à l’origine :
        beta_0 = origin(data, beta_1)

        # Étape 4 : Prédiction pour X heures :
        heure = 2.5
        predict = linear_regression(heure, beta_0, beta_1)
        print(f"\nLa note prédit pour {heure} heures travaillée : {predict}/10")

        # Étape 5 : Affichage du graphique avec la droite de régression :
        save_regression_graph(data, beta_0, beta_1)

    except Exception as e:
        print("Erreur lors du calcul de la régression :", e)
        return 2

    return 0


###############################################################################
if __name__ == "__main__":
    sys.exit(main())