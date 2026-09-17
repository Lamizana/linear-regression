<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-1.26-010101?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-2.2-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Matplotlib-3.9-FFCA28?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Topic-Machine%20Learning-2088FF?style=for-the-badge&logoColor=white" alt="Machine Learning">
</p>

# Linear Regression

> Created by Alex Lamizana on 31/05/2025

> Dernière mise à jour : 15/09/2026

Introduction au machine learning : mise en œuvre d'une **régression linéaire simple** selon **deux approches différentes**, organisées en deux parties indépendantes :

| Approche | Dossier | Méthode | Données |
| --- | --- | --- | --- |
| **Moindres carrés** (OLS) | `method_moindre_carre/` | Formule analytique (solution fermée) | Notes d'école + prix de voiture |
| **Descente de gradient** | `method_decente_gradient/` | Optimisation itérative (10 000 étapes) | Prix de voiture |

---

## Structure

```console
linear-regression/
├── requirements.txt
├── method_moindre_carre/            # Partie 1 : moindres carrés (OLS)
│   ├── predict_note_ecole/          # Prédiction de notes selon les heures d'étude
│   └── predict_prix_voiture/        # Prédiction du prix d'une voiture selon le kilométrage
└── method_decente_gradient/         # Partie 2 : descente de gradient
    ├── entrainement.py              # Entraîne le modèle → thetas.json
    ├── predict_prix.py              # Prédiction interactive du prix
    ├── bonus.py                     # Évaluation (MSE, MAE, R²) + visualisation
    ├── logger.py                    # Logging coloré partagé
    └── utils.py                     # Fonctions partagées (chargement CSV, lecture des thetas)
```

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> [!NOTE]
> Les images (`*.png`) et le fichier `thetas.json` sont générés à l'exécution et sont ignorés par git.

---

## Partie 1: Moindres carrés (`method_moindre_carre/`)

Régression linéaire par **solution analytique** (OLS) : calcul direct de la pente `beta_1`
et de l'ordonnée à l'origine `beta_0`.

- **Notes d'école** : prédit une note (`/10`) en fonction des heures d'étude. Le nombre d'heures est configurable via la constante `HEURES` en tête de `predict_note.py` (défaut : `25` h).

  ```bash
  cd method_moindre_carre/predict_note_ecole && python3 predict_note.py
  ```

- **Prix de voiture** : prédit un prix pour 20 000 km.

  ```bash
  cd method_moindre_carre/predict_prix_voiture && python3 predict_prix.py
  ```

Chaque script enregistre un nuage de points et le graphique avec la droite de régression.

---

## Partie 2: Descente de gradient (`method_decente_gradient/`)

Régression linéaire par **descente de gradient** sur données normalisées
(min-max vers `[0, 1]`), puis dénormalisation des paramètres `theta0` / `theta1`.

1. **Entraînement** : lit `data.csv`, ajuste les paramètres, sauvegarde `thetas.json`.

   ```bash
   cd method_decente_gradient && python3 entrainement.py
   ```

2. **Prédiction** : demande un kilométrage et estime le prix (`theta0 + theta1 * mileagge`).

   ```bash
   cd method_decente_gradient && python3 predict_prix.py
   ```

3. **Bonus** : trace la droite de régression et calcule MSE, MAE et R².

   ```bash
   cd method_decente_gradient && python3 bonus.py
   ```

Le détail du sujet (énoncé, formules, consignes) se trouve dans [**README.md**](method_decente_gradient/README.md).

---

## Comparaison des deux méthodes

| Aspect | Moindres carrés | Descente de gradient |
| --- | --- | --- |
| ***Résolution*** | Analytique (formule fermée) | Itérative (10 000 étapes) |
| ***Normalisation*** | Aucune | Min-max vers `[0, 1]` |
| ***Persistance des paramètres*** | Aucune (calcul à la volée) | `thetas.json` |
| ***Métriques*** | Aucune (visuel uniquement) | MSE, MAE, R² |
| ***Type de code*** | Script autonome par cas d'usage | Modulaire (entraînement / prédiction / évaluation) |
