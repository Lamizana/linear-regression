# Moindres carrés (OLS)

> Partie 1 — régression linéaire simple par la **méthode des moindres carrés**
> (solution analytique, sans optimisation itérative).

## Principe

Étant donné un nuage de points `(x, y)`, la droite de régression `y = beta_0 + beta_1 * x`
minimise la somme des carrés des écarts. Les coefficients sont calculés directement :

```
beta_1 = Σ[(Xi − mean(X)) · (Yi − mean(Y))] / Σ[(Xi − mean(X))²]
beta_0 = mean(Y) − beta_1 · mean(X)
```

## Scripts

### Prédiction de notes (`predict_note_ecole/`)

Prédit une note (`/10`) en fonction des heures d'étude.

- Données : `predict_note.csv` (colonnes `Heures`, `Notes`)
- Sorties : `relation_notes_heures.png`, `regression_lineaire.png`
- Prédiction pour 2,5 h d'étude

```bash
cd predict_note_ecole && python3 predict_note.py
```

### Prédiction de prix de voiture (`predict_prix_voiture/`)

Prédit le prix d'une voiture (`€`) en fonction de son kilométrage.

- Données : `predict_prix.csv` (colonnes `km`, `price`)
- Sortie : `regression_lineaire_voiture.png`
- Prédiction pour 20 000 km

```bash
cd predict_prix_voiture && python3 predict_prix.py
```

## Bibliothèques

- `pandas` : chargement des données CSV
- `matplotlib` : visualisation (nuage de points + droite de régression)