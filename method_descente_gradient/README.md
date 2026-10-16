# Descente de gradient

> Partie 2 de la régression linéaire simple par **descente de gradient**
> (optimisation itérative sur données normalisées).

## Principe

On cherche la droite `estimatePrice(mileage) = theta_0 + theta_1 * mileage` qui minimise l'erreur.
Contrairement aux moindres carrés, les paramètres sont ajustés **par itérations** :

1. Les variables sont normalisées **min-max** vers `[0, 1]` pour stabiliser la convergence.
2. À chaque itération, on calcule le gradient moyen de l'erreur pour `theta_0` et `theta_1`, puis on
   met à jour les deux paramètres **simultanément** :

```
gradient_theta_0 = (1 / m) · Σ (y_pred − y)
gradient_theta_1 = (1 / m) · Σ (y_pred − y) · x_norm

theta_0 -= learningRate · gradient_theta_0
theta_1 -= learningRate · gradient_theta_1
```

3. Une fois l'entraînement terminé, `theta_0` et `theta_1` sont **dénormalisés** pour revenir à
   l'échelle d'origine des données.

## Scripts

### Entraînement (`entrainement.py`)

Lit `data.csv`, entraîne le modèle (10 000 itérations, learning rate `0.01`) et sauvegarde les
paramètres.

- Données : `data.csv` (colonnes `km`, `price`)
- Sortie : `thetas.json` (`theta0`, `theta1`)
- Paramètres modifiables en tête du script : `ITERATIONS`, `LEARNING_RATE`

```bash
cd method_descente_gradient && python3 entrainement.py
```

### Prédiction (`predict_prix.py`)

Demande un kilométrage puis estime le prix via `theta_0 + theta_1 * mileage`.

- Entrée : `thetas.json` (produit par `entrainement.py`)
- Sortie : prix estimé affiché dans la console

```bash
cd method_descente_gradient && python3 predict_prix.py
```

### Bonus (`bonus.py`)

Trace le nuage de points et la droite de régression, puis calcule les métriques de précision.

- Entrées : `data.csv`, `thetas.json`
- Sortie : `regression_lineaire.png` (nuage de points + droite)
- Métriques : MSE, MAE, R²

```bash
cd method_descente_gradient && python3 bonus.py
```

> Les modules `logger.py` (logging coloré) et `utils.py` (chargement CSV, lecture des thetas) sont
> partagés par les trois scripts ; ils ne s'exécutent pas seuls.

## Bibliothèques

- `pandas` : chargement des données CSV
- `numpy` : calcul matriciel et vectorisation de la descente de gradient
- `matplotlib` : visualisation (nuage de points + droite de régression)

---

> L'énoncé officiel du projet (formules, consignes, AI Instructions) est conservé dans
> [`enonce.md`](enonce.md).
