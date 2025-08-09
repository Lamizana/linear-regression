# Ft_linear_regression

> Created by alex lamizana in 31/05/2025
> Mise a jour : 26/06/2025

Introduction Introduction machine learning.
Dans ce projet, on va mettre en œuvre notre premier algorithme d'apprentissage automatique.

```bash
> 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
```

----------------------------------------------------------------------------

## Avant-propos

Ce que je pense être la meilleure définition de l'apprentissage automatique :

« On dit d'un programme informatique qu'il apprend de l'expérience E en ce qui concerne une certaine
classe de tâches T et une mesure de performance P, si sa performance pour des tâches dans
T, telle que mesurée par P, s'améliore avec l'expérience E. »

Tom M. Mitchell

## Instruction

L'apprentissage automatique est un domaine de l'informatique en plein essor qui peut sembler un peu compliqué et réservé aux mathématiciens. Vous avez peut-être entendu parler des réseaux neuronaux ou du regroupement k-means, mais vous ne comprenez pas comment ils fonctionnent ni comment coder ce type d'algorithmes...

Mais ne vous inquiétez pas, nous allons commencer par un algorithme simple et basique d'apprentissage automatique.

## Objectif

L'objectif de ce projet est de nous initier au concept de base de l'apprentissage automatique.

Pour ce projet, vous devrez créer un programme qui prédit le prix d'une voiture par
en utilisant un train de fonctions linéaires ***(linear function)*** avec un algorithme de descente de gradient ***(gradient descent algorithm)***.
Nous travaillerons sur un exemple précis pour ce projet, mais une fois que vous aurez terminé, vous serez en mesure d'utiliser l'algorithme avec n'importe quel autre ensemble de données.

## Instruction générales

Dans ce projet, vous êtes libre d'utiliser le langage que vous voulez.

Vous êtes également libre d'utiliser les bibliothèques de votre choix, **à condition qu'elles ne fassent pas tout le travail à votre place**. Par exemple, l'utilisation de numpy.polyfit de python est considérée comme une tricherie.

> [!NOTE]
> Vous devriez utiliser un langage qui vous permet de visualiser facilement vos données: cela vous sera très utile pour le débogage.

## Partie obligatoire

Vous allez mettre en œuvre une régression linéaire simple avec une seule caractéristique - dans ce cas, le kilométrage de la voiture .

Pour ce faire, vous devez créer deux programmes :

- Le premier programme sera utilisé pour prédire le prix d'une voiture pour un kilométrage donné. Lorsque vous lancez le programme, il doit vous demander un kilométrage, puis vous donner le prix estimé pour ce kilométrage. 
Le programme utilisera l'hypothèse suivante pour prédire le prix :

  - ```estimateP rice(mileage) = θ0 + (θ1 ∗ mileage)```

    - θ₀ : le prix de départ (si la voiture a 0 km),
    - θ₁ : la perte de valeur par kilomètre.

Le but de l’apprentissage est de trouver les meilleurs θ₀ et θ₁ pour que cette équation corresponde le mieux possible à tes données réelles.

> Avant l'exécution du programme d'entraînement, theta0 et theta1 sont mis à 0.

- Le second programme sera utilisé pour entraîner votre modèle. Il lira votre fichier de données et effectuera une régression linéaire sur les données.
Une fois la régression linéaire terminée, vous enregistrerez les variables **theta0 et theta1** pour les utiliser dans le premier programme.
Vous utiliserez les formules suivantes :

> [!IMPORTANT]
> 

## Niveau minimal des logs affichés (INFO, DEBUG, WARNING, ERROR, CRITICAL)

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
```