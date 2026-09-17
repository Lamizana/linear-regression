<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-1.24-010101?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Matplotlib-3.7-FFCA28?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Topic-Machine%20Learning-2088FF?style=for-the-badge&logoColor=white" alt="Machine Learning">
</p>

# Ft_linear_regression

> Created by alex lamizana in 31/05/2025

> Mise a jour : 15/09/2026

Introduction au machine learning.
Dans ce projet, on va mettre en œuvre notre premier algorithme d'apprentissage automatique.

---

## Avant-propos

Ce que je pense être la meilleure définition de l'apprentissage automatique :

« On dit d'un programme informatique qu'il apprend de l'expérience E en ce qui concerne une certaine
classe de tâches T et une mesure de performance P, si sa performance pour des tâches dans
T, telle que mesurée par P, s'améliore avec l'expérience E. »

Tom M. Mitchell

---

## Instruction

L'apprentissage automatique est un domaine de l'informatique en plein essor qui peut sembler un peu compliqué et réservé aux mathématiciens. Vous avez peut-être entendu parler des réseaux neuronaux ou du regroupement k-means, mais vous ne comprenez pas comment ils fonctionnent ni comment coder ce type d'algorithmes...

Mais ne vous inquiétez pas, nous allons commencer par un algorithme simple et basique d'apprentissage automatique.

---

## Objectif

L'objectif de ce projet est de nous initier au concept de base de l'apprentissage automatique.

Pour ce projet, vous devrez créer un programme qui prédit le prix d'une voiture en utilisant une fonction linéaire ***(linear function)*** entraînée par un algorithme de descente de gradient ***(gradient descent algorithm)***.
Nous travaillerons sur un exemple précis pour ce projet, mais une fois que vous aurez terminé, vous serez en mesure d'utiliser l'algorithme avec n'importe quel autre ensemble de données.

---

## Instruction générales

Dans ce projet, vous êtes libre d'utiliser le langage que vous voulez.

Vous êtes également libre d'utiliser les bibliothèques de votre choix, **à condition qu'elles ne fassent pas tout le travail à votre place**. Par exemple, l'utilisation de numpy.polyfit de python est considérée comme une tricherie.

> [!NOTE]
> Vous devriez utiliser un langage qui vous permet de visualiser facilement vos données: cela vous sera très utile pour le débogage.

---

## AI Instructions

### Contexte

L'IA est désormais un partenaire de codage puissant — aux côtés de vos pairs — pour aborder des projets vastes et exigeants. Vous la guiderez à travers les aspects techniques et non techniques de votre travail.

Les outils d'IA peuvent booster votre efficacité et améliorer la qualité de vos livrables, mais vous devez être capable d'explorer en profondeur n'importe quelle partie du projet sans dépendre d'eux.

Votre partenaire IA vous soutient, mais vous restez entièrement responsable des décisions techniques éclairées, que vous devez être capable d'expliquer et de défendre clairement.

### Message principal

- Visez une utilisation mature et responsable de l'IA.
- Ne laissez jamais l'IA prendre les décisions à votre place — surtout lorsqu'elle n'a pas connaissance de vos objectifs, contraintes ou dynamiques d'équipe.
- Maintenez créativité, innovation et supervision humaine grâce à une collaboration active avec vos pairs. L'IA est entraînée sur des données existantes et génère rarement de véritables nouvelles idées.
- Restez informé des tendances émergentes et soyez prêt à vous adapter à de nouveaux concepts et technologies.

### Règles d'apprentissage

- Gardez la direction intellectuelle de vos projets et prenez vos propres décisions éclairées.
- Priorisez l'intelligence collective de votre équipe et de vos pairs.
- Restez activement informé de l'évolution continue des technologies IA.

### Impacts de la phase

- Compétences en ingénierie IA.
- Efficacité accrue.
- Plus grande fiabilité et qualité.
- Un état d'esprit pionnier.

### Commentaires et exemples

- Vos pairs peuvent identifier les compromis, remettre en question les hypothèses et vous aider à vous améliorer. La première réponse d'une IA n'est pas forcément la meilleure — elle peut manquer d'efficacité, de sécurité ou de réelle valeur ajoutée. Plus que jamais, vous devez vous appuyer sur vos pairs.
- L'IA peut vous rendre plus rapide, mais vos pairs vous rendent meilleur. La collaboration, la discussion et la remise en question mutuelle sont la clé du succès.
- Soyez transparent sur l'utilisation de l'IA dans vos projets et identifiez clairement ce qui a été généré par des outils IA.

> [!IMPORTANT]
> ✓ **Bonne pratique**: j'ai demandé à l'IA de m'aider à générer des tests unitaires pour mon API. Je les ai relus avec mon coéquipier, et nous les avons ajustés pour couvrir les cas limites. Cela a fait gagner du temps, et nous avons tous les deux appris quelque chose de nouveau.

> [!WARNING]
> ✗ **Mauvaise pratique**: j'ai demandé à l'IA de générer toute l'architecture de mon projet. Elle « fonctionne », mais lorsqu'on me demande d'expliquer les choix de conception lors de la soutenance, je ne peux pas. Je perds en crédibilité et j'échoue.

---

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

```console
tmpθ0 = learningRate × (1/m) × Σ (estimatePrice(mileage[i]) − price[i]),     pour i de 0 à m−1

tmpθ1 = learningRate × (1/m) × Σ (estimatePrice(mileage[i]) − price[i]) × mileage[i],   pour i de 0 à m−1
```

Je vous laisse deviner ce que vaut m 😉

> [!IMPORTANT]
> Notez que l'estimation du prix (estimatePrice) est la même que dans notre premier programme, mais ici, 
> elle utilise vos valeurs temporaires theta0 et theta1 calculées en dernier.
> N'oubliez pas non plus de mettre à jour simultanément theta0 et theta1.

---

## Partie Bonus

Voici quelques bonus qui pourraient vous être très utiles :
    - Représenter les données sous forme de graphique pour voir leur répartition.
    - Représenter la ligne résultant de votre régression linéaire dans le même graphique, pour voir
    le résultat de votre travail !
    - Un programme qui calcule la précision de votre algorithme.

> [!IMPORTANT]
> La partie bonus ne sera évaluée que si la partie obligatoire est PARFAITE. Parfaite signifie que
> la partie obligatoire a été intégralement réalisée et fonctionne sans défaillance. Si vous n'avez pas
> validé toutes les exigences obligatoires, votre partie bonus ne sera pas du tout évaluée.

---
