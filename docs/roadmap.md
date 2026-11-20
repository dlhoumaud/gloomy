# Gloomy — travaux à venir

Cette roadmap ne liste que les travaux non terminés. Les fonctionnalités déjà livrées sont récapitulées dans le [journal des changements](../CHANGELOG.md); les descriptions techniques détaillées restent dans les guides.

La priorité retenue pour la suite est la latence et le coût CPU, en comparant chaque gain à la qualité prédictive et à son coût mémoire.

## 1. Définir les objectifs et le protocole de mesure

- remplacer les proxies synthétiques par des données applicatives réelles, leurs dimensions d'entrée/sortie et la cible de déploiement avant de fixer les seuils de qualité, latence, débit et RAM ;
- compléter les profils synthétiques par les workloads réels sur la cible de déploiement.

## 2. Réduire les coûts du pipeline de données et de mémoire

- mesurer les compromis des stratégies de replay et des formats compacts sur des données représentatives.

## 3. Évaluer les précisions réduites

- comparer int8/int16 sur des jeux de données représentatifs, avec erreurs après déquantification et métriques de tâche ;
- étudier ensuite la quantification des poids et du calcul, qui nécessite un protocole distinct ;
- mesurer qualité, latence, mémoire et coût de conversion ensemble avant de choisir un format.

## 4. Renforcer l'évaluation et le continual learning

- élargir le benchmark au-delà des fonctions synthétiques, avec séparation temporelle ou expérimentale adaptée ;
- ajouter des workloads applicatifs représentatifs avec séparation temporelle ou expérimentale adaptée ;
- étudier la correction du biais d'échantillonnage Prioritized Replay ;
- généraliser les tests mathématiques : gradients sur activations/couches, stabilité et reproductibilité.

## 5. Valider sur une cible embarquée

- choisir une cible concrète avant de déclarer un gain embarqué ;
- mesurer latence, RAM, Flash et énergie sur l'appareil ;
- valider compilation, compatibilité numérique et résultats sur cette cible.

## 6. Clarifier les métadonnées du dépôt

- confirmer la licence du projet et ajouter le fichier légal correspondant si nécessaire; le README historique annonçait MIT, mais aucun fichier `LICENSE` n'est présent.
