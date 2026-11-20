# Gloomy — travaux à venir

Cette roadmap ne liste que les travaux non terminés. Les fonctionnalités déjà livrées sont récapitulées dans le [journal des changements](../CHANGELOG.md); les descriptions techniques détaillées restent dans les guides.

La priorité retenue pour la suite est la latence et le coût CPU, en comparant chaque gain à la qualité prédictive et à son coût mémoire.

## 1. Définir les objectifs et le protocole de mesure

- utiliser comme baseline provisoire le workload Dense `1 -> 8 -> 1` : régression synthétique, `400` exemples d'entraînement, `80` de validation, batch/replay `8`, capacités mémoire `32/64/128/256`, trois seeds ;
- utiliser A → B → A comme workload de rétention : `40` observations par régime, dix passages par phase, `50` updates par phase et capacité mémoire `64` ;
- préciser les workloads applicatifs, dimensions d'entrée/sortie et la cible matérielle avant de fixer des seuils de qualité, latence, débit et RAM ;
- compléter le profil par des workloads applicatifs et une mesure sur cible ; la mesure globale `inference_time_us` inclut toujours évaluation, perte et métriques ;
- mesurer le pic RSS, les allocations et les coûts sur cible lorsque le matériel sera défini ;
- conserver les métriques brutes et des baselines comparables plutôt qu'un score composite opaque.

## 2. Réduire la latence et le coût CPU

- comparer les optimisations sur les mêmes données, seeds, architecture et nombre d'updates ;
- ne retenir un gain de vitesse qu'avec son impact mesuré sur la qualité et la mémoire.

## 3. Réduire les coûts du pipeline de données et de mémoire

- mesurer les compromis des stratégies de replay et des formats compacts sur des données représentatives.

## 4. Évaluer les précisions réduites

- comparer int8/int16 sur des jeux de données représentatifs, avec erreurs après déquantification et métriques de tâche ;
- étudier ensuite la quantification des poids et du calcul, qui nécessite un protocole distinct ;
- mesurer qualité, latence, mémoire et coût de conversion ensemble avant de choisir un format.

## 5. Renforcer l'évaluation et le continual learning

- élargir le benchmark au-delà des fonctions synthétiques, avec séparation temporelle ou expérimentale adaptée ;
- ajouter des workloads applicatifs représentatifs avec séparation temporelle ou expérimentale adaptée ;
- étudier la correction du biais d'échantillonnage Prioritized Replay ;
- généraliser les tests mathématiques : gradients sur activations/couches, stabilité et reproductibilité.

## 6. Valider sur une cible embarquée

- choisir une cible concrète avant de déclarer un gain embarqué ;
- mesurer latence, RAM, Flash et énergie sur l'appareil ;
- valider compilation, compatibilité numérique et résultats sur cette cible.

## 7. Clarifier les métadonnées du dépôt

- confirmer la licence du projet et ajouter le fichier légal correspondant si nécessaire; le README historique annonçait MIT, mais aucun fichier `LICENSE` n'est présent.
