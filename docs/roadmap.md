# Gloomy — travaux à venir

Cette roadmap ne liste que les travaux non terminés. Les fonctionnalités déjà livrées sont récapitulées dans le [journal des changements](../CHANGELOG.md); les descriptions techniques détaillées restent dans les guides.

La priorité retenue pour la suite est la latence et le coût CPU, en comparant chaque gain à la qualité prédictive et à son coût mémoire.

## 1. Définir les objectifs et le protocole de mesure

- préciser les workloads représentatifs, les dimensions d'entrée/sortie et la cible matérielle ;
- fixer des seuils mesurables de qualité, latence, débit et mémoire ;
- profiler séparément inférence, entraînement, replay, copies et allocations avant de choisir une optimisation ;
- ajouter des mesures de percentiles de latence, pic RSS et coûts sur cible lorsque le matériel sera défini ;
- conserver les métriques brutes et des baselines comparables plutôt qu'un score composite opaque.

## 2. Réduire la latence et le coût CPU

- étudier des buffers contigus, préalloués et réutilisables pour les poids, activations et gradients ;
- réduire les allocations et copies dans les couches et les chemins d'inférence/entraînement ;
- comparer les optimisations sur les mêmes données, seeds, architecture et nombre d'updates ;
- ne retenir un gain de vitesse qu'avec son impact mesuré sur la qualité et la mémoire.

## 3. Réduire les coûts du pipeline de données et de mémoire

- éviter de conserver simultanément le dataset brut et une copie normalisée lorsqu'un traitement en flux est possible ;
- étudier un parsing incrémental et un pipeline borné en mémoire ;
- proposer des limites mémoire en octets, en tenant compte des métadonnées, paramètres de quantification et allocations ;
- mesurer les compromis des stratégies de replay et des formats compacts sur des données représentatives.

## 4. Séparer les formats de persistance

- concevoir un artefact inference-only minimal, distinct du checkpoint de reprise d'entraînement ;
- définir les données strictement nécessaires à l'inférence, le versionnement et la compatibilité ;
- mesurer taille, temps de chargement et latence du runtime d'inférence compact ;
- garder l'état de l'optimiseur, la mémoire de replay et l'état aléatoire dans le format de reprise lorsque nécessaire.

## 5. Évaluer les précisions réduites

- comparer int8/int16 sur des jeux de données représentatifs, avec erreurs après déquantification et métriques de tâche ;
- étudier ensuite la quantification des poids et du calcul, qui nécessite un protocole distinct ;
- mesurer qualité, latence, mémoire et coût de conversion ensemble avant de choisir un format.

## 6. Renforcer l'évaluation et le continual learning

- élargir le benchmark au-delà des fonctions synthétiques, avec séparation temporelle ou expérimentale adaptée ;
- tester des dérives progressives, plusieurs régimes et des workloads représentatifs ;
- étudier la correction du biais d'échantillonnage Prioritized Replay ;
- faire évoluer Hybrid vers une politique explicitement évaluée utilisant les signaux d'importance réellement disponibles ;
- généraliser les tests mathématiques : gradients sur activations/couches, stabilité et reproductibilité.

## 7. Valider sur une cible embarquée

- choisir une cible concrète avant de déclarer un gain embarqué ;
- mesurer latence, RAM, Flash et énergie sur l'appareil ;
- valider compilation, compatibilité numérique et résultats sur cette cible.

## 8. Clarifier les métadonnées du dépôt

- confirmer la licence du projet et ajouter le fichier légal correspondant si nécessaire; le README historique annonçait MIT, mais aucun fichier `LICENSE` n'est présent.
