# Gloomy — travaux à venir

Cette roadmap ne liste que les travaux non terminés. Les fonctionnalités déjà livrées sont récapitulées dans le [journal des changements](../CHANGELOG.md); les descriptions techniques détaillées restent dans les guides.

La priorité absolue du projet est de développer puis valider un modèle de langage capable d'apprendre à partir de textes, de suivre des consignes, de répondre à des questions et de générer du texte cohérent. La réussite devra être évaluée sur des tâches et des corpus de test autorisés et définis, pas seulement à partir de quelques exemples générés.

## 1. Développer un véritable modèle de langage

- définir des corpus applicatifs autorisés à partir du domaine cible, puis remplacer le mini-corpus original de test logiciel, qui n'est pas représentatif d'une application ;
- compléter les tâches contrôlées de prédiction du prochain token et QA factuelle par des tâches de compréhension/génération définies avec leurs références et critères ;
- valider le Transformer causal sur ces données représentatives et améliorer sa robustesse/performance sans dégrader les mesures tenues à l'écart ;
- mesurer perplexité, stabilité et tâches linguistiques sur les splits documentaires autorisés; vérifier la reprise bit-à-bit aux frontières d'epoch sur des corpus représentatifs ;
- publier les données, tailles, empreintes, seeds, configuration, qualité, latence, débit et mémoire; reproduire ces mesures sur la cible matérielle applicative lorsqu'elle sera définie.

## 2. Définir les objectifs et le protocole de mesure

- remplacer les proxies synthétiques par des données applicatives réelles, leurs dimensions d'entrée/sortie et la cible de déploiement avant de fixer les seuils de qualité, latence, débit et RAM ;
- compléter les profils synthétiques par les workloads réels sur la cible de déploiement.

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

## Contrainte permanente

L'exécutable Gloomy ne doit utiliser aucune bibliothèque qui ne puisse pas être liée statiquement au programme. Les scripts de développement peuvent utiliser la bibliothèque standard de leur interpréteur, mais ne sont pas des dépendances du binaire.
