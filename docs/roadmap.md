# Gloomy — travaux à venir

Cette roadmap ne liste que les travaux non terminés. Les fonctionnalités déjà livrées sont récapitulées dans le [journal des changements](../CHANGELOG.md); les descriptions techniques détaillées restent dans les guides.

La priorité absolue du projet est de développer puis valider un modèle de langage capable d'apprendre à partir de textes, de suivre des consignes, de répondre à des questions et de générer du texte cohérent. Le GLOOMY actuel ne traite que des vecteurs numériques via des couches Dense; il n'est pas encore un modèle de langage. La réussite devra être évaluée sur des tâches et des corpus de test définis, pas seulement à partir de quelques exemples générés.

## 1. Développer un véritable modèle de langage

- définir les corpus d'entraînement, les tâches de compréhension et génération, les jeux de validation/test séparés et les ressources de calcul disponibles ;
- construire un pipeline texte avec normalisation Unicode, tokenizer sous-mot versionné, vocabulaire, séquences de contexte et masques d'entraînement ;
- implémenter une architecture séquentielle adaptée au langage, en priorité un Transformer causal avec embeddings, attention, connexions résiduelles et normalisation ;
- entraîner à prédire le prochain token avec une perte cross-entropy, des batches, un optimiseur adapté, la reprise de checkpoints et un suivi de la perplexité ;
- ajouter au CLI l'entraînement sur corpus, le prompt et la génération autoregressive avec options de décodage reproductibles ;
- sauvegarder ensemble tokenizer, configuration, poids et état d'entraînement dans un format de modèle de langage versionné ;
- évaluer les réponses et la génération sur des textes jamais vus, avec des métriques de langage et des tâches de compréhension, et publier les limites, la taille du modèle, les données et le coût de calcul.

Le premier modèle expérimental devra valider toute la chaîne sur une petite échelle. La priorité de long terme reste un modèle de langage évalué sur la compréhension; un prototype qui ne fait que compléter des caractères ou mémoriser son corpus ne suffira pas.

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
