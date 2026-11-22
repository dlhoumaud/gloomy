# Gloomy — travaux à venir

Cette roadmap ne liste que les travaux non terminés. Les fonctionnalités déjà livrées sont récapitulées dans le [journal des changements](../CHANGELOG.md); les descriptions techniques détaillées restent dans les guides.

La priorité absolue du projet est de développer puis valider un modèle de langage capable d'apprendre à partir de textes, de suivre des consignes, de répondre à des questions et de générer du texte cohérent. Sa réussite doit être évaluée sur des tâches applicatives et des corpus autorisés, définis avec leurs utilisateurs.

## 1. Développer et valider un modèle de langage

L'objectif à terme est un pipeline en trois étapes, distinct du prototype documentaire actuel :

### 1.1 Pré-entraînement en français

- sélectionner et documenter des corpus français autorisés, assez divers pour couvrir vocabulaire, grammaire, syntaxe et connaissances générales; dédupliquer les sources et vérifier les contaminations train/validation/test ;
- faire évoluer le tokenizer, le modèle, le chargement des données et l'entraînement pour la taille de corpus et le budget matériel visés, tout en conservant checkpoints, reprise déterministe et mesures de coût ;
- évaluer la langue et les connaissances sur des jeux français tenus à l'écart; compléter perplexité et bits par octet par des tâches distinctes de compréhension et de connaissances.

### 1.2 Fine-tuning supervisé (SFT)

- définir un format de dialogue français explicite pour les tours système, utilisateur et assistant, ainsi que les marqueurs de fin de tour ;
- constituer des données autorisées d'instructions, de dialogue et de raisonnement, avec références et splits par source; entraîner sur les réponses assistant sans apprendre à recopier les prompts ;
- évaluer séparément suivi d'instructions, qualité des réponses et tâches de raisonnement sur des exemples non vus, avec critères validés par les utilisateurs.

### 1.3 Alignement

- définir les préférences attendues et les consignes de comportement avec les utilisateurs, puis recueillir des comparaisons de réponses annotées selon ces critères ;
- choisir et implémenter une méthode d'alignement adaptée au modèle et aux ressources (par exemple optimisation directe sur préférences ou apprentissage par récompense) ;
- comparer avant/après alignement la pertinence, cohérence, qualité et conformité des réponses sur des évaluations tenues à l'écart, et surveiller les régressions sur les capacités acquises en pré-entraînement et SFT.

- convenir avec les utilisateurs métier des tâches, formats d'entrée/sortie, corpus autorisés, références et seuils d'acceptation; faire valider le protocole QA provisoire avant de traiter ses seuils comme des critères applicatifs ;
- dépasser de façon reproductible la baseline majoritaire QA de 50 % sur les tâches documentaires, sans dégrader les bits par octet, puis confirmer le gain sur le test tenu à l'écart ;
- ajouter des workloads représentatifs du domaine applicatif réel au-delà des corpus documentaires actuels, avec séparations adaptées et mesures multi-seed de QA et bits par octet ;
- reproduire qualité, latence, débit, RAM et empreinte du modèle sur une cible de déploiement choisie avec les utilisateurs.

## Priorité haute — travaux transversaux pour le modèle de langage

- répéter les mesures float32/int8/int16 du Transformer sur plusieurs seeds et workloads documentaires représentatifs, avec erreur numérique, qualité, latence, RSS et coût d'évaluation ;
- mesurer sur la cible retenue les opérations entières; étudier un checkpoint compact si une réduction persistante de Flash/RAM est nécessaire, car les poids du checkpoint restent stockés en float64 ;
- étendre le pilote de replay de séquences à plusieurs seeds, ordres de phases et capacités; mesurer rétention, QA, temps et mémoire du buffer, puis implémenter et évaluer une correction d'importance pour le replay priorisé.

## 2. Définir les objectifs et le protocole de mesure

- fixer les seuils de qualité, latence, débit et RAM après choix des tâches applicatives, des dimensions d'entrée/sortie, des données autorisées et de la cible de déploiement ;
- compléter les proxies synthétiques par des workloads réels mesurés sur cette cible.

## 3. Réduire les coûts du pipeline de données et de mémoire

- mesurer les compromis des stratégies de replay et des formats compacts sur des données représentatives.

## 4. Évaluer les précisions réduites

- comparer int8/int16 du runtime Dense sur des jeux de données représentatifs, avec erreurs après déquantification et métriques de tâche ;
- évaluer le stockage compact des poids du Transformer et son coût de conversion ;
- mesurer qualité, latence, mémoire et coût de conversion ensemble avant de choisir un format.

## 5. Renforcer l'évaluation et le continual learning

- élargir les benchmarks du runtime Dense au-delà des fonctions synthétiques avec séparation temporelle ou expérimentale adaptée ;
- ajouter des workloads applicatifs représentatifs au benchmark du runtime Dense ;
- généraliser la correction du biais d'échantillonnage Prioritized Replay si elle est retenue pour l'apprentissage continu ;
- étendre les tests mathématiques aux gradients d'autres activations et couches, à la stabilité et à la reproductibilité sur plusieurs configurations.

## 6. Valider sur une cible embarquée

- choisir une cible concrète avant de déclarer un gain embarqué ;
- mesurer latence, RAM, Flash et énergie sur l'appareil ;
- valider compilation, compatibilité numérique et résultats sur cette cible.

## 7. Clarifier les métadonnées du dépôt

- confirmer la licence du projet et ajouter le fichier légal correspondant si nécessaire; le README historique annonçait MIT, mais aucun fichier `LICENSE` n'est présent.

## Contrainte permanente

L'exécutable Gloomy ne doit utiliser aucune bibliothèque qui ne puisse pas être liée statiquement au programme. Les scripts de développement peuvent utiliser la bibliothèque standard de leur interpréteur, mais ne sont pas des dépendances du binaire.
