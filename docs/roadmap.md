# Gloomy — travaux à venir

Cette roadmap ne liste que les travaux non terminés. Les fonctionnalités déjà livrées sont récapitulées dans le [journal des changements](../CHANGELOG.md); les descriptions techniques détaillées restent dans les guides.

La priorité absolue du projet est de développer puis valider un modèle de langage capable d'apprendre à partir de textes, de suivre des consignes, de répondre à des questions et de générer du texte cohérent. Sa réussite doit être évaluée sur des tâches applicatives et des corpus autorisés, définis avec leurs utilisateurs.

## 1. Développer et valider un modèle de langage

- convenir avec les utilisateurs métier des tâches, formats d'entrée/sortie, corpus autorisés, références et seuils d'acceptation; faire valider le protocole QA provisoire avant de traiter ses seuils comme des critères applicatifs ;
- dépasser de façon reproductible la baseline majoritaire QA de 50 % sur les tâches documentaires, sans dégrader les bits par octet, puis confirmer le gain sur le test tenu à l'écart ;
- ajouter des workloads représentatifs du domaine applicatif réel au-delà des corpus documentaires actuels, avec séparations adaptées et mesures multi-seed de QA et bits par octet ;
- reproduire qualité, latence, débit, RAM et empreinte du modèle sur une cible de déploiement choisie avec les utilisateurs.

## Priorité haute — travaux transversaux pour le modèle de langage

- compléter le pilote d'arrondi des poids par l'évaluation des activations et du calcul à précision réduite; mesurer l'erreur numérique, la qualité des tâches, la latence, la mémoire et le coût de conversion sur des workloads représentatifs ;
- implémenter et mesurer un format réellement compact et/ou des opérations à précision réduite avant de conclure à un gain de RAM ou de performance; le pilote actuel conserve les poids en float64 après déquantification ;
- décider avec les utilisateurs si l'apprentissage continu fait partie du besoin; si oui, concevoir un replay de séquences et comparer ses stratégies, coûts et biais d'échantillonnage sur des données représentatives.

## 2. Définir les objectifs et le protocole de mesure

- fixer les seuils de qualité, latence, débit et RAM après choix des tâches applicatives, des dimensions d'entrée/sortie, des données autorisées et de la cible de déploiement ;
- compléter les proxies synthétiques par des workloads réels mesurés sur cette cible.

## 3. Réduire les coûts du pipeline de données et de mémoire

- mesurer les compromis des stratégies de replay et des formats compacts sur des données représentatives.

## 4. Évaluer les précisions réduites

- comparer int8/int16 du runtime Dense sur des jeux de données représentatifs, avec erreurs après déquantification et métriques de tâche ;
- étudier la quantification des poids et du calcul du Transformer au-delà du pilote de poids déquantifiés en float64 ;
- mesurer qualité, latence, mémoire et coût de conversion ensemble avant de choisir un format.

## 5. Renforcer l'évaluation et le continual learning

- élargir les benchmarks du runtime Dense au-delà des fonctions synthétiques avec séparation temporelle ou expérimentale adaptée ;
- ajouter des workloads applicatifs représentatifs au benchmark du runtime Dense ;
- étudier la correction du biais d'échantillonnage Prioritized Replay si cette stratégie est retenue pour l'apprentissage continu ;
- étendre les tests mathématiques aux gradients d'autres activations et couches, à la stabilité et à la reproductibilité sur plusieurs configurations.

## 6. Valider sur une cible embarquée

- choisir une cible concrète avant de déclarer un gain embarqué ;
- mesurer latence, RAM, Flash et énergie sur l'appareil ;
- valider compilation, compatibilité numérique et résultats sur cette cible.

## 7. Clarifier les métadonnées du dépôt

- confirmer la licence du projet et ajouter le fichier légal correspondant si nécessaire; le README historique annonçait MIT, mais aucun fichier `LICENSE` n'est présent.

## Contrainte permanente

L'exécutable Gloomy ne doit utiliser aucune bibliothèque qui ne puisse pas être liée statiquement au programme. Les scripts de développement peuvent utiliser la bibliothèque standard de leur interpréteur, mais ne sont pas des dépendances du binaire.
