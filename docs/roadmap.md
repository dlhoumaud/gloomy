# Gloomy — travaux à venir

Cette roadmap ne liste que les travaux non terminés. Les fonctionnalités déjà livrées sont récapitulées dans le [journal des changements](../CHANGELOG.md); les descriptions techniques détaillées restent dans les guides.

La priorité absolue du projet est de développer puis valider un modèle de langage capable d'apprendre à partir de textes, de suivre des consignes, de répondre à des questions et de générer du texte cohérent. La réussite devra être évaluée sur des tâches et des corpus de test autorisés et définis, pas seulement à partir de quelques exemples générés.

## 1. Développer un véritable modèle de langage

- dépasser de façon reproductible la baseline majoritaire QA de 50 % sur les tâches documentaires Ocara, sans dégrader les bits par octet, puis confirmer le gain sur le test tenu à l'écart ;
- convenir avec les utilisateurs métier des tâches, références et seuils d'acceptation avant de traiter les résultats documentaires comme une qualité applicative ;
- reproduire les mesures de référence sur le matériel de déploiement lorsqu'une cible applicative concrète sera définie; la campagne actuelle caractérise uniquement la baseline CPU locale.

Le parcours documentaire local multi-corpus, les tâches contrôlées prochain-token/QA, la sélection sur validation, l'évaluation tenue à l'écart, les mesures multi-seed et la reprise déterministe sont consignés dans le [journal des changements](../CHANGELOG.md). La campagne Ocara élargit le domaine, mais ne suffit pas à démontrer une qualité métier générale. Aucun gain QA au-dessus de la baseline, aucun seuil accepté par les utilisateurs et aucune cible de déploiement ne sont encore établis.

## Priorité haute — travaux transversaux pour le modèle de langage

- définir avec les utilisateurs les tâches métier du modèle, les formats d'entrée/sortie, les corpus autorisés, les références et les seuils d'acceptation; appliquer ce protocole aux benchmarks documentaires avant de conclure à une amélioration applicative ;
- élargir l'évaluation à des workloads représentatifs du domaine visé, avec des séparations train/validation/test adaptées et sans fuite entre documents; conserver les mesures multi-seed, la QA et les bits par octet ;
- mesurer sur ces workloads la qualité, la latence, le débit et la RAM du Transformer, puis reproduire ces mesures sur la cible de déploiement lorsqu'elle sera choisie ;
- évaluer les compromis de précision réduite pour le modèle de langage (poids, calcul et activations), avec erreurs numériques et métriques de tâche; les mémoires int8/int16 existantes concernent le runtime de replay et ne quantifient pas le Transformer ;
- renforcer les vérifications mathématiques propres au Transformer : gradients des opérations et couches, stabilité numérique et reproductibilité des entraînements ;
- si l'apprentissage continu du modèle est retenu, étudier l'utilisation d'un replay de séquences et mesurer les stratégies mémoire, leurs coûts et le biais d'échantillonnage Prioritized Replay sur des données représentatives.

Ces travaux reprennent les besoins généraux des sections 2 à 5 en les appliquant explicitement au modèle de langage. Les objectifs métier, le profil sur cible, les précisions réduites et l'apprentissage continu restent à établir; ils ne sont pas couverts par le benchmark CPU actuel.

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
