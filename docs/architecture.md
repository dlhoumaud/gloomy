# Architecture et pipeline

## Chemins d'exécution

Le runtime configuré (`./bin/gloomy -f CONFIG`) est le chemin actuel pour l'apprentissage et l'inférence persistants :

```text
configuration + dataset entrée|cible
  -> validation et TrainingSample (double)
  -> normalisation des entrées
  -> training : réseau Dense -> perte -> optimiseur
  -> online_learning : observation -> mémoire -> scheduler/replay -> mise à jour
  -> sauvegarde GLOOMY_MODEL
```

En mode `inference`, le modèle existant et son normaliseur sont chargés, puis les sorties sont écrites une par ligne. Le runtime lit le dataset complet en mémoire et crée une copie normalisée pour l'apprentissage; le parsing et le traitement en flux ne sont donc pas des chemins à mémoire bornée.

Le CLI historique (`./bin/gloomy "..."`) est une démonstration séparée : il crée un réseau à poids aléatoires et, pour plusieurs prédictions, reconstruit le réseau avec de nouveaux poids à chaque étape. Il ne charge pas le modèle configuré et ne constitue pas un entraînement autorégressif.

## Couche dense

Une `DenseLayer` relie chaque entrée à chaque neurone de sortie. Pour `I` entrées et `O` sorties, elle possède `I * O` poids et `O` biais.

Le CLI historique construit :

```text
entrée (taille de la séquence)
  -> L couches cachées de N neurones
  -> sortie de 1 neurone
```

Avec `-l 0`, il n'y a pas de couche cachée. Le runtime configuré utilise `hidden_layers` et `neurons`, et détermine la taille de sortie à partir des cibles du dataset; elle n'est pas nécessairement égale à `1`.

Augmenter la largeur ou le nombre de couches accroît la capacité, le nombre de paramètres, le calcul et le risque de surajustement. Le runtime configuré prend en charge les pertes MSE/MAE/Huber et SGD/Momentum/Adam. Il ne fournit pas de split entraînement/validation automatique ni de régularisation.

## Apprentissage, mémoire et normalisation

Le `LearningEngine` gère les mini-batches, l'entraînement depuis une mémoire et l'ajout d'observations avant replay. Le runtime `online_learning` connecte cette boucle à un `EveryNScheduler`. Les stratégies FIFO, Reservoir, Prioritized, Novelty et Hybrid partagent l'interface `LearningMemory`.

`LearningEngine::train()` parcourt les mini-batches dans le dataset sans les recopier dans un vecteur temporaire. L'entraînement depuis une mémoire reçoit toutefois un batch indexé contenant des copies des exemples échantillonnés. Ces réductions de copies ne rendent pas le pipeline du runtime streaming : le dataset et une version normalisée sont tous deux conservés pendant l'apprentissage.

`NeuralNetwork::forwardRef()` expose par référence les sorties mises en cache et réutilisées par les couches; `LearningEngine` s'en sert pour éviter de recopier les sorties à chaque exemple entraîné. `backwardRef()` réutilise de la même manière les buffers de gradients par couche. Les API `forward()` et `backward()` restent disponibles et renvoient des copies indépendantes. Les caches d'entrée et les buffers de gradient restent dynamiques et ne sont pas contigus avec les poids.

`StreamingNormalizer` calcule moyenne et variance avec Welford. Pour un nouveau modèle, le runtime met à jour le normaliseur avec les entrées de tout le dataset chargé; il ne crée pas de split train/validation/test. À la reprise, les statistiques sauvegardées sont réutilisées. Il faut donc fournir un dataset d'entraînement approprié et réserver séparément les données d'évaluation.

La capacité mémoire est généralement exprimée en nombre d'échantillons, pas en budget d'octets strict. Les mémoires FIFO int16/int8 exposent une estimation des octets occupés par les échantillons, hors coût complet du processus.

## Priorité d'optimisation machine

Le benchmark actuel mesure un temps moyen d'inférence par exemple, le temps total d'entraînement, les débits et une estimation mémoire. Il ne mesure ni les percentiles de latence, ni le pic RSS, ni l'énergie, ni le coût CPU sur une cible embarquée. La priorité de conception retenue est la latence et le coût CPU; les optimisations doivent être comparées avec la qualité prédictive et la mémoire.

Les couches stockent encore poids et activations avec des vecteurs dynamiques imbriqués en `double`. Le traitement du dataset n'est pas en flux et les poids ne sont pas quantifiés. Les buffers contigus, la réutilisation des buffers et les mesures sur matériel cible restent à évaluer.
