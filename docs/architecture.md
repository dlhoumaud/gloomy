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

En mode `inference`, le modèle existant et son normaliseur sont chargés, puis les sorties sont écrites une par ligne. En mode entraînement ou apprentissage en ligne, un lecteur valide le fichier ligne par ligne sans conserver le dataset complet. Le runtime relit le fichier pour calculer les statistiques Welford si le modèle est neuf, puis pour alimenter le replay et l'entraînement. Il conserve uniquement le mini-batch courant, l'exemple en cours et la mémoire configurée. La calibration int16/int8 ajoute une passe qui calcule ses bornes sans stocker les exemples.

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

Le runtime construit des mini-batches de taille maximale `batch_size` et appelle `LearningEngine::trainBatch()`; il n'alloue pas de vecteur par epoch pour tout le dataset. L'entraînement depuis une mémoire reçoit un batch indexé de copies réutilisées par l'engine. En mode `training`, les exemples sont aussi ajoutés au replay configuré avant les epochs, comme auparavant; la taille conservée est limitée par la capacité de cette mémoire.

`NeuralNetwork::forwardRef()` expose par référence les sorties mises en cache et réutilisées par les couches; `backwardRef()` réutilise les buffers de gradients par couche. `LearningEngine` utilise aussi `gradientInto()` et conserve son buffer de gradient de perte entre les exemples. Pour l'inférence, `forwardInferenceRef()` ne remplit pas les caches requis pour la rétropropagation et le runtime normalise dans un buffer fourni. Les API historiques `forward()` et `backward()` restent disponibles et renvoient des copies indépendantes. Les buffers d'activations/gradients sont réutilisés, mais restent des vecteurs dynamiques séparés. Les poids et gradients de poids restent stockés en vecteurs imbriqués, car l'API mutable `weights()[i][j]` est utilisée par les optimiseurs, la sérialisation et les consommateurs existants; un stockage contigu demanderait une migration d'API coordonnée.

Les stratégies mémoire intégrées peuvent remplir un tableau `MemoryEntry` fourni par l'engine. Les mémoires réutilisent ce tableau et les scratch buffers de sélection; les versions quantifiées réutilisent également les buffers de déquantification et d'encodage. L'API historique `sampleIndexed()` reste disponible. Une implémentation externe de `LearningMemory` non migrée garde le comportement par défaut, qui construit encore le résultat par valeur.

`StreamingNormalizer` calcule moyenne et variance avec Welford. Pour un nouveau modèle, le runtime met à jour le normaliseur avec les entrées de tout le dataset chargé; il ne crée pas de split train/validation/test. À la reprise, les statistiques sauvegardées sont réutilisées. Il faut donc fournir un dataset d'entraînement approprié et réserver séparément les données d'évaluation.

Le calibrage des mémoires FIFO int16/int8 parcourt directement les entrées et cibles pour calculer leurs bornes min/max; il ne crée pas de vecteurs aplatis supplémentaires proportionnels à la taille du dataset.

La capacité mémoire est généralement exprimée en nombre d'échantillons, pas en budget d'octets strict. Les mémoires FIFO int16/int8 exposent une estimation des octets occupés par les échantillons, hors coût complet du processus.

## Priorité d'optimisation machine

Le benchmark actuel mesure un temps moyen d'inférence par exemple, le temps total d'entraînement, les débits et une estimation mémoire. Il ne mesure ni les percentiles de latence, ni le pic RSS, ni l'énergie, ni le coût CPU sur une cible embarquée. La priorité de conception retenue est la latence et le coût CPU; les optimisations doivent être comparées avec la qualité prédictive et la mémoire.

Les couches stockent encore poids et activations avec des vecteurs dynamiques imbriqués en `double`. Le traitement du dataset n'est pas en flux et les poids ne sont pas quantifiés. Les buffers contigus, la réutilisation des buffers et les mesures sur matériel cible restent à évaluer.
