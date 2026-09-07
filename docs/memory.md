# Mémoire d'apprentissage

## Contrat

`LearningMemory` sépare la politique de conservation du réseau neuronal et du moteur d'apprentissage. Elle manipule uniquement des `TrainingSample` composés d'un vecteur `input` et d'un vecteur `target`.

L'interface fournit :

- `add(sample)` : proposer un échantillon à la mémoire ;
- `remove(index)` : retirer un échantillon ;
- `sample(batch_size)` : produire un batch de replay ;
- `size()` et `capacity()` : observer l'occupation ;
- `clear()` : vider la mémoire.

Le réseau ne connaît pas cette abstraction et la mémoire ne connaît pas les détails des couches.

## FIFO / fenêtre glissante

`FIFOMemory` possède une capacité fixe en nombre d'échantillons. Lorsqu'elle est pleine, l'ajout d'un nouvel échantillon supprime le plus ancien.

Exemple avec une capacité de `3` :

```text
ajout : A B C  -> mémoire : A B C
ajout : D      -> mémoire : B C D
```

`sample(n)` renvoie au maximum `n` éléments dans l'ordre de la mémoire, en commençant par le plus ancien restant. Ce choix est déterministe et pratique pour les tests, mais ne constitue pas encore un replay aléatoire.

## Reservoir Sampling

`ReservoirMemory` conserve un échantillon de taille fixe d'un flux dont la longueur totale peut être inconnue. Les premiers éléments remplissent le réservoir. Pour chaque élément suivant, l'algorithme tire une position aléatoire dans le préfixe déjà observé ; si cette position appartient au réservoir, l'ancien élément est remplacé.

Après `T` observations et une capacité `K`, chaque observation a une probabilité théorique proche de `K / T` d'être conservée. La mémoire ne stocke jamais plus de `K` échantillons. `sample(n)` tire ensuite jusqu'à `n` éléments sans remise.

La seed du générateur est configurable afin de rendre les expériences reproductibles. Le coût d'ajout est constant en moyenne et le stockage reste borné, mais le générateur aléatoire ajoute un petit coût CPU par observation.

## Choix initial

- FIFO est préférable si les données récentes sont les plus pertinentes et si un comportement totalement déterministe est recherché.
- Reservoir est préférable si le flux est long ou de durée inconnue et si l'on veut conserver une représentation historique globale.

Ces deux stratégies peuvent maintenant être transmises au même `LearningEngine` sans modifier le réseau, la perte ou l'optimiseur.

## Limites actuelles

La capacité est limitée en nombre de `TrainingSample`, pas en octets. Les vecteurs `input` et `target` utilisent actuellement des `double` et leurs tailles peuvent varier ; le budget RAM réel n'est donc pas encore strictement contrôlé.

`QuantizedFIFOMemory` constitue une exception expérimentale : ses vecteurs `input` et `target` sont stockés en `int16` après calibration et la classe expose `memoryUsedBytes()`. Les autres stratégies utilisent encore leur représentation `double` native.

`QuantizedInt8FIFOMemory` fournit maintenant la variante int8, avec la même politique FIFO et le même contrat de replay. Elle réduit davantage les octets par valeur, au prix d'une erreur de quantification potentiellement supérieure.

La mémoire FIFO favorise les données récentes et peut provoquer du catastrophic forgetting. Reservoir limite ce biais temporel, mais ne tient pas encore compte de l'erreur, de la nouveauté ou des régimes de données.

Le `LearningEngine` peut maintenant appeler `trainFromMemory(memory, batch_size)` pour sélectionner un batch avec `memory.sample()` puis le transmettre à `trainBatch()`. La méthode `learn(memory, sample, batch_size)` ajoute une observation et déclenche immédiatement un replay.

Cette première intégration entraîne après chaque observation lorsqu'elle est utilisée avec `learn()`. Un `TrainingScheduler` peut maintenant différer l'entraînement pour réduire le coût CPU.

## Prioritized Replay

`PrioritizedMemory` utilise le champ `TrainingSample::priority`. Lorsqu'elle est pleine, elle remplace l'échantillon de plus faible priorité si le nouvel échantillon est plus important. Lors d'un replay, la probabilité de sélection est proportionnelle à `priority^alpha`, avec une petite valeur minimale pour conserver une exploration des priorités nulles.

`alpha = 0` rend la sélection uniforme. Une valeur plus élevée renforce la préférence pour les exemples prioritaires. La valeur par défaut est `0.6`.

La méthode `LearningEngine::learn()` calcule automatiquement une priorité égale à l'erreur absolue maximale entre la prédiction et la cible. Les appels directs à `LearningMemory::add()` peuvent définir eux-mêmes la priorité dans `TrainingSample`.

La priorité est recalculée après chaque replay indexé. La correction de biais d'échantillonnage n'est pas encore implémentée ; elle sera nécessaire pour une comparaison expérimentale rigoureuse avec un replay uniforme.

## Novelty Memory

`NoveltyMemory` compare chaque nouvelle entrée aux entrées déjà conservées avec une distance euclidienne au carré. Le paramètre `novelty_threshold` est donc comparé à :

```text
distance2(x, y) = somme((x_i - y_i)^2)
```

Une observation dont la distance minimale est inférieure au seuil est considérée comme redondante et ignorée. Les observations suffisamment différentes sont ajoutées jusqu'à la capacité. Lorsque la mémoire est pleine, le représentant ayant le voisin le plus proche est remplacé, afin de réduire la redondance locale.

Les entrées doivent avoir une dimension constante et ne doivent pas être vides. L'insertion examine les `K` éléments conservés ; la politique est donc simple et adaptée aux petites mémoires, mais elle coûte `O(K)` par observation. Le remplacement complet recherche en plus le représentant le plus redondant.

Novelty Memory réduit les séries d'observations presque identiques, mais elle ne tient pas encore compte de l'erreur de prédiction, de la récence ou de la diversité des cibles. Une combinaison avec Prioritized Replay ou une mémoire hybride sera étudiée ultérieurement.

## Mémoire hybride

`HybridMemory` divise une capacité globale entre quatre partitions configurables :

```cpp
HybridMemoryRatios ratios{
	0.25, // recent
	0.25, // error
	0.25, // novelty
	0.25  // historical
};
HybridMemory memory(256, ratios, novelty_threshold);
```

Les ratios sont normalisés automatiquement et les arrondis sont attribués à la partition récente. La capacité totale reste donc toujours égale à `256` au maximum.

La politique de routage actuelle est déterministe : une priorité supérieure à `0.5` dirige l'observation vers `error`; une observation suffisamment éloignée dirige l'observation vers `novelty`; les autres observations alternent entre `recent` et `historical`. Chaque observation n'est stockée que dans une partition, afin d'éviter les duplications.

La partition `recent` évince son élément le plus ancien lorsqu'elle est pleine. La partition `historical` remplace un élément choisi aléatoirement. Le replay final mélange les partitions sans remise.

Cette version est une base expérimentale : l'âge n'est pas encore une métadonnée de `TrainingSample`, la récence est donc approximée par le routage. Une future version pourra utiliser un score d'importance normalisé et une vraie politique hybride adaptative.

## Score d'importance

`ImportanceScorer` combine cinq composantes indépendantes, chacune bornée dans `[0, 1]` :

- `error` : erreur de prédiction ;
- `novelty` : différence avec les observations conservées ;
- `rarity` : rareté estimée d'un motif ;
- `recency` : caractère récent de l'observation ;
- `diversity` : contribution à la diversité globale.

Le score est une moyenne pondérée. Un poids égal à zéro désactive une composante, ce qui permet de comparer `error-only`, `novelty-only` ou des combinaisons sans changer la mémoire.

`TrainingSample` conserve maintenant `priority`, `error`, `novelty`, `rarity`, `recency`, `diversity`, `age` et `usage_count`. Pour l'instant, `LearningEngine::learn()` renseigne automatiquement l'erreur et calcule une priorité `error-only` normalisée. À chaque appel de replay, `advanceAges()` augmente l'âge de tous les échantillons et la mémoire incrémente `usage_count` pour les éléments sélectionnés.

Le replay indexé fournit maintenant l'indice mémoire avec chaque copie. `LearningEngine::trainFromMemory()` recalcule donc l'erreur après la mise à jour des poids et réécrit précisément l'échantillon concerné avec sa nouvelle priorité. Cela fonctionne même si deux observations ont des valeurs identiques, car leurs indices sont distincts.

## Training Scheduler

Le `TrainingScheduler` décide si une observation doit déclencher un replay :

- `EverySampleScheduler` entraîne après chaque observation ;
- `EveryNScheduler(N)` entraîne tous les `N` échantillons ;
- `OnHighErrorScheduler(seuil)` entraîne seulement lorsque l'erreur atteint le seuil.

Exemple :

```cpp
EveryNScheduler scheduler(10);
engine.learn(memory, sample, 8, scheduler);
```

L'observation est conservée même lorsque l'entraînement est différé. Le scheduler réduit donc les mises à jour, mais ne réduit pas à lui seul la taille de la mémoire. Le nombre d'updates et le coût CPU devront être mesurés dans les futurs benchmarks.
