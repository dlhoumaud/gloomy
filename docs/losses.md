# Fonctions de perte

Toutes les pertes implémentent l'abstraction `LossFunction` avec `compute(prediction, target)` et `gradient(prediction, target)`.

## MSE

```text
MSE = moyenne((prediction - target)^2)
dMSE/dprediction = 2 * (prediction - target) / N
```

MSE pénalise fortement les grandes erreurs. Elle est adaptée comme baseline de régression, mais une valeur aberrante peut dominer le batch.

## MAE

```text
MAE = moyenne(abs(prediction - target))
dMAE/dprediction = sign(prediction - target) / N
```

MAE est plus robuste aux valeurs aberrantes. Au point exact `prediction = target`, cette implémentation utilise un gradient nul, car la dérivée n'est pas unique.

## Huber

Avec un seuil `delta` :

```text
0.5 * erreur^2                         si abs(erreur) <= delta
 delta * (abs(erreur) - 0.5 * delta)  sinon
```

Huber est quadratique près de la cible et linéaire pour les grandes erreurs. Elle fournit un compromis entre la précision locale de MSE et la robustesse de MAE.

```cpp
HuberLoss loss(1.0);
LearningEngine engine(network, loss, optimizer);
```

Les trois pertes sont indépendantes du réseau, de l'optimiseur et de la mémoire. Il est donc possible de comparer leurs performances dans le benchmark sans modifier l'architecture.

## Cross-entropy (classification)

`CrossEntropyLoss` est une perte de classification, pas de régression comme les trois précédentes. Elle suppose que `prediction` est déjà une distribution de probabilités (typiquement en sortie d'un réseau avec `post_algorithm=softmax`) et que `target` en est une aussi (typiquement one-hot) :

```text
CrossEntropy = -somme(target_i * log(max(prediction_i, epsilon)))
dCrossEntropy/dprediction_i = -target_i / max(prediction_i, epsilon)
```

**Différence volontaire avec MSE/MAE/Huber** : cette perte ne divise **pas** par le nombre de classes. La division par la dimension de sortie, pour les trois pertes de régression, normalise une erreur *par dimension* — cela n'a pas le même sens pour une distribution de probabilités, où diviser par le nombre de classes changerait arbitrairement l'échelle de la perte selon le nombre de classes choisi, sans justification.

`gradient()` retourne le gradient par rapport à la sortie **post-softmax** (`-target_i/prediction_i`) : il est destiné à être composé avec le jacobien softmax déjà implémenté dans `DenseLayer::backward` (voir [Couches et neurones](architecture.md) et `testSoftmaxGradientCheck`/`testCrossEntropyLoss` dans `tests/loss_tests.cpp`, qui vérifient cette composition par différence finie sur un réseau complet).

```cpp
NeuralNetwork network;
network.post_algorithm = "softmax";
// ... couches ...
CrossEntropyLoss loss;
LearningEngine engine(network, loss, optimizer);
```

Une prédiction négative (par exemple des logits bruts passés par erreur, sans `post_algorithm=softmax`) est rejetée explicitement plutôt que de produire silencieusement `log(négatif) = NaN`.

**Ce qui reste hors de portée** : cette perte existe et son gradient est vérifié, mais aucun runtime CLI (`online_learning`/`training`) ne l'exploite — ces runtimes sont structurellement des runtimes de régression scalaire (`window_size` valeurs en entrée, un seul scalaire en sortie), pas de classification multi-classe. Une vraie pipeline de classification multi-classe dans le CLI (sortie à `K` neurones, encodage/décodage des classes, métrique d'exactitude) reste un chantier distinct, plus large — voir [feuille de route](roadmap.md).
