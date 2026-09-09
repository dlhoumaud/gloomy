# Optimiseurs

## SGD

`SGDOptimizer` applique :

```text
parameter = parameter - learning_rate * gradient
```

Il ne conserve pas d'état supplémentaire par paramètre. C'est le choix de référence pour l'embarqué lorsque la RAM est prioritaire.

## SGD avec momentum

`MomentumOptimizer` conserve une vitesse par poids et par biais :

```text
velocity = momentum * velocity + gradient
parameter = parameter - learning_rate * velocity
```

Le paramètre `momentum` doit être dans `[0, 1)`, avec `0.9` comme valeur de départ classique. L'état du momentum consomme un `double` supplémentaire par paramètre, donc environ la même mémoire que les paramètres float64 eux-mêmes.

```cpp
MomentumOptimizer optimizer(0.01, 0.9);
LearningEngine engine(network, loss, optimizer);
```

Momentum peut accélérer la convergence et lisser les directions bruyantes, mais il n'est pas automatiquement supérieur à SGD sous une contrainte RAM stricte. La méthode `stateBytes()` permet de mesurer son coût.

## Comparaison expérimentale

Comparer au minimum :

- perte finale ;
- nombre d'updates ;
- temps d'entraînement ;
- stabilité en online learning ;
- mémoire des paramètres ;
- mémoire d'état de l'optimiseur.

## Adam

`AdamOptimizer` combine un premier moment, un second moment et une correction de biais :

```text
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2
parameter -= learning_rate * corrected_m / (sqrt(corrected_v) + epsilon)
```

Il conserve deux `double` par paramètre, soit environ deux fois le coût d'état de Momentum. Adam peut être plus rapide sur certains problèmes, mais sa mémoire et ses opérations supplémentaires le rendent moins évident pour un microcontrôleur. `stateBytes()` permet de comparer ce coût directement.

Adam doit être comparé à SGD et Momentum sur les mêmes données, seeds, nombre d'updates et budgets mémoire. Il ne doit pas être considéré comme supérieur par défaut.

## Adam à état compressé

`CompressedAdamOptimizer` applique exactement la même formule qu'`AdamOptimizer`, mais conserve les moments (premier, second) **quantifiés en int16** (`Int16Quantizer`) entre deux appels à `update()` plutôt qu'en `double` natif — le même principe que `QuantizedFIFOMemory` appliqué à l'état d'un optimiseur plutôt qu'à une mémoire de replay :

```cpp
CompressedAdamOptimizer optimizer(0.01);
LearningEngine engine(network, loss, optimizer);
```

À chaque `update()` : les moments sont déquantifiés, mis à jour avec la formule Adam standard **en double** (le calcul du pas courant n'est pas dégradé), puis recalibrés et requantifiés avant d'être restockés. Contrairement à `QuantizedFIFOMemory` (calibrée une fois pour toutes à la construction sur un jeu d'échantillons fixe), la recalibration doit se refaire à **chaque** pas ici, car la plage des moments évolue tout au long de l'entraînement.

### Impact mesuré

Sur un réseau 1-8-1 (seed `4242`, 80 epochs, SGD lr=0.01, tâche y=2x+1), comparé à `AdamOptimizer` :

- perte finale : `~0.000001` pour les deux (convergence quasi identique — la quantification des moments à chaque pas n'a pas empêché l'entraînement de converger sur cette tâche) ;
- `stateBytes()` : `228` octets contre `400` pour Adam natif (~1.75×), **loin du 4× théorique** (`double` 8 octets vs `int16` 2 octets) — même constat que pour `NetworkQuantization` (voir [Quantification](quantization.md)) : le coût fixe de calibration par vecteur (`scale`/`zero_point`, 4 vecteurs par couche) domine sur un réseau aussi petit. Le gain se rapprocherait de 4× pour des couches beaucoup plus larges.

`optimizer=compressed_adam` le sélectionne dans le CLI (`online_learning`/`training`) — voir [Configurations et limites](configurations.md). Limite connue : `OptimizerSerialization`/`ModelSerialization` ne le reconnaissent pas encore (seuls SGD/Momentum/Adam le sont) ; utiliser `model_path` avec cet optimiseur échoue avec une erreur explicite (`"Unsupported optimizer type for serialization"`) plutôt que de corrompre silencieusement l'état sauvegardé.

## Persistance de l'état

`OptimizerSerialization` sauvegarde et restaure n'importe quel `Optimizer` concret (SGD, Momentum ou Adam) dans un fichier binaire versionné, protégé par un checksum FNV-1a :

```cpp
OptimizerSerialization::save("optimizer.bin", optimizer);
std::unique_ptr<Optimizer> restored =
    OptimizerSerialization::load("optimizer.bin", network.layers());
```

Le fichier contient le type d'optimiseur, ses hyperparamètres (`learning_rate`, `momentum`, `beta1`/`beta2`/`epsilon`), le compte d'updates pour Adam, ainsi que les vitesses de Momentum ou les premiers et seconds moments d'Adam. `load()` reconstruit l'optimiseur concret et vérifie que la forme de chaque couche sauvegardée correspond exactement au réseau fourni (`network.layers()`) avant de restaurer le buffer : un fichier taillé pour une autre architecture, tronqué, corrompu ou de version incompatible est rejeté avec une exception plutôt que de restaurer un état invalide.

Cette persistance reste utile comme export séparé du réseau et de la mémoire d'apprentissage, mais elle est aussi intégrée dans le format unifié `GLOOMY_MODEL` via `ModelSerialization` (voir [feuille de route](roadmap.md)). Les runtimes `online_learning` et `training` peuvent désormais sauvegarder l'état complet d'un apprentissage dans un seul fichier lorsqu'un `model_path` est fourni, et **le relire au démarrage** pour reprendre l'entraînement là où il s'était arrêté au lieu de repartir d'un réseau neuf (voir [Configurations et limites](configurations.md)).

## Validation des gradients

`SGDOptimizer`, `MomentumOptimizer` et `AdamOptimizer` rejettent désormais, dans `update()`, tout gradient (poids ou biais) non fini (`NaN`/infini), ainsi qu'un `gradient_scale` non fini ou non positif. `DenseLayer::backward` valide déjà son gradient *entrant*, mais pas le résultat de ses propres multiplications/accumulations internes (`input * gradient_local`) : un dépassement de capacité du `double` (deux valeurs finies mais extrêmes dont le produit dépasse `~1.8e308`) reste donc possible en amont de l'optimiseur, sans jamais impliquer de `NaN`/infini injecté directement. C'est ce cas — gradients finis en entrée, mais devenus non finis après calcul — que cette validation attrape, avant que l'optimiseur ne les applique aux poids.
