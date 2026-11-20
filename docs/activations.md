# Fonctions d'activation

Dans une couche, chaque neurone calcule d'abord `z = somme(x_i * w_i) + b`, puis applique `f(z)`. Le CLI historique utilise `-a`; le runtime configuré utilise `activation`.

## `none`

`f(z) = z`. C'est une transformation linéaire. Plusieurs couches sans activation restent équivalentes à une seule transformation linéaire, donc cette option n'est pas adaptée à un réseau profond.

## `sigmoid`

`f(z) = 1 / (1 + exp(-z))` produit une valeur entre 0 et 1. Elle convient aux sorties binaires ou à des valeurs bornées, mais elle sature pour les grandes valeurs et peut ralentir l'apprentissage.

## `sigmoid_derivative`

`f(z) = sigmoid(z) * (1 - sigmoid(z))`. Cette valeur est la dérivée de sigmoid par rapport à son entrée. Elle est utile dans une rétropropagation, pas comme activation de toutes les couches. Elle est acceptée uniquement par le CLI historique pour compatibilité; le runtime configuré sélectionne `sigmoid` et calcule les dérivées en interne.

## `relu`

`f(z) = max(0, z)`. Elle est rapide et généralement un bon choix pour les couches cachées. Les neurones bloqués dans la zone négative ont toutefois une dérivée nulle.

## `leaky_relu`

`f(z) = z` si `z > 0`, sinon `0.01 * z`. Elle conserve un petit gradient côté négatif et limite le problème des neurones ReLU morts. C'est souvent le meilleur choix de départ avec `relu`.

## `tanh`

`f(z) = tanh(z)` produit une valeur entre -1 et 1 et est centrée autour de zéro. Elle convient aux séries normalisées et aux signaux comportant des valeurs positives et négatives, mais elle sature également.

## `tanh_derivative`

`f(z) = 1 - tanh(z)^2`. Comme `sigmoid_derivative`, c'est une dérivée destinée à la rétropropagation. Elle est acceptée uniquement par le CLI historique et ne remplace pas `tanh` comme activation d'un réseau entraîné.

## Impact de `softmax`

Pour un vecteur de logits `z`, softmax calcule `exp(z_i) / somme(exp(z_j))`. Les valeurs sont positives et leur somme vaut 1 : elles peuvent représenter une distribution de classes.

Dans le code, softmax est appliqué uniquement à la dernière couche. Il ne faut pas l'utiliser après chaque couche cachée : cela détruirait l'information d'échelle avant la suite du réseau.

Le CLI historique construit une sortie à un neurone (`... -> 1`). Avec un seul élément, softmax vaut toujours `1`, quelle que soit l'activation précédente; `-A softmax` est donc inutile sur ce chemin. Le runtime configuré permet une taille de sortie déterminée par les cibles, mais le dépôt ne fournit pas encore de perte classification/cross-entropy ni de protocole de classification complet.

Softmax ne corrige pas une activation mal choisie : il normalise seulement les sorties finales. Il est adapté à une sortie multi-classe, pas à une régression ou à une série numérique scalaire.
