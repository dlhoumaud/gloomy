# Fonctions d'activation

Dans une couche, chaque neurone calcule d'abord `z = somme(x_i * w_i) + b`, puis applique `f(z)`. Le paramètre `-a` sélectionne `f`.

## `none`

`f(z) = z`. C'est une transformation linéaire. Plusieurs couches sans activation restent équivalentes à une seule transformation linéaire, donc cette option n'est pas adaptée à un réseau profond.

## `sigmoid`

`f(z) = 1 / (1 + exp(-z))` produit une valeur entre 0 et 1. Elle convient aux sorties binaires ou à des valeurs bornées, mais elle sature pour les grandes valeurs et peut ralentir l'apprentissage.

## `sigmoid_derivative`

`f(z) = sigmoid(z) * (1 - sigmoid(z))`. Cette valeur est la dérivée de sigmoid par rapport à son entrée. Elle est utile dans une rétropropagation, pas comme activation de toutes les couches. Dans ce projet elle est disponible par compatibilité CLI, mais le programme n'entraîne pas le réseau.

## `relu`

`f(z) = max(0, z)`. Elle est rapide et généralement un bon choix pour les couches cachées. Les neurones bloqués dans la zone négative ont toutefois une dérivée nulle.

## `leaky_relu`

`f(z) = z` si `z > 0`, sinon `0.01 * z`. Elle conserve un petit gradient côté négatif et limite le problème des neurones ReLU morts. C'est souvent le meilleur choix de départ avec `relu`.

## `tanh`

`f(z) = tanh(z)` produit une valeur entre -1 et 1 et est centrée autour de zéro. Elle convient aux séries normalisées et aux signaux comportant des valeurs positives et négatives, mais elle sature également.

## `tanh_derivative`

`f(z) = 1 - tanh(z)^2`. Comme `sigmoid_derivative`, c'est une dérivée destinée à la rétropropagation. Elle ne remplace pas une activation dans une architecture entraînée.

## Impact de `softmax`

Pour un vecteur de logits `z`, softmax calcule `exp(z_i) / somme(exp(z_j))`. Les valeurs sont positives et leur somme vaut 1 : elles peuvent représenter une distribution de classes.

Dans le code, softmax est appliqué uniquement à la dernière couche. Il ne faut pas l'utiliser après chaque couche cachée : cela détruirait l'information d'échelle avant la suite du réseau.

La sortie actuelle contient un seul neurone (`... -> 1`). Avec un seul élément, softmax vaut toujours `1`, quelle que soit l'activation précédente. `-A softmax` n'a donc aucun intérêt pour ce mode de prédiction scalaire. Pour une classification, il faudrait une sortie de `K` neurones, une cible de classe et une fonction de perte, puis choisir la classe de probabilité maximale.

Softmax ne corrige pas une activation mal choisie : il normalise seulement les sorties finales. Il est adapté à une sortie multi-classe, pas à une régression ou à une série numérique scalaire.
