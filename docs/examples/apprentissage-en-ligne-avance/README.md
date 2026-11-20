# Apprentissage en ligne avancé avec dérive et replay Hybrid

Cet exemple combine les capacités d'apprentissage actuellement disponibles dans le runtime :

- trois valeurs d'entrée et deux valeurs cibles ;
- apprentissage en ligne dans l'ordre des observations ;
- perte Huber et optimiseur Adam ;
- mémoire de replay Hybrid qui répartit sa capacité entre échantillons récents, erreurs, nouveauté et historique ;
- checkpoint `GLOOMY_MODEL` qui conserve le réseau, l'état Adam, la mémoire Hybrid et son état aléatoire.

Le jeu de données passe du régime A au régime B : les mêmes entrées ont des cibles différentes après le changement. Les deux régimes sont des régressions linéaires multi-sorties synthétiques. Le replay donne au modèle des exemples antérieurs pendant l'apprentissage du second régime; il ne garantit pas à lui seul l'absence d'oubli.

Toutes les commandes se lancent depuis la racine du dépôt :

```bash
make
mkdir -p artifacts/examples
./bin/gloomy -f docs/examples/apprentissage-en-ligne-avance/phase-a.config
./bin/gloomy -f docs/examples/apprentissage-en-ligne-avance/phase-b.config
./bin/gloomy -f docs/examples/apprentissage-en-ligne-avance/inference.config
```

La première commande d'entraînement initialise le réseau et la mémoire avec le régime A. La seconde charge ce même checkpoint puis ajoute les observations du régime B au replay. L'inférence teste ensuite une entrée du régime B. Pour comparer la rétention, évaluez aussi des entrées du régime A avant et après la phase B; le runtime ne calcule pas ces métriques automatiquement.

Le dataset reste le format texte pris en charge `entrées | cibles`; CSV et JSON ne sont pas encore gérés. Cet exemple est volontairement synthétique et sert à explorer le pipeline, pas à démontrer une rétention robuste sur un problème réel.
