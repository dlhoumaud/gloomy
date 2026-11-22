# Modèle de langage causal Gloomy

L'exemple applicatif est un assistant français de documentation Gloomy : il apprend à prédire les tokens des documents du dépôt et à générer des réponses factuelles à partir de questions définies. Il ne démontre ni une compréhension générale ni la qualité sur un domaine externe.

## Corpus applicatif du dépôt

Le corpus d'entraînement est préparé depuis les fichiers Markdown de Gloomy par [prepare_gloomy_docs_corpus.py](../../../scripts/prepare_gloomy_docs_corpus.py). Les fichiers source sont regroupés entièrement dans un seul split — aucun fichier Markdown ne traverse train, validation et test. Les QA train, validation et test ont des contextes propres à leur split; les prompts sont distincts et les tests équilibrent les réponses `oui`/`non` pour rendre visible la baseline majoritaire à 50 %. Les splits, réponses de référence, empreintes source/split et la déclaration d'autorisation locale sont générés sous `data/lm/gloomy-docs/manifest.json`.

Le dépôt ne contient pas de fichier `LICENSE`. L'utilisation locale des documents du projet est faite à la demande du propriétaire du projet pour cette expérience; cela ne constitue pas une déclaration de licence ou d'autorisation de redistribution. Aucune donnée tierce n'est téléchargée.

Les tâches sont :

- prédiction du prochain token, mesurée par cross-entropy, perplexité et bits par octet UTF-8 pour comparer les tokenizers ;
- QA factuelle fermée conditionnée par un extrait source fourni dans le prompt, sous forme de verdict `oui`/`non`; le benchmark rapporte à la fois l'exact match de génération libre et une accuracy fermée qui compare la log-vraisemblance des labels candidats, ainsi que F1 de tokens.

La cible de référence disponible est le Ryzen 3 5300U sous Linux x86_64. C'est une baseline locale, pas une cible embarquée déclarée.

Le prototype ne récupère pas lui-même les passages pertinents : le contexte est déjà inclus dans le prompt de l'exemple QA. Les séquences d'entraînement QA sont répétées 8 fois; le tokenizer BPE est plafonné au même facteur afin que des répétitions de gradient n'altèrent pas le vocabulaire. Les hyperparamètres peuvent être sélectionnés sur la perplexité et la QA de validation, jamais sur le test. La campagne mesure donc l'évaluation d'affirmations et la génération conditionnée, pas la recherche documentaire de bout en bout.

Pour préparer, entraîner puis mesurer le modèle :

```sh
make language-model-docs-example
make benchmark-language-model-docs
```

Le benchmark publie les résultats bruts trois seeds et leur manifeste dans [`gloomy-docs-3-seeds.csv`](gloomy-docs-3-seeds.csv) et [`gloomy-docs-3-seeds.manifest.json`](gloomy-docs-3-seeds.manifest.json). Les références de validation servent à contrôler la progression; le split test reste séparé. Le test coûteux de reprise bit-à-bit après une epoch sur tout le corpus documentaire est lancé séparément par `make test-language-model-docs-resume`; `make test` conserve des tests d'intégration rapides.

Résultat de référence sur Ryzen 3 5300U (Linux x86_64, g++ 15.2.0) : vocabulaire 512, contexte 32, dimension 8, 1 couche, 2 têtes, batch 2, 3 epochs, taux d'apprentissage 0,003, seeds 42/43/44. Les 9/2/2 documents des splits train/validation/test font respectivement 55 188/4 311/4 636 octets; les SHA-256 sont dans le manifeste. La perplexité test est de 72,89 à 77,01 (moyenne 74,72), soit 3,235–3,277 bits par octet UTF-8 (moyenne 3,254). La QA validation générative obtient 3/6, 2/6 et 2/6 (38,9 % en moyenne). Sur test, la génération libre obtient 3/6, 0/6 et 3/6 (33,3 % en moyenne); le choix fermé obtient 3/6 pour chaque seed (50 %), égal à la baseline majoritaire. Le modèle ne démontre donc pas une compréhension QA utile. Un essai à 32 répétitions QA a été rejeté sur validation : l'exact match génératif moyen tombe à 27,8 % sans gain QA test moyen; son petit gain bits/octet ne compense pas la baisse de validation QA. L'entraînement retenu prend 154,0–154,4 s par seed, le pic RSS mesuré est 75 632–75 636 KiB et les checkpoints font 251 575–251 600 octets. La génération libre prend 61,4–785,5 ms par exemple (27,1–40,7 tokens/s); évaluer les deux labels fermés prend 0,691–0,738 s pour six exemples. Ces mesures forment une baseline étroite et reproductible, pas un seuil de qualité atteint.

Le corpus fictif décrit ci-dessous est conservé comme fixture minuscule pour les tests d'intégration rapides, pas comme corpus de validation applicative.

## Données

- `train.txt` : textes de prédiction du prochain token; le tokenizer est aussi ajusté sur `qa-train.txt`, jamais sur validation/test.
- `validation.txt` : suivi de la cross-entropy et de la perplexité pendant l'entraînement.
- `test.txt` : textes non vus pour l'évaluation finale de perplexité.
- `qa-train.txt` : paires d'instruction/réponse pour apprentissage supervisé; le masque exclut le prompt de la perte et n'apprend que la réponse.
- `qa-validation.txt` : affirmations et réponses de référence basées exclusivement sur les documents de validation.
- `qa-test.txt` : questions/réponses non vues, équilibrées `oui`/`non`, basées exclusivement sur les documents de test; l'exact match glouton et la sélection fermée par log-vraisemblance sont mesurés séparément.

Ces phrases et exemples sont originaux, écrits pour le test logiciel du dépôt; aucune source tierce n'est utilisée. C'est un mini-corpus fictif autorisé pour valider la chaîne, pas un corpus applicatif représentatif. Les lignes sont des documents indépendants. Train, validation et test sont fixés et disjoints; les réponses QA de test sont justifiées uniquement par des documents du split test. Le manifeste d'exécution consigne la provenance déclarée, l'autorisation, les SHA-256, tailles, paramètres, seeds et la cible locale.

Les tâches définies ici sont la prédiction du prochain token et l'extraction de faits courts en QA fermée. L'exact match strict et le F1 par tokens ne couvrent ni la paraphrase, ni le raisonnement, ni la qualité ouverte de génération. Pour une validation applicative, il faut un domaine cible et un corpus autorisé correspondant.

Pour préparer des splits depuis un corpus texte brut (paragraphes séparés par une ligne vide, ou un document par ligne), l'outil de dépôt normalise en NFC, déduplique et écrit les empreintes dans un manifeste :

```sh
python3 scripts/prepare_lm_corpus.py \
  --input /chemin/vers/corpus-autorise.txt \
  --train data/lm/train.txt \
  --validation data/lm/validation.txt \
  --test data/lm/test.txt \
  --manifest data/lm/manifest.json \
  --source "Nom ou URL du corpus" --license "Licence ou autorisation" \
  --seed 42 --validation-ratio 0.1 --test-ratio 0.1
```

Le script ne télécharge aucun corpus et n'en établit pas la licence; ces informations doivent être vérifiées avant l'utilisation. Le fichier test produit doit rester réservé à l'évaluation finale.

```bash
make language-model-docs-example
python3 scripts/prepare_gloomy_docs_corpus.py
mkdir -p artifacts
./bin/gloomy lm train \
        --train data/lm/gloomy-docs/train.txt \
        --validation data/lm/gloomy-docs/validation.txt \
        --qa-train data/lm/gloomy-docs/qa-train.txt \
        --model artifacts/gloomy-docs.gloomy \
        --vocab 512 --context 32 --dimension 8 --heads 2 --layers 1 \
        --batch-size 2 --epochs 3 --learning-rate 0.003 --seed 42
./bin/gloomy lm evaluate \
        --model artifacts/gloomy-docs.gloomy \
        --text data/lm/gloomy-docs/test.txt \
        --qa data/lm/gloomy-docs/qa-test.txt \
        --qa-label oui \
        --qa-label non
./bin/gloomy lm generate \
        --model artifacts/gloomy-docs.gloomy \
        --prompt "Question: La formule MAE est-elle une moyenne d'erreurs absolues ? Contexte: MAE = moyenne(abs(prediction - cible)) Réponse:" \
        --tokens 4 \
        --temperature 0 \
        --seed 42
```

## Construire et entraîner

Depuis la racine du dépôt :

```sh
make language-model-example
```

Cette cible construit l'exécutable si nécessaire, crée `artifacts/` et entraîne `artifacts/language-mini.gloomy`. Les checkpoints `.gloomy` sont ignorés par Git; après un clone, il faut exécuter cette étape avant les commandes d'évaluation et de génération ci-dessous.

Le modèle contient embeddings de tokens et positions, attention multi-tête causale, connexions résiduelles, LayerNorm, MLP GELU et projection vocabulaire. L'objectif est la cross-entropy du prochain token. Les fenêtres de contexte se chevauchent; le masque n'applique la perte qu'aux nouveaux tokens, une fois chacun. Pour les paires QA, il ignore aussi la perte sur le prompt et l'applique seulement à la réponse. AdamW, clipping du gradient et état des moments sont sauvegardés dans le checkpoint. Les options de génération acceptent température, top-k, top-p et seed; température zéro sélectionne gloutonnement.

## Évaluer et générer

Après l'entraînement, les deux commandes suivantes utilisent le checkpoint créé. Pour le générer dans un clone neuf, exécuter d'abord `make language-model-example`.

```sh
./bin/gloomy lm evaluate --model artifacts/language-mini.gloomy \
  --text docs/examples/language-model/test.txt \
  --qa docs/examples/language-model/qa-test.txt

./bin/gloomy lm generate --model artifacts/language-mini.gloomy \
  --prompt "Question: Quel animal miaule ? Réponse:" \
  --tokens 12 --temperature 0 --seed 42
```

L'évaluation rapporte la cross-entropy et la perplexité tokenisées, l'exact match et le F1 par tokens. L'exact match compare la réponse générée à la référence après normalisation NFC, de la casse ASCII et des espaces; il ne mesure pas les réponses sémantiquement équivalentes. Le F1 par tokens tolère des recouvrements partiels, mais ne mesure pas la factualité. Le runner vérifie que les documents des trois splits sont disjoints et que les réponses QA de test ne figurent pas dans train/validation. La perplexité est une métrique de prédiction de tokens, pas une preuve de compréhension.

Pour conserver les métriques brutes sur plusieurs initialisations, le dépôt fournit un runner reproductible :

```sh
python3 scripts/benchmark_language_model.py \
  --train docs/examples/language-model/train.txt \
  --validation docs/examples/language-model/validation.txt \
  --qa-train docs/examples/language-model/qa-train.txt \
  --test docs/examples/language-model/test.txt \
  --qa docs/examples/language-model/qa-test.txt \
  --corpus-source "Gloomy original synthetic language-model fixture" \
  --corpus-authorization "original text written for repository tests; no third-party corpus" \
  --output /tmp/language-model-results.csv \
  --seeds 42,43,44 --vocab 320 --context 32 --dimension 8 \
  --heads 2 --layers 1 --batch-size 2 --epochs 1 --learning-rate 0.003
```

Le résultat brut de cette commande est conservé dans [`results-3-seeds.csv`](results-3-seeds.csv); le JSON homonyme `.manifest.json` conserve les empreintes des données et le profil logiciel/matériel. Le CSV rapporte aussi la QA validation et test (génération libre et choix fermé par score des labels), leur baseline majoritaire d'entraînement, le temps/RSS d'entraînement et d'évaluation, les tokens/s d'évaluation, la latence QA moyenne et tokens/s de génération, les dimensions/taille/empreintes des splits et la taille du checkpoint. La cross-entropy par token n'est pas directement comparable si le tokenizer change; `test_bits_per_byte` divise la somme des pertes (y compris les tokens EOS) par les octets UTF-8 du texte test normalisé, hors séparateurs de lignes. La cible est la machine CPU locale décrite par le manifeste, pas une cible embarquée. Le runner ne calcule aucun score composite.

Un contrôle séparé de 50 epochs mémorise exactement 2 réponses sur les 3 prompts QA vus et atteint une cross-entropy train de 1,65. Sur les questions tenues à l'écart, il reste à 0/2 et la perplexité test est de 144,23. Les mesures brutes sont dans [`qa-overfit-diagnostic.csv`](qa-overfit-diagnostic.csv). Ce contrôle montre l'écart entre mémoriser quelques exemples et généraliser; il ne constitue pas un benchmark de qualité.

## Reprendre l'entraînement

`--resume` recharge le tokenizer, les poids, le compteur d'updates, les moments AdamW et l'état du mélange du chemin `--model`, puis continue avec les données train fournies. Les options d'architecture et le seed du checkpoint sont conservés :

```sh
./bin/gloomy lm train \
  --train docs/examples/language-model/train.txt \
  --validation docs/examples/language-model/validation.txt \
  --qa-train docs/examples/language-model/qa-train.txt \
  --model artifacts/language-mini.gloomy --resume \
  --batch-size 2 --epochs 1
```

## Limites mesurées et connues

Le modèle est volontairement minuscule (dimension 8, un bloc, contexte 32, vocabulaire 320, 6 584 paramètres), entraîné sur quelques phrases et trois exemples QA masqués. Le tokenizer byte-level part des 256 octets, puis apprend des fusions BPE; un caractère valide absent du corpus reste donc encodable. Sur les trois seeds, le checkpoint complet pèse 168 789–168 792 octets, une époque prend 0,457–0,477 s et le pic RSS échantillonné atteint 45 872–66 680 KiB sur le Ryzen 3 5300U. La perplexité test varie de 232,57 à 244,23 et l'exact match reste 0/2; la génération QA prend 664–672 ms par exemple. La sortie générée reste incohérente; ces mesures ne valident aucune compréhension. Les résultats et leurs conditions sont conservés dans le CSV et son manifeste. Le tokenizer applique NFC (données Unicode 16.0); le CLI charge les corpus en mémoire et limite la taille des graphes d'autodifférentiation pour éviter des allocations excessives.
