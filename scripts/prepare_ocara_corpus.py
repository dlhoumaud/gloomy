#!/usr/bin/env python3
"""Prepare document-separated Ocara training and balanced factual QA splits."""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs/examples/language-model-ocara/docs"
ALL_SOURCE_FILES = [
    "README.md",
    "EBNF.md",
    "adding-builtins.md",
    "adding-types.md",
    "agents/copilot-ocara.agent.md",
    "agents/instruction-copilot.agent.md",
    "compilation-guide.md",
    "copilot-instructions.md",
    "diagnostics.md",
    "workflow-compilation.md",
    "builtins/Array.md",
    "builtins/Convert.md",
    "builtins/Date.md",
    "builtins/DateTime.md",
    "builtins/Directory.md",
    "builtins/DotEnv.md",
    "builtins/File.md",
    "builtins/HTML.md",
    "builtins/HTMLComponent.md",
    "builtins/HTTPRequest.md",
    "builtins/HTTPServer.md",
    "builtins/IO.md",
    "builtins/JSON.md",
    "builtins/Map.md",
    "builtins/Math.md",
    "builtins/Mutex.md",
    "builtins/MySQL.md",
    "builtins/Regex.md",
    "builtins/SDL.md",
    "builtins/SQLite.md",
    "builtins/String.md",
    "builtins/System.md",
    "builtins/Tauri.md",
    "builtins/Thread.md",
    "builtins/Time.md",
    "builtins/UnitTest.md",
    "builtins/YAML.md",
]
VALIDATION_FILES = [
    "compilation-guide.md",
    "diagnostics.md",
    "builtins/Array.md",
    "builtins/JSON.md",
    "builtins/Thread.md",
]
TEST_FILES = [
    "builtins/HTTPRequest.md",
    "builtins/UnitTest.md",
    "builtins/DateTime.md",
    "builtins/Regex.md",
    "builtins/SQLite.md",
    "builtins/Tauri.md",
]
TRAIN_FILES = sorted(set(ALL_SOURCE_FILES) - set(VALIDATION_FILES) - set(TEST_FILES))
QA_LABELS = {"oui", "non"}
QA_TRAIN_REPEATS = 32
QA_TOKENIZER_REPEAT_CAP = 8


def pair(question: str, context: str, answer: str) -> tuple[str, str]:
    if " | " in question or " | " in context:
        raise ValueError("QA prompts cannot contain the dataset delimiter ' | '")
    return f"Question: {question} Contexte: {context} Réponse:", answer


QASpec = tuple[str, str, str, str]


def qa(question: str, source: str, anchor: str, answer: str) -> QASpec:
    return question, source, anchor, answer


QA_TRAIN_SPECS = [
    qa("File::write écrase-t-il le fichier s'il existe ?", "builtins/File.md", "Écrit du contenu texte dans un fichier", "oui"),
    qa("File::write ajoute-t-il le contenu à la fin du fichier ?", "builtins/File.md", "Écrit du contenu texte dans un fichier", "non"),
    qa("File::append crée-t-il le fichier s'il n'existe pas ?", "builtins/File.md", "Ajoute du contenu à la fin d'un fichier", "oui"),
    qa("File::append échoue-t-il si le fichier n'existe pas ?", "builtins/File.md", "Ajoute du contenu à la fin d'un fichier", "non"),
    qa("Directory::createRecursive crée-t-il tous les parents ?", "builtins/Directory.md", "mkdir -p", "oui"),
    qa("Directory::createRecursive exige-t-il que tous les parents existent déjà ?", "builtins/Directory.md", "mkdir -p", "non"),
    qa("Directory::remove supprime-t-il un répertoire non vide ?", "builtins/Directory.md", "Supprime un répertoire vide", "non"),
    qa("Directory::remove est-il destiné aux répertoires vides ?", "builtins/Directory.md", "Supprime un répertoire vide", "oui"),
    qa("MySQL::queryOne retourne-t-il la première ligne ou null si aucun résultat ?", "builtins/MySQL.md", "la première ligne, ou `null`", "oui"),
    qa("MySQL::queryOne retourne-t-il toujours un tableau de lignes ?", "builtins/MySQL.md", "la première ligne, ou `null`", "non"),
    qa("String::replace remplace-t-elle toutes les occurrences ?", "builtins/String.md", "Remplace **toutes**", "oui"),
    qa("String::replace remplace-t-elle uniquement la première occurrence ?", "builtins/String.md", "Remplace **toutes**", "non"),
    qa("Convert::strToBool ne lève-t-elle jamais d'exception pour une valeur inconnue ?", "builtins/Convert.md", "valeurs inconnues", "oui"),
    qa("Convert::strToBool lève-t-elle une exception pour une valeur inconnue ?", "builtins/Convert.md", "valeurs inconnues", "non"),
    qa("HTTPServer workers définit-il la capacité de traitement parallèle ?", "builtins/HTTPServer.md", "nombre maximum de connexions simultanées", "oui"),
    qa("Le nombre de workers fixe-t-il le nombre maximum de connexions simultanées ?", "builtins/HTTPServer.md", "nombre maximum de connexions simultanées", "non"),
    qa("Convert::strToInt retourne-t-elle 0 si la conversion échoue ?", "builtins/Convert.md", "Retourne `0` si la conversion échoue", "oui"),
    qa("Convert::strToInt lève-t-elle une exception si la conversion échoue ?", "builtins/Convert.md", "Retourne `0` si la conversion échoue", "non"),
    qa("File::size retourne-t-elle une taille en octets ?", "builtins/File.md", "taille d'un fichier en octets", "oui"),
    qa("File::size retourne-t-elle une taille en bits ?", "builtins/File.md", "taille d'un fichier en octets", "non"),
]

QA_VALIDATION_SPECS = [
    qa("L'option --check produit-elle un binaire ?", "compilation-guide.md", "n'émet aucun binaire", "non"),
    qa("L'option --check sert-elle à l'analyse sémantique uniquement ?", "compilation-guide.md", "Analyse sémantique uniquement", "oui"),
    qa("L'option --no-link génère-t-elle un fichier objet .o ?", "compilation-guide.md", "génère hello.o", "oui"),
    qa("L'option --no-link lie-t-elle directement l'exécutable final ?", "compilation-guide.md", "génère hello.o", "non"),
    qa("Un warning empêche-t-il la compilation ?", "diagnostics.md", "n'empêche pas la compilation", "non"),
    qa("Un warning est-il informatif et non bloquant pour la compilation ?", "diagnostics.md", "n'empêche pas la compilation", "oui"),
    qa("Les messages de diagnostic du compilateur sont-ils en anglais ?", "diagnostics.md", "Langue des messages", "oui"),
    qa("Les messages de diagnostic du compilateur sont-ils en français ?", "diagnostics.md", "Langue des messages", "non"),
    qa("Array::contains utilise-t-elle une comparaison stricte ?", "builtins/Array.md", "comparaison stricte", "oui"),
    qa("Array::contains utilise-t-elle une comparaison approximative ?", "builtins/Array.md", "comparaison stricte", "non"),
    qa("Array::len retourne-t-elle 0 pour un tableau vide ?", "builtins/Array.md", "Array::len([])", "oui"),
    qa("Array::len lève-t-elle une exception pour un tableau vide ?", "builtins/Array.md", "Array::len([])", "non"),
    qa("JSON::decode retourne-t-elle null si le JSON est invalide ?", "builtins/JSON.md", "Retourne `null` (0)", "oui"),
    qa("JSON::decode retourne-t-elle une map vide si le JSON est invalide ?", "builtins/JSON.md", "Retourne `null` (0)", "non"),
    qa("JSON::pretty retourne-t-elle le texte original si le JSON est invalide ?", "builtins/JSON.md", "Retourne la string originale si le JSON est invalide", "oui"),
    qa("JSON::pretty retourne-t-elle une chaîne vide si le JSON est invalide ?", "builtins/JSON.md", "Retourne la string originale si le JSON est invalide", "non"),
    qa("Thread::join bloque-t-il jusqu'à la fin du thread ?", "builtins/Thread.md", "Bloque le thread courant jusqu'à ce que le thread", "oui"),
    qa("Thread::join retourne-t-il avant la fin du thread ?", "builtins/Thread.md", "Bloque le thread courant jusqu'à ce que le thread", "non"),
    qa("Thread::sleep exprime-t-il sa durée en millisecondes ?", "builtins/Thread.md", "Thread::sleep(500)", "oui"),
    qa("Thread::sleep exprime-t-il sa durée en secondes ?", "builtins/Thread.md", "Thread::sleep(500)", "non"),
]

QA_TEST_SPECS = [
    qa("La méthode HTTPRequest utilise-t-elle GET par défaut ?", "builtins/HTTPRequest.md", "Méthode par défaut", "oui"),
    qa("La méthode HTTPRequest utilise-t-elle POST par défaut ?", "builtins/HTTPRequest.md", "Méthode par défaut", "non"),
    qa("HTTPRequest::ok accepte-t-elle le statut HTTP 204 ?", "builtins/HTTPRequest.md", "entre `200` et `299` inclus", "oui"),
    qa("HTTPRequest::ok accepte-t-elle le statut HTTP 404 ?", "builtins/HTTPRequest.md", "entre `200` et `299` inclus", "non"),
    qa("Les handles HTTPRequest sont-ils libérés automatiquement ?", "builtins/HTTPRequest.md", "doit libérer chaque handle explicitement", "non"),
    qa("Faut-il libérer explicitement chaque handle HTTPRequest après son utilisation ?", "builtins/HTTPRequest.md", "doit libérer chaque handle explicitement", "oui"),
    qa("ocaraunit exécute-t-il les fonctions dont le nom se termine par Test ?", "builtins/UnitTest.md", "Toutes les fonctions dont le nom se termine par", "oui"),
    qa("ocaraunit exécute-t-il les fonctions dont le nom se termine par Helper ?", "builtins/UnitTest.md", "Toutes les fonctions dont le nom se termine par", "non"),
    qa("Les méthodes statiques publiques suffixées par Test sont-elles reconnues par ocaraunit ?", "builtins/UnitTest.md", "public static method fooTest", "oui"),
    qa("Les méthodes privées suffixées par Test sont-elles exécutées par ocaraunit ?", "builtins/UnitTest.md", "private → ignorée", "non"),
    qa("Les timestamps DateTime sont-ils exprimés en secondes depuis l'epoch Unix ?", "builtins/DateTime.md", "secondes depuis l'epoch Unix", "oui"),
    qa("Les timestamps DateTime sont-ils exprimés en millisecondes depuis l'epoch Unix ?", "builtins/DateTime.md", "secondes depuis l'epoch Unix", "non"),
    qa("DateTime::fromTimestamp produit-elle une date au format ISO 8601 ?", "builtins/DateTime.md", "Convertit un timestamp en chaîne au format ISO 8601", "oui"),
    qa("DateTime::fromTimestamp produit-elle uniquement une date sans heure ?", "builtins/DateTime.md", "Convertit un timestamp en chaîne au format ISO 8601", "non"),
    qa("Regex::test retourne-t-elle true dès qu'une sous-chaîne correspond au pattern ?", "builtins/Regex.md", "au moins une correspondance", "oui"),
    qa("Regex::test exige-t-elle toujours que la chaîne entière corresponde ?", "builtins/Regex.md", "au moins une correspondance", "non"),
    qa("Regex::replaceAll remplace-t-elle toutes les occurrences correspondantes ?", "builtins/Regex.md", "Remplace **toutes** les occurrences", "oui"),
    qa("Regex::replaceAll remplace-t-elle uniquement la première occurrence ?", "builtins/Regex.md", "Remplace **toutes** les occurrences", "non"),
    qa("SQLite::open ouvre-t-elle ou crée-t-elle une base de données ?", "builtins/SQLite.md", "Ouvre ou crée une base de données SQLite", "oui"),
    qa("SQLite::open peut-elle uniquement ouvrir une base déjà existante ?", "builtins/SQLite.md", "Ouvre ou crée une base de données SQLite", "non"),
    qa("SQLite::queryOne retourne-t-elle une map vide si aucun résultat n'existe ?", "builtins/SQLite.md", "map vide si aucun résultat", "oui"),
    qa("SQLite::queryOne retourne-t-elle null si aucun résultat n'existe ?", "builtins/SQLite.md", "map vide si aucun résultat", "non"),
    qa("SQLite attribue-t-elle le code 103 à une requête qui échoue ?", "builtins/SQLite.md", "103  | QUERY", "oui"),
    qa("SQLite attribue-t-elle le code 102 à une requête qui échoue ?", "builtins/SQLite.md", "103  | QUERY", "non"),
    qa("ui.handlers enregistre-t-elle plusieurs commandes en une fois ?", "builtins/Tauri.md", "plusieurs à la fois", "oui"),
    qa("ui.handlers enregistre-t-elle uniquement une commande à la fois ?", "builtins/Tauri.md", "plusieurs à la fois", "non"),
    qa("Tauri::dialog affiche-t-elle une vraie boîte de dialogue ?", "builtins/Tauri.md", "Ne montre aucune boîte de dialogue", "non"),
    qa("Tauri::dialog retourne-t-elle toujours la chaîne ok ?", "builtins/Tauri.md", "retourne toujours", "oui"),
    qa("La méthode Tauri listen est-elle réellement déclenchée par un événement JavaScript ?", "builtins/Tauri.md", "jamais déclenché par un vrai événement JS", "non"),
    qa("Tauri::listen enregistre-t-elle un callback pour un événement nommé ?", "builtins/Tauri.md", "Enregistre un callback pour un événement nommé", "oui"),
    qa("Tauri::run ouvre-t-elle une vraie fenêtre ?", "builtins/Tauri.md", "Ouvre la vraie fenêtre (bloquant", "oui"),
    qa("Tauri::run est-elle décrite comme une fenêtre simulée uniquement ?", "builtins/Tauri.md", "Ouvre la vraie fenêtre (bloquant", "non"),
    qa("Le statut HTTP 2xx est-il considéré comme réussi par HTTPRequest::ok ?", "builtins/HTTPRequest.md", "entre `200` et `299` inclus", "oui"),
    qa("Le statut HTTP 500 est-il considéré comme réussi par HTTPRequest::ok ?", "builtins/HTTPRequest.md", "entre `200` et `299` inclus", "non"),
    qa("Une méthode UnitTest publique suffixée par Test est-elle exécutée ?", "builtins/UnitTest.md", "Seules les méthodes **publiques**", "oui"),
    qa("Une méthode UnitTest privée suffixée par Test est-elle exécutée ?", "builtins/UnitTest.md", "Seules les méthodes **publiques**", "non"),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def normalized_words(text: str) -> str:
    return " ".join(re.findall(r"\w+", normalized(text), re.UNICODE))


def normalized_document(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    return normalized(text.replace("\r\n", "\n").replace("\r", "\n"))


def write_lines(path: Path, lines: list[str]) -> dict[str, object]:
    content = "".join(line + "\n" for line in lines).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return {"documents": len(lines), "bytes": len(content), "sha256": sha256(content)}


def validate_pairs(
    pairs: list[tuple[str, str]],
    split_documents: list[str],
    forbidden_documents: list[str],
    expected_labels: set[str],
    split_name: str,
) -> None:
    split_text = " ".join(normalized_words(document) for document in split_documents)
    forbidden_text = " ".join(normalized_words(document) for document in forbidden_documents)
    for prompt, answer in pairs:
        if answer not in expected_labels:
            raise ValueError(f"Unexpected {split_name} QA label: {answer}")
        if "Contexte: " not in prompt or " Réponse:" not in prompt:
            raise ValueError(f"{split_name} QA prompt has no delimited source context: {prompt}")
        context = prompt.split("Contexte: ", 1)[1].split(" Réponse:", 1)[0].strip()
        context_words = normalized_words(context)
        if not context_words or context_words not in split_text:
            raise ValueError(f"{split_name} QA context is not supported by its source split: {prompt}")
        if context_words in forbidden_text:
            raise ValueError(f"{split_name} QA context leaks across source-file splits: {prompt}")


def materialize_pairs(
    specifications: list[QASpec],
    source_dir: Path,
    split_files: list[str],
    split_name: str,
) -> list[tuple[str, str]]:
    allowed_files = set(split_files)
    result = []
    for question, source, anchor, answer in specifications:
        if source not in allowed_files:
            raise ValueError(f"{split_name} QA source is assigned to another split: {source}")
        source_text = (source_dir / source).read_text(encoding="utf-8-sig")
        context_line = next(
            (line.strip() for line in source_text.splitlines() if anchor.casefold() in line.casefold()),
            None,
        )
        if context_line is None:
            raise ValueError(f"{split_name} QA source anchor not found in {source}: {anchor}")
        context = re.sub(r"\s*\|\s*", " / ", context_line)
        result.append(pair(question, context, answer))
    return result


def prepare(source_dir: Path, output_dir: Path) -> dict[str, object]:
    if len(ALL_SOURCE_FILES) != len(set(ALL_SOURCE_FILES)):
        raise ValueError("Duplicate source path in Ocara corpus definition")
    if set(VALIDATION_FILES) & set(TEST_FILES):
        raise ValueError("A source document appears in validation and test")
    if not set(VALIDATION_FILES + TEST_FILES) <= set(ALL_SOURCE_FILES):
        raise ValueError("Validation/test source is missing from the corpus inventory")

    source_paths = {
        split: [source_dir / name for name in names]
        for split, names in (
            ("train", TRAIN_FILES),
            ("validation", VALIDATION_FILES),
            ("test", TEST_FILES),
        )
    }
    for path in (path for paths in source_paths.values() for path in paths):
        if not path.is_file():
            raise FileNotFoundError(f"Ocara corpus source document not found: {path}")

    documents = {
        split: [normalized_document(path) for path in paths]
        for split, paths in source_paths.items()
    }
    for split, split_docs in documents.items():
        if not split_docs or any(not document for document in split_docs):
            raise ValueError(f"Ocara {split} split must contain non-empty source documents")

    qa_pairs_by_split = {
        "train": materialize_pairs(QA_TRAIN_SPECS, source_dir, TRAIN_FILES, "train"),
        "validation": materialize_pairs(
            QA_VALIDATION_SPECS, source_dir, VALIDATION_FILES, "validation"
        ),
        "test": materialize_pairs(QA_TEST_SPECS, source_dir, TEST_FILES, "test"),
    }
    if any(len(pairs) % 2 for pairs in qa_pairs_by_split.values()):
        raise ValueError("QA splits must have balanced label counts")
    for split_name, pairs in qa_pairs_by_split.items():
        labels = [answer for _, answer in pairs]
        if set(labels) != QA_LABELS or labels.count("oui") != labels.count("non"):
            raise ValueError(f"Ocara QA {split_name} must be balanced between oui and non")
    prompts = [
        normalized(prompt)
        for pairs in qa_pairs_by_split.values()
        for prompt, _ in pairs
    ]
    if len(prompts) != len(set(prompts)):
        raise ValueError("Ocara QA prompts must be unique across train/validation/test")

    validate_pairs(
        qa_pairs_by_split["train"],
        documents["train"],
        documents["validation"] + documents["test"],
        QA_LABELS,
        "train",
    )
    validate_pairs(
        qa_pairs_by_split["validation"],
        documents["validation"],
        documents["train"] + documents["test"],
        QA_LABELS,
        "validation",
    )
    validate_pairs(
        qa_pairs_by_split["test"],
        documents["test"],
        documents["train"] + documents["validation"],
        QA_LABELS,
        "test",
    )

    split_metadata = {}
    source_metadata = {}
    for split, paths in source_paths.items():
        split_metadata[split] = write_lines(output_dir / f"{split}.txt", documents[split])
        source_metadata[split] = [
            {
                "path": path.relative_to(source_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()),
            }
            for path in paths
        ]

    qa_sources = {
        "qa_train": qa_pairs_by_split["train"] * QA_TRAIN_REPEATS,
        "qa_validation": qa_pairs_by_split["validation"],
        "qa_test": qa_pairs_by_split["test"],
    }
    qa_metadata = {}
    for name, pairs in qa_sources.items():
        qa_metadata[name] = write_lines(
            output_dir / f"{name.replace('_', '-')}.txt",
            [f"{prompt} | {answer}" for prompt, answer in pairs],
        )

    manifest = {
        "format": "ocara-documentation-corpus-v1",
        "purpose": "French technical-language assistance and documentation-grounded QA for Ocara",
        "authorization": (
            "Ocara project documentation was provided by the project user for local training "
            "and evaluation; no redistribution license is asserted."
        ),
        "source_root": (
            "docs/examples/language-model-ocara/docs"
            if source_dir.resolve() == SOURCE_DIR.resolve()
            else "custom --source-dir"
        ),
        "normalization": f"NFC (Unicode {unicodedata.unidata_version}); whitespace collapsed per source file",
        "split_strategy": "fixed source-file-level split; no source file crosses train, validation, and test",
        "qa_train_oversampling_factor": QA_TRAIN_REPEATS,
        "qa_tokenizer_repeat_cap": QA_TOKENIZER_REPEAT_CAP,
        "tasks": [
            {
                "name": "next-token prediction",
                "split": "test",
                "metrics": ["cross_entropy", "perplexity", "bits_per_utf8_byte"],
            },
            {
                "name": "document-grounded binary factual QA",
                "split": "qa_train/qa_validation/qa_test",
                "labels": sorted(QA_LABELS),
                "test_examples": len(qa_pairs_by_split["test"]),
                "class_balance": {
                    "oui": len(qa_pairs_by_split["test"]) // 2,
                    "non": len(qa_pairs_by_split["test"]) // 2,
                },
                "metrics": ["free_generation_accuracy", "closed_set_accuracy", "token_f1", "latency"],
                "selection": "validation only; test remains held out",
            },
        ],
        "sources": source_metadata,
        "splits": split_metadata,
        "qa": qa_metadata,
    }
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/lm/ocara")
    args = parser.parse_args()
    manifest = prepare(args.source_dir, args.output_dir)
    print(json.dumps(manifest["splits"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
