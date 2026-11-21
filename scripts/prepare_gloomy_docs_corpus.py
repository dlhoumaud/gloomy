#!/usr/bin/env python3
"""Prepare a document-level corpus from the repository's Gloomy documentation."""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "train": [
        "README.md",
        "docs/architecture.md",
        "docs/benchmark.md",
        "docs/configurations.md",
        "docs/losses.md",
        "docs/memory.md",
        "docs/optimizers.md",
        "docs/quantization.md",
        "docs/serialization.md",
    ],
    "validation": [
        "docs/activations.md",
        "docs/examples/README.md",
    ],
    "test": [
        "docs/README.md",
        "docs/metrics.md",
    ],
}
QA_TRAIN = [
    (
        "Question: MSE pénalise-t-elle les grandes erreurs ? Contexte: MSE pénalise fortement les grandes erreurs. Réponse:",
        "oui",
    ),
    (
        "Question: SGD conserve-t-il un état supplémentaire par paramètre ? Contexte: Il ne conserve pas d'état supplémentaire par paramètre Réponse:",
        "non",
    ),
    (
        "Question: Huber devient-elle linéaire pour les grandes erreurs ? Contexte: Huber est quadratique près de la cible et linéaire pour les grandes erreurs. Réponse:",
        "oui",
    ),
    (
        "Question: FIFO conserve-t-elle l'ancien élément si elle est pleine ? Contexte: Lorsqu'elle est pleine, l'ajout d'un nouvel échantillon supprime le plus ancien. Réponse:",
        "non",
    ),
    (
        "Question: Le runtime configuré fournit-il un split validation automatique ? Contexte: Il ne fournit pas de split entraînement/validation automatique ni de régularisation. Réponse:",
        "non",
    ),
    (
        "Question: Adam conserve-t-il deux moments par paramètre ? Contexte: Il conserve deux `double` par paramètre Réponse:",
        "oui",
    ),
    (
        "Question: La correction du biais d'échantillonnage est-elle implémentée ? Contexte: La correction de biais d'échantillonnage n'est pas encore implémentée Réponse:",
        "non",
    ),
    (
        "Question: Le stockage int16 utilise-t-il deux octets par valeur ? Contexte: Le stockage des valeurs passe de 8 octets par `double` à 2 octets par valeur Réponse:",
        "oui",
    ),
]
QA_TEST = [
    (
        "Question: La formule MAE est-elle une moyenne d'erreurs absolues ? Contexte: MAE = moyenne(abs(prediction - cible)) Réponse:",
        "oui",
    ),
    (
        "Question: La formule MAE est-elle une moyenne d'erreurs au carré ? Contexte: MAE = moyenne(abs(prediction - cible)) Réponse:",
        "non",
    ),
    (
        "Question: La convention forgetting est-elle avant moins après ? Contexte: forgetting = performance_before - performance_after Réponse:",
        "oui",
    ),
    (
        "Question: La convention forgetting est-elle après moins avant ? Contexte: forgetting = performance_before - performance_after Réponse:",
        "non",
    ),
    (
        "Question: Les bibliothèques du binaire doivent-elles pouvoir être liées statiquement ? Contexte: ne doit dépendre d'aucune bibliothèque qui ne puisse pas être liée statiquement au programme. Réponse:",
        "oui",
    ),
    (
        "Question: L'exécutable peut-il dépendre d'une bibliothèque qui ne peut pas être liée statiquement ? Contexte: ne doit dépendre d'aucune bibliothèque qui ne puisse pas être liée statiquement au programme. Réponse:",
        "non",
    ),
]
QA_VALIDATION = [
    (
        "Question: Sigmoid produit-elle une valeur entre 0 et 1 ? Contexte: `f(z) = 1 / (1 + exp(-z))` produit une valeur entre 0 et 1. Réponse:",
        "oui",
    ),
    (
        "Question: Sigmoid produit-elle une valeur entre -1 et 1 ? Contexte: `f(z) = 1 / (1 + exp(-z))` produit une valeur entre 0 et 1. Réponse:",
        "non",
    ),
    (
        "Question: ReLU est-elle définie par max(0, z) ? Contexte: `f(z) = max(0, z)`. Réponse:",
        "oui",
    ),
    (
        "Question: ReLU est-elle définie par min(0, z) ? Contexte: `f(z) = max(0, z)`. Réponse:",
        "non",
    ),
    (
        "Question: Le runtime accepte-t-il actuellement un fichier texte avec une paire de vecteurs par ligne ? Contexte: Le runtime accepte actuellement un fichier texte avec une paire de vecteurs par ligne Réponse:",
        "oui",
    ),
    (
        "Question: Le runtime fournit-il actuellement des lecteurs CSV ou JSON ? Contexte: Ces exemples varient les dimensions, modes et précisions de mémoire, mais ne constituent pas des lecteurs CSV ou JSON. Réponse:",
        "non",
    ),
]
QA_TRAIN_REPEATS = 8
QA_TOKENIZER_REPEAT_CAP = 8
QA_LABELS = {"oui", "non"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_document(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return " ".join(text.split())


def prompt_context(prompt: str) -> str:
    if "Contexte: " not in prompt or " Réponse:" not in prompt:
        raise ValueError(f"QA prompt must include a delimited source context: {prompt}")
    return prompt.split("Contexte: ", 1)[1].split(" Réponse:", 1)[0].strip()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def write_lines(path: Path, lines: list[str]) -> dict[str, object]:
    content = "".join(line + "\n" for line in lines).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return {"documents": len(lines), "bytes": len(content), "sha256": sha256(content)}


def prepare(output_dir: Path) -> dict[str, object]:
    resolved = {name: [ROOT / source for source in sources] for name, sources in SPLITS.items()}
    all_paths = [path for sources in resolved.values() for path in sources]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("A source document appears in more than one split")
    for path in all_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Corpus source document not found: {path}")

    documents = {
        name: [normalized_document(path) for path in paths]
        for name, paths in resolved.items()
    }
    all_documents = [doc for split in documents.values() for doc in split]
    if any(not doc for doc in all_documents):
        raise ValueError("Corpus source documents must not be empty")
    normalized_splits = [
        {doc.casefold() for doc in documents[name]}
        for name in ("train", "validation", "test")
    ]
    if any(
        normalized_splits[left] & normalized_splits[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Corpus documents must be disjoint across splits")

    qa_text = {
        "qa_train": QA_TRAIN * QA_TRAIN_REPEATS,
        "qa_validation": QA_VALIDATION,
        "qa_test": QA_TEST,
    }
    train_text = " ".join(doc.casefold() for doc in documents["train"])
    validation_text = " ".join(doc.casefold() for doc in documents["validation"])
    train_and_validation = " ".join(
        doc.casefold()
        for split in ("train", "validation")
        for doc in documents[split]
    )
    test_text = " ".join(doc.casefold() for doc in documents["test"])
    for prompt, answer in QA_TRAIN:
        context = normalized(prompt_context(prompt))
        if context not in normalized(train_text):
            raise ValueError(f"QA training context is not supported by a training document: {prompt}")
        if answer.casefold() not in QA_LABELS:
            raise ValueError(f"Unexpected QA training label: {answer}")
    for prompt, answer in QA_TEST:
        context = normalized(prompt_context(prompt))
        if context not in normalized(test_text):
            raise ValueError(f"QA test context is not supported by a test document: {prompt}")
        if context in normalized(train_and_validation):
            raise ValueError(f"QA test context leaks from train/validation documents: {prompt}")
        if answer.casefold() not in QA_LABELS:
            raise ValueError(f"Unexpected QA test label: {answer}")
    for prompt, answer in QA_VALIDATION:
        context = normalized(prompt_context(prompt))
        if context not in normalized(validation_text):
            raise ValueError(f"QA validation context is not supported by a validation document: {prompt}")
        if context in normalized(train_text) or context in normalized(test_text):
            raise ValueError(f"QA validation context leaks from train/test documents: {prompt}")
        if answer.casefold() not in QA_LABELS:
            raise ValueError(f"Unexpected QA validation label: {answer}")

    source_metadata = {}
    split_metadata = {}
    for split, paths in resolved.items():
        split_metadata[split] = write_lines(
            output_dir / f"{split}.txt", documents[split]
        )
        source_metadata[split] = [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()),
            }
            for path in paths
        ]
    qa_metadata = {}
    for name, pairs in qa_text.items():
        qa_path = output_dir / f"{name.replace('_', '-')}.txt"
        qa_metadata[name] = write_lines(
            qa_path, [f"{prompt} | {answer}" for prompt, answer in pairs]
        )

    manifest = {
        "format": "gloomy-project-documentation-corpus-v1",
        "purpose": "French technical-documentation QA for the Gloomy project",
        "authorization": (
            "Local use of repository documentation was requested by the project user; "
            "the repository contains no separate LICENSE file, and this manifest does "
            "not assert a redistribution license."
        ),
        "normalization": f"NFC (Unicode {unicodedata.unidata_version}); whitespace collapsed per source file",
        "split_strategy": "fixed source-file-level split; no source file crosses splits",
        "qa_train_oversampling_factor": QA_TRAIN_REPEATS,
        "qa_tokenizer_repeat_cap": QA_TOKENIZER_REPEAT_CAP,
        "tasks": [
            {
                "name": "next-token prediction",
                "split": "test",
                "metrics": ["cross_entropy", "perplexity"],
            },
            {
                "name": "document-grounded answer generation",
                "split": "qa_train/qa_validation/qa_test",
                "input": "question plus source-document context",
                "labels": sorted(QA_LABELS),
                "metrics": [
                    "free_generation_accuracy",
                    "closed_set_accuracy",
                    "token_f1",
                    "generation_latency",
                    "generation_tokens_per_second",
                ],
                "selection": "validation perplexity and QA accuracy; test is held out",
            },
        ],
        "sources": source_metadata,
        "splits": split_metadata,
        "qa": qa_metadata,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/lm/gloomy-docs",
        help="Output directory for generated splits and provenance manifest",
    )
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(json.dumps(manifest["splits"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
