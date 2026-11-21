#!/usr/bin/env python3
"""Normalize, deduplicate, and deterministically split plain-text documents."""

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path


def read_documents(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    chunks = (re.split(r"\n\s*\n", text.strip()) if re.search(r"\n\s*\n", text)
              else text.splitlines())
    documents = []
    for chunk in chunks:
        normalized = " ".join(chunk.split())
        if normalized:
            documents.append(normalized)
    return documents


def unique_documents(documents: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    unique = []
    for document in documents:
        fingerprint = hashlib.sha256(document.encode("utf-8")).digest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(document)
    return unique, len(documents) - len(unique)


def write_split(path: Path, documents: list[str]) -> dict[str, object]:
    payload = "".join(document + "\n" for document in documents).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {
        "documents": len(documents),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 source; blank-line-separated paragraphs or one document per line")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Corpus name or source URL")
    parser.add_argument("--license", required=True, help="License or authorization basis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    outputs = [args.train.resolve(), args.validation.resolve(), args.test.resolve(), args.manifest.resolve()]
    if len(set(outputs)) != len(outputs) or args.input.resolve() in outputs:
        parser.error("input, split outputs, and manifest must use distinct paths")
    if not (0 < args.validation_ratio < 1 and 0 < args.test_ratio < 1
            and args.validation_ratio + args.test_ratio < 1):
        parser.error("validation/test ratios must be positive and sum to less than 1")
    if not args.source.strip() or not args.license.strip():
        parser.error("source and license metadata must be non-empty")

    raw = args.input.read_bytes()
    documents, duplicates = unique_documents(read_documents(args.input))
    if len(documents) < 3:
        parser.error("at least three unique non-empty documents are required")
    random.Random(args.seed).shuffle(documents)
    validation_count = max(1, round(len(documents) * args.validation_ratio))
    test_count = max(1, round(len(documents) * args.test_ratio))
    if validation_count + test_count >= len(documents):
        parser.error("ratios leave no document for the training split")
    train_count = len(documents) - validation_count - test_count
    train = documents[:train_count]
    validation = documents[train_count:train_count + validation_count]
    test = documents[train_count + validation_count:]

    split_metadata = {
        "train": write_split(args.train, train),
        "validation": write_split(args.validation, validation),
        "test": write_split(args.test, test),
    }
    manifest = {
        "format": "gloomy-corpus-split-v1",
        "generator_version": sys.version.split()[0],
        "source": args.source,
        "license": args.license,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "normalization": f"NFC (Unicode {unicodedata.unidata_version}); whitespace collapsed per document",
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "duplicate_documents_removed": duplicates,
        "splits": split_metadata,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_name(args.manifest.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(args.manifest)
    print(json.dumps(manifest["splits"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
