#!/usr/bin/env python3
"""Run reproducible language-model train/validation/test trials and retain raw metrics."""

import argparse
import csv
from collections import Counter
import hashlib
import json
import platform
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
from pathlib import Path


def metrics(output: str) -> dict[str, str]:
    return dict(re.findall(r"([a-z][a-z0-9_]*)=([0-9.eE+-]+)", output))


def records(path: Path) -> list[str]:
    text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n"))
    paragraphs = re.split(r"\n\s*\n", text.strip()) if re.search(r"\n\s*\n", text) else text.splitlines()
    return [" ".join(paragraph.split()) for paragraph in paragraphs if paragraph.strip()]


def qa_pairs(path: Path) -> list[tuple[str, str]]:
    result = []
    for line in records(path):
        if " | " not in line:
            raise ValueError(f"invalid QA row in {path}; expected prompt | answer")
        prompt, answer = line.split(" | ", 1)
        if not prompt or not answer:
            raise ValueError(f"empty QA prompt or answer in {path}")
        result.append((prompt, answer))
    return result


def normalized_words(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold(), re.UNICODE))


def validate_splits(
    train: list[str],
    validation: list[str],
    test: list[str],
    qa_train_path: Path | None,
    qa_test_path: Path,
    allowed_qa_labels: set[str] | None = None,
    qa_validation_path: Path | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    splits = [set(map(normalized_words, part)) for part in (train, validation, test)]
    if any(not part for part in (train, validation, test)):
        raise ValueError("train, validation, and test must each contain documents")
    if splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2]:
        raise ValueError("train, validation, and test documents must be disjoint")

    qa_train = qa_pairs(qa_train_path) if qa_train_path else []
    qa_validation = qa_pairs(qa_validation_path) if qa_validation_path else []
    qa_test = qa_pairs(qa_test_path)
    if not qa_test:
        raise ValueError("QA test set must contain at least one pair")
    train_prompts = {normalized_words(prompt) for prompt, _ in qa_train}
    validation_prompts = {normalized_words(prompt) for prompt, _ in qa_validation}
    test_prompts = {normalized_words(prompt) for prompt, _ in qa_test}
    if train_prompts & validation_prompts or train_prompts & test_prompts or validation_prompts & test_prompts:
        raise ValueError("QA train, validation, and test prompts must be disjoint")

    train_and_validation = " ".join(normalized_words(doc) for part in (train, validation) for doc in part)
    train_text = " ".join(normalized_words(doc) for doc in train)
    validation_text = " ".join(normalized_words(doc) for doc in validation)
    test_text = " ".join(normalized_words(doc) for doc in test)
    allowed_labels = {normalized_words(label) for label in (allowed_qa_labels or set())}
    for prompt, answer in qa_validation:
        normalized_answer = normalized_words(answer)
        if "Contexte: " in prompt:
            context = normalized_words(prompt.split("Contexte: ", 1)[1].split(" Réponse:", 1)[0])
            if not context or context not in validation_text:
                raise ValueError(f"QA validation context is not supported by a validation document: {prompt}")
            if context in train_text or context in test_text:
                raise ValueError(f"QA validation context leaks from train/test documents: {prompt}")
        if normalized_answer in allowed_labels:
            if "Contexte: " not in prompt:
                raise ValueError(f"Closed QA labels require a source context: {prompt}")
            continue
        if normalized_answer not in validation_text:
            raise ValueError(f"QA validation answer is not supported by a validation document: {prompt}")
        if normalized_answer in train_text or normalized_answer in test_text:
            raise ValueError(f"QA validation answer leaks from train/test documents: {prompt}")
    for prompt, answer in qa_test:
        normalized_answer = normalized_words(answer)
        if "Contexte: " in prompt:
            context = normalized_words(prompt.split("Contexte: ", 1)[1].split(" Réponse:", 1)[0])
            if not context or context not in test_text:
                raise ValueError(f"QA test context is not supported by a test document: {prompt}")
            if context in train_and_validation:
                raise ValueError(f"QA test context leaks from train/validation documents: {prompt}")
        if normalized_answer in allowed_labels:
            if "Contexte: " not in prompt:
                raise ValueError(f"Closed QA labels require a source context: {prompt}")
            continue
        if normalized_answer not in test_text:
            raise ValueError(f"QA test answer is not supported by a test document: {prompt}")
        if normalized_answer in train_and_validation:
            raise ValueError(f"QA test answer leaks from train/validation documents: {prompt}")
    return qa_train, qa_validation, qa_test


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_rss_kib(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def timed(command: list[str]) -> tuple[str, float, int]:
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    max_rss = 0
    finished = threading.Event()

    def monitor_rss() -> None:
        nonlocal max_rss
        while not finished.is_set():
            max_rss = max(max_rss, process_rss_kib(process.pid))
            time.sleep(0.001)
        max_rss = max(max_rss, process_rss_kib(process.pid))

    monitor = threading.Thread(target=monitor_rss, daemon=True)
    monitor.start()
    start = time.perf_counter()
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - start
    finished.set()
    monitor.join()
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, command, stdout=stdout, stderr=stderr
        )
    return stdout, elapsed, max_rss


def cpu_model() -> str:
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        for line in cpu_info.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[-1].strip()
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("./bin/gloomy"))
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--qa-train", type=Path)
    parser.add_argument("--qa-validation", type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument(
        "--qa-allowed-label",
        action="append",
        default=[],
        help="Closed QA reference label that need not occur verbatim in source documents",
    )
    parser.add_argument("--corpus-source", required=True, help="Corpus source or original fixture identifier")
    parser.add_argument("--corpus-authorization", required=True, help="License or authorization basis")
    parser.add_argument("--source-manifest", type=Path, help="Optional manifest linking splits to source documents")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--vocab", type=int, default=320)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("seeds must be a non-empty comma-separated list without duplicates")
    if not args.corpus_source.strip() or not args.corpus_authorization.strip():
        parser.error("corpus source and authorization metadata must be non-empty")
    if not args.binary.is_file():
        parser.error(f"model executable not found: {args.binary}")

    train_records = records(args.train)
    validation_records = records(args.validation)
    test_records = records(args.test)
    try:
        qa_train, qa_validation, qa_test = validate_splits(
            train_records, validation_records, test_records, args.qa_train, args.qa,
            set(args.qa_allowed_label),
            args.qa_validation,
        )
    except ValueError as error:
        parser.error(str(error))

    compiler = subprocess.run(
        ["g++", "--version"], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.splitlines()[0]
    paths = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
        "qa_test": args.qa,
    }
    if args.qa_train:
        paths["qa_train"] = args.qa_train
    if args.qa_validation:
        paths["qa_validation"] = args.qa_validation
    corpus = {
        name: {
            "path": str(path),
            "documents": len(records(path)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    }
    source_manifest = None
    if args.source_manifest:
        source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        declared_splits = source_manifest.get("splits", {})
        for name in ("train", "validation", "test"):
            metadata = declared_splits.get(name, {})
            if metadata.get("sha256") != corpus[name]["sha256"]:
                parser.error(f"source manifest SHA-256 mismatch for {name} split")
        qa_metadata = source_manifest.get("qa", {})
        for name in ("qa_train", "qa_validation", "qa_test"):
            path = paths.get(name)
            if path is None:
                continue
            metadata = qa_metadata.get(name, {})
            if metadata.get("sha256") != corpus[name]["sha256"]:
                parser.error(f"source manifest SHA-256 mismatch for {name}")
    manifest = {
        "format": "gloomy-language-benchmark-v3",
        "corpus_source": args.corpus_source,
        "corpus_authorization": args.corpus_authorization,
        "corpus": corpus,
        "corpus_provenance": source_manifest,
        "seeds": seeds,
        "model": {
            "vocab": args.vocab,
            "context": args.context,
            "dimension": args.dimension,
            "heads": args.heads,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        "target": {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "cpu_model": cpu_model(),
            "logical_cpus": __import__("os").cpu_count(),
            "compiler": compiler,
            "binary": str(args.binary.resolve()),
            "binary_sha256": sha256(args.binary),
            "timer": "Python standard-library perf_counter",
            "rss_measurement": "Linux /proc/<pid>/status VmHWM sampled at 1 ms; may miss short peaks",
            "rss_unit": "KiB",
        },
    }
    columns = [
        "seed", "updates", "parameters", "train_cross_entropy",
        "validation_cross_entropy", "validation_perplexity", "test_cross_entropy",
        "test_perplexity", "test_tokens", "test_eval_wall_seconds",
        "test_eval_max_rss_kib", "test_tokens_per_second", "qa_exact_match",
        "qa_token_f1", "qa_correct", "qa_total", "qa_generated_tokens",
        "qa_majority_baseline",
        "qa_validation_exact_match", "qa_validation_correct",
        "qa_validation_total",
        "qa_generation_seconds", "qa_generation_ms_per_example",
        "qa_generation_tokens_per_second",
        "train_wall_seconds", "train_max_rss_kib", "checkpoint_bytes",
        "train_documents", "validation_documents", "test_documents",
        "train_bytes", "validation_bytes", "test_bytes", "train_sha256",
        "validation_sha256", "test_sha256", "cpu_model", "compiler",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="gloomy-lm-benchmark-") as temp_dir:
        temporary = Path(temp_dir)
        for seed in seeds:
            model = temporary / f"model-{seed}.gloomy"
            command = [
                str(args.binary.resolve()), "lm", "train",
                "--train", str(args.train), "--validation", str(args.validation),
                "--model", str(model), "--vocab", str(args.vocab),
                "--context", str(args.context), "--dimension", str(args.dimension),
                "--heads", str(args.heads), "--layers", str(args.layers),
                "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
                "--learning-rate", str(args.learning_rate), "--seed", str(seed),
            ]
            if args.qa_train:
                command.extend(["--qa-train", str(args.qa_train)])
            training_output, train_seconds, train_rss = timed(command)
            train_epochs = [metrics(line) for line in training_output.splitlines() if line.startswith("epoch=")]
            if not train_epochs:
                raise RuntimeError(f"training produced no epoch metrics for seed {seed}")
            train = train_epochs[-1]
            evaluation, eval_seconds, eval_rss = timed(
                [
                    str(args.binary.resolve()), "lm", "evaluate", "--model", str(model),
                    "--text", str(args.test),
                ]
            )
            test = metrics(next(line for line in evaluation.splitlines() if line.startswith("cross_entropy=")))
            qa_output, _, _ = timed(
                [
                    str(args.binary.resolve()), "lm", "evaluate", "--model", str(model),
                    "--qa", str(args.qa),
                ]
            )
            qa = metrics(next(line for line in qa_output.splitlines() if line.startswith("qa_exact_match=")))
            qa_validation = {}
            if args.qa_validation:
                qa_validation_output, _, _ = timed(
                    [
                        str(args.binary.resolve()), "lm", "evaluate", "--model", str(model),
                        "--qa", str(args.qa_validation),
                    ]
                )
                qa_validation = metrics(
                    next(line for line in qa_validation_output.splitlines() if line.startswith("qa_exact_match="))
                )
            qa_answer_counts = Counter(
                normalized_words(answer) for _, answer in qa_train
            )
            rows.append({
                "seed": seed,
                "updates": train["updates"],
                "parameters": train["parameters"],
                "train_cross_entropy": train["train_cross_entropy"],
                "validation_cross_entropy": train["validation_cross_entropy"],
                "validation_perplexity": train["validation_perplexity"],
                "test_cross_entropy": test["cross_entropy"],
                "test_perplexity": test["perplexity"],
                "test_tokens": test["tokens"],
                "test_eval_wall_seconds": eval_seconds,
                "test_eval_max_rss_kib": eval_rss,
                "test_tokens_per_second": float(test["tokens"]) / eval_seconds if eval_seconds else 0,
                "qa_exact_match": qa["qa_exact_match"],
                "qa_token_f1": qa["qa_token_f1"],
                "qa_correct": qa["correct"],
                "qa_total": qa["total"],
                "qa_majority_baseline": (
                    max(qa_answer_counts.values()) / len(qa_train)
                    if qa_train else 0
                ),
                "qa_validation_exact_match": qa_validation.get("qa_exact_match", ""),
                "qa_validation_correct": qa_validation.get("correct", ""),
                "qa_validation_total": qa_validation.get("total", ""),
                "qa_generated_tokens": qa["generated_tokens"],
                "qa_generation_seconds": qa["qa_generation_seconds"],
                "qa_generation_ms_per_example": qa["qa_generation_ms_per_example"],
                "qa_generation_tokens_per_second": qa["qa_generation_tokens_per_second"],
                "train_wall_seconds": train_seconds,
                "train_max_rss_kib": train_rss,
                "checkpoint_bytes": model.stat().st_size,
                "train_documents": len(train_records),
                "validation_documents": len(validation_records),
                "test_documents": len(test_records),
                "train_bytes": corpus["train"]["bytes"],
                "validation_bytes": corpus["validation"]["bytes"],
                "test_bytes": corpus["test"]["bytes"],
                "train_sha256": corpus["train"]["sha256"],
                "validation_sha256": corpus["validation"]["sha256"],
                "test_sha256": corpus["test"]["sha256"],
                "cpu_model": manifest["target"]["cpu_model"],
                "compiler": compiler,
            })
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} raw seed records to {args.output}")
    print(f"Wrote reproducibility manifest to {manifest_path}")


if __name__ == "__main__":
    main()
