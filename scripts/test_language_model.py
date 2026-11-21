#!/usr/bin/env python3
"""Exercise language-model CLI loading, evaluation, generation, and exact resume."""

import subprocess
import sys
import tempfile
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "bin/gloomy"
EXAMPLE = ROOT / "docs/examples/language-model"
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_language_model import metrics, qa_pairs, records, validate_splits
from prepare_gloomy_docs_corpus import prepare as prepare_gloomy_docs


def run(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(BINARY), *arguments],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def train_args(
    model: Path,
    epochs: int,
    *options: str,
    train: Path | None = None,
    validation: Path | None = None,
    qa_train: Path | None = None,
) -> tuple[str, ...]:
    return (
        "lm", "train",
        "--train", str(train or EXAMPLE / "train.txt"),
        "--validation", str(validation or EXAMPLE / "validation.txt"),
        "--qa-train", str(qa_train or EXAMPLE / "qa-train.txt"),
        "--model", str(model),
        "--batch-size", "2",
        "--epochs", str(epochs),
        *options,
    )


def main() -> None:
    assert metrics("qa_token_f1=0.25 qa_exact_match=0") == {
        "qa_token_f1": "0.25",
        "qa_exact_match": "0",
    }

    if not BINARY.is_file():
        raise SystemExit("bin/gloomy not found; run `make` first")

    qa_test_path = EXAMPLE / "qa-test.txt"
    validate_splits(
        records(EXAMPLE / "train.txt"),
        records(EXAMPLE / "validation.txt"),
        records(EXAMPLE / "test.txt"),
        EXAMPLE / "qa-train.txt",
        qa_test_path,
    )
    try:
        validate_splits(
            ["Le phare d'Orme s'allume à l'aube."],
            ["Document de validation sans réponse."],
            ["Le phare d'Orme s'allume à l'aube lorsque la brume approche."],
            None,
            qa_test_path,
        )
    except ValueError as error:
        assert "leaks" in str(error)
    else:
        raise AssertionError("QA fact leakage from training into test was not detected")

    with tempfile.TemporaryDirectory(prefix="gloomy-language-tests-") as directory:
        work = Path(directory)
        corpus_dir = work / "gloomy-docs"
        corpus_manifest = prepare_gloomy_docs(corpus_dir)
        docs_qa_test = corpus_dir / "qa-test.txt"
        docs_qa_validation = corpus_dir / "qa-validation.txt"
        validate_splits(
            records(corpus_dir / "train.txt"),
            records(corpus_dir / "validation.txt"),
            records(corpus_dir / "test.txt"),
            corpus_dir / "qa-train.txt",
            docs_qa_test,
            {"oui", "non"},
            docs_qa_validation,
        )
        unlabeled_context = work / "qa-label-without-context.txt"
        unlabeled_context.write_text("Question: Le fait est-il vrai ? | oui\n")
        try:
            validate_splits(
                records(corpus_dir / "train.txt"),
                records(corpus_dir / "validation.txt"),
                records(corpus_dir / "test.txt"),
                None,
                unlabeled_context,
                {"oui", "non"},
            )
        except ValueError as error:
            assert "require a source context" in str(error)
        else:
            raise AssertionError("Closed QA label without a source context was accepted")
        assert len(corpus_manifest["sources"]["train"]) == 9
        assert len(corpus_manifest["sources"]["validation"]) == 2
        assert len(corpus_manifest["sources"]["test"]) == 2
        assert corpus_manifest["qa"]["qa_train"]["documents"] == 64
        assert corpus_manifest["qa_tokenizer_repeat_cap"] == 8
        assert corpus_manifest["qa"]["qa_validation"]["documents"] == 6
        assert corpus_manifest["qa"]["qa_test"]["documents"] == 6
        test_answers = [answer for _, answer in qa_pairs(docs_qa_test)]
        assert test_answers.count("oui") == test_answers.count("non") == 3
        assert json.loads((corpus_dir / "manifest.json").read_text()) == corpus_manifest

        missing = run(
            "lm", "evaluate", "--model", str(work / "missing.gloomy"),
            "--text", str(EXAMPLE / "test.txt"), expected=1,
        )
        assert "make language-model-example" in missing.stderr

        full = work / "full.gloomy"
        resumed = work / "resumed.gloomy"
        model_options = (
            "--vocab", "320", "--context", "32", "--dimension", "8",
            "--heads", "2", "--layers", "1",
            "--learning-rate", "0.003", "--seed", "42",
        )
        run(*train_args(full, 2, *model_options))
        run(*train_args(resumed, 1, *model_options))
        run(*train_args(resumed, 1, "--resume"))
        assert full.read_bytes() == resumed.read_bytes(), (
            "resuming after one epoch did not reproduce uninterrupted two-epoch training"
        )

        evaluation = run(
            "lm", "evaluate", "--model", str(resumed),
            "--text", str(EXAMPLE / "test.txt"),
            "--qa", str(qa_test_path),
            "--qa-label", "À l'aube.",
            "--qa-label", "Dans une boîte bleue.",
        )
        assert "perplexity=" in evaluation.stdout
        assert "qa_exact_match=" in evaluation.stdout
        assert "qa_token_f1=" in evaluation.stdout
        assert "qa_generation_ms_per_example=" in evaluation.stdout
        assert "qa_generation_tokens_per_second=" in evaluation.stdout
        assert "qa_closed_set_accuracy=" in evaluation.stdout
        assert "qa_closed_set_total=2" in evaluation.stdout
        invalid_closed_set = run(
            "lm", "evaluate", "--model", str(resumed),
            "--qa", str(qa_test_path),
            "--qa-label", "À l'aube.",
            "--qa-label", "À L'AUBE.",
            expected=1,
        )
        assert "unique" in invalid_closed_set.stderr
        missing_closed_label = run(
            "lm", "evaluate", "--model", str(resumed),
            "--qa", str(qa_test_path),
            "--qa-label", "À l'aube.",
            "--qa-label", "Un autre lieu.",
            expected=1,
        )
        assert "not included" in missing_closed_label.stderr

        generation_args = (
            "lm", "generate", "--model", str(resumed),
            "--prompt", "Question: Quel animal miaule ? Réponse:",
            "--tokens", "8", "--temperature", "0", "--seed", "42",
        )
        first = run(*generation_args)
        second = run(*generation_args)
        assert first.stdout == second.stdout, "greedy generation changed for identical checkpoint and prompt"

        if os.environ.get("GLOOMY_TEST_DOC_RESUME") == "1":
            docs_full = work / "docs-full.gloomy"
            docs_resumed = work / "docs-resumed.gloomy"
            docs_options = (
                "--vocab", "512", "--context", "32", "--dimension", "8",
                "--heads", "2", "--layers", "1",
                "--learning-rate", "0.003", "--seed", "42",
            )
            docs_args = {
                "train": corpus_dir / "train.txt",
                "validation": corpus_dir / "validation.txt",
                "qa_train": corpus_dir / "qa-train.txt",
            }
            run(*train_args(docs_full, 2, *docs_options, **docs_args))
            run(*train_args(docs_resumed, 1, *docs_options, **docs_args))
            run(*train_args(docs_resumed, 1, "--resume", **docs_args))
            assert docs_full.read_bytes() == docs_resumed.read_bytes(), (
                "document-corpus resume did not reproduce uninterrupted training"
            )
            docs_evaluation = run(
                "lm", "evaluate", "--model", str(docs_resumed),
                "--text", str(corpus_dir / "test.txt"),
                "--qa", str(docs_qa_test),
                "--qa-label", "oui",
                "--qa-label", "non",
            )
            assert "perplexity=" in docs_evaluation.stdout
            assert "qa_exact_match=" in docs_evaluation.stdout
            assert "qa_token_f1=" in docs_evaluation.stdout
            assert "qa_closed_set_total=6" in docs_evaluation.stdout
            docs_validation_qa = run(
                "lm", "evaluate", "--model", str(docs_resumed),
                "--qa", str(docs_qa_validation),
                "--qa-label", "oui",
                "--qa-label", "non",
            )
            assert "qa_exact_match=" in docs_validation_qa.stdout
            assert "qa_closed_set_total=6" in docs_validation_qa.stdout

    print("Language-model CLI tests passed.")


if __name__ == "__main__":
    main()
