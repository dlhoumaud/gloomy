#!/usr/bin/env python3
"""Exercise language-model CLI loading, evaluation, generation, and exact resume."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "bin/gloomy"
EXAMPLE = ROOT / "docs/examples/language-model"
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_language_model import metrics, records, validate_splits


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


def train_args(model: Path, epochs: int, *options: str) -> tuple[str, ...]:
    return (
        "lm", "train",
        "--train", str(EXAMPLE / "train.txt"),
        "--validation", str(EXAMPLE / "validation.txt"),
        "--qa-train", str(EXAMPLE / "qa-train.txt"),
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
        )
        assert "perplexity=" in evaluation.stdout
        assert "qa_exact_match=" in evaluation.stdout
        assert "qa_token_f1=" in evaluation.stdout
        assert "qa_generation_ms_per_example=" in evaluation.stdout
        assert "qa_generation_tokens_per_second=" in evaluation.stdout

        generation_args = (
            "lm", "generate", "--model", str(resumed),
            "--prompt", "Question: Quel animal miaule ? Réponse:",
            "--tokens", "8", "--temperature", "0", "--seed", "42",
        )
        first = run(*generation_args)
        second = run(*generation_args)
        assert first.stdout == second.stdout, "greedy generation changed for identical checkpoint and prompt"

    print("Language-model CLI tests passed.")


if __name__ == "__main__":
    main()
