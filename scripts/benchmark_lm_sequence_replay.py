#!/usr/bin/env python3
"""Compare document-window replay strategies during one Ocara adaptation phase."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchmark_language_model import metrics, records, timed
from prepare_gloomy_docs_corpus import prepare as prepare_gloomy_docs
import prepare_ocara_corpus


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_records(path: Path, rows: list[str]) -> None:
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")


def evaluate(binary: Path, model: Path, text: Path, qa: Path) -> tuple[dict[str, str], float, int]:
    output, wall, rss = timed([
        str(binary), "lm", "evaluate", "--model", str(model),
        "--text", str(text), "--qa", str(qa), "--qa-label", "oui", "--qa-label", "non",
    ])
    return metrics(output), wall, rss


def document_losses(binary: Path, model: Path, documents: list[str], directory: Path) -> list[float]:
    losses = []
    for index, document in enumerate(documents):
        print(f"Scoring priority candidate {index + 1}/{len(documents)}", flush=True)
        path = directory / f"priority-{index}.txt"
        write_records(path, [document])
        output, _, _ = timed([
            str(binary), "lm", "evaluate", "--model", str(model), "--text", str(path),
            "--compute-precision", "float32",
        ])
        losses.append(float(metrics(output)["cross_entropy"]))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/gloomy")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/gloomy-docs.gloomy")
    parser.add_argument("--replay-documents", type=int, default=3)
    parser.add_argument("--phase-b-documents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/examples/language-model/sequence-replay-pilot.csv")
    args = parser.parse_args()
    if not args.binary.is_file() or not args.model.is_file():
        parser.error("binary and phase-A checkpoint must exist; run make language-model-docs-example first")
    if args.replay_documents < 1 or args.phase_b_documents < 2:
        parser.error("replay capacity must be positive and phase B must contain at least two documents")

    with tempfile.TemporaryDirectory(prefix="gloomy-sequence-replay-") as temporary:
        work = Path(temporary)
        gloomy = work / "gloomy"
        ocara = work / "ocara"
        prepare_gloomy_docs(gloomy)
        prepare_ocara_corpus.prepare(prepare_ocara_corpus.SOURCE_DIR, ocara)
        old_train = records(gloomy / "train.txt")
        all_new_train = records(ocara / "train.txt")
        phase_b_count = min(args.phase_b_documents, len(all_new_train))
        phase_b_indices = [round(i * (len(all_new_train) - 1) / (phase_b_count - 1)) for i in range(phase_b_count)]
        new_train = [all_new_train[i] for i in phase_b_indices]
        old_test, new_test = gloomy / "test.txt", ocara / "test.txt"
        old_qa, new_qa = gloomy / "qa-test.txt", ocara / "qa-test.txt"
        priority_indices = [round(i * (len(old_train) - 1) / (min(6, len(old_train)) - 1)) for i in range(min(6, len(old_train)))]
        priority_documents = [old_train[i] for i in priority_indices]
        print(f"Scoring {len(priority_documents)} phase-A documents for prioritized replay", flush=True)
        losses = document_losses(args.binary, args.model, priority_documents, work)
        count = min(args.replay_documents, len(old_train))
        rng = random.Random(args.seed)
        selections = {
            "none": [],
            "fifo": old_train[-count:],
            "reservoir": rng.sample(old_train, count),
        }
        priorities = [max(loss, 1e-9) for loss in losses]
        selected_prioritized = []
        weighted_rng = random.Random(args.seed)
        population = list(range(len(priority_documents)))
        weights = priorities[:]
        for _ in range(count):
            chosen = weighted_rng.choices(population, weights=weights, k=1)[0]
            selected_prioritized.append(priority_documents[chosen])
        selections["prioritized_loss"] = selected_prioritized
        selections["prioritized_corrected"] = selected_prioritized
        loss_by_document = dict(zip(priority_documents, losses))
        uniform_loss = sum(losses) / len(losses)
        priority_probabilities = [priority / sum(priorities) for priority in priorities]
        prioritized_expected_loss = sum(probability * loss for probability, loss in zip(priority_probabilities, losses))
        expected_sampling_bias = prioritized_expected_loss - uniform_loss

        rows = []
        for strategy, replay in selections.items():
            print(f"Training and evaluating strategy: {strategy}", flush=True)
            model = work / f"{strategy}.gloomy"
            shutil.copyfile(args.model, model)
            training_path = work / f"{strategy}-phase-b.txt"
            write_records(training_path, new_train + replay)
            sample_weights_path = work / f"{strategy}-sample-weights.txt"
            replay_weights = []
            if strategy == "prioritized_corrected":
                for document in replay:
                    probability = priorities[priority_documents.index(document)] / sum(priorities)
                    replay_weights.append(1.0 / (len(priority_documents) * probability))
            sample_weights_path.write_text("".join("1.0\n" for _ in new_train) + "".join(f"{weight:.17g}\n" for weight in replay_weights or [1.0] * len(replay)), encoding="utf-8")
            train_command = [
                str(args.binary), "lm", "train", "--resume", "--train", str(training_path),
                "--validation", str(ocara / "validation.txt"), "--model", str(model),
                "--batch-size", "2", "--epochs", "1",
                "--sample-weights", str(sample_weights_path),
            ]
            train_output, train_wall, train_rss = timed(train_command)
            train_values = metrics(train_output)
            old_metrics, old_wall, old_rss = evaluate(args.binary, model, old_test, old_qa)
            new_metrics, new_wall, new_rss = evaluate(args.binary, model, new_test, new_qa)
            selected_loss = (
                sum(loss_by_document[doc] for doc in replay) / len(replay)
                if replay and strategy.startswith("prioritized")
                else None
            )
            rows.append({
                "strategy": strategy,
                "replay_documents": len(replay),
                "replay_selection": ";".join(str(old_train.index(doc)) for doc in replay),
                "uniform_priority_pool_cross_entropy": f"{uniform_loss:.9g}",
                "selected_replay_cross_entropy": f"{selected_loss:.9g}" if selected_loss is not None else "",
                "replay_loss_sampling_bias": f"{expected_sampling_bias:.9g}" if strategy.startswith("prioritized") else "",
                "importance_weight_mean": f"{sum(replay_weights) / len(replay_weights):.9g}" if replay_weights else "",
                "replay_buffer_estimated_bytes": sum(len(doc.encode("utf-8")) for doc in replay),
                "updates": train_values.get("updates", ""),
                "train_wall_seconds": f"{train_wall:.9g}",
                "train_max_rss_kib": train_rss,
                "old_test_cross_entropy": old_metrics.get("cross_entropy", ""),
                "old_test_perplexity": old_metrics.get("perplexity", ""),
                "old_qa_exact_match": old_metrics.get("qa_exact_match", ""),
                "old_qa_closed_set_accuracy": old_metrics.get("qa_closed_set_accuracy", ""),
                "old_eval_wall_seconds": f"{old_wall:.9g}",
                "old_eval_max_rss_kib": old_rss,
                "new_test_cross_entropy": new_metrics.get("cross_entropy", ""),
                "new_test_perplexity": new_metrics.get("perplexity", ""),
                "new_qa_exact_match": new_metrics.get("qa_exact_match", ""),
                "new_qa_closed_set_accuracy": new_metrics.get("qa_closed_set_accuracy", ""),
                "new_eval_wall_seconds": f"{new_wall:.9g}",
                "new_eval_max_rss_kib": new_rss,
                "checkpoint_bytes": model.stat().st_size,
            })

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "format": "gloomy-language-sequence-replay-pilot-v1",
            "phase_a": "Gloomy documentation training split and existing Gloomy checkpoint",
            "phase_b": "Evenly spaced subset of Ocara training documents; one resumed epoch, no QA supervision",
            "replay_unit": "source document; the CLI tokenizes each replayed line into overlapping training windows",
            "strategies": {
                "none": "no phase-A documents replayed",
                "fifo": "last N phase-A documents",
                "reservoir": "N uniformly sampled phase-A documents, fixed seed",
                "prioritized_loss": "N documents sampled with replacement proportional to phase-A per-document cross-entropy, without correction",
                "prioritized_corrected": "Same prioritized samples with beta=1 document-level importance weight 1/(N*p_i); each document contributes its token-average loss and the denominator is independent of importance weights",
            },
            "replay_documents": args.replay_documents,
            "phase_b_documents": len(new_train),
            "phase_b_source_indices": phase_b_indices,
            "prioritized_phase_a_source_indices": priority_indices,
            "seed": args.seed,
            "priority_scoring_precision": "float32",
            "phase_a_model": {"path": str(args.model), "bytes": args.model.stat().st_size, "sha256": sha256(args.model)},
            "phase_a_test_sha256": sha256(old_test),
            "phase_b_test_sha256": sha256(new_test),
            "phase_a_qa_sha256": sha256(old_qa),
            "phase_b_qa_sha256": sha256(new_qa),
            "results": args.output.name,
            "limitations": [
                "single training seed, phase order and replay capacity; exploratory comparison, not a general ranking",
                "QA metrics are reported but phase-B adaptation trains next-token text only",
                "training updates differ because replay documents produce different numbers of token windows; a fixed-update comparison remains to be run",
            ],
        }
        manifest_path = args.output.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output} and {manifest_path}")


if __name__ == "__main__":
    main()
