#!/usr/bin/env python3
"""Compare float64 and fake-quantized Transformer weights on one fixed checkpoint."""

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

from benchmark_language_model import metrics, records, timed


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "bin/gloomy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.binary, args.model, args.text, args.qa):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    rows = []
    test_utf8_bytes = sum(len(document.encode("utf-8")) for document in records(args.text))
    variants = (
        ("float64", None, None),
        ("int8_fake_weights", "int8", None),
        ("int16_fake_weights", "int16", None),
        ("float32_compute", None, "float32"),
        ("int8_compute_activations", None, "int8"),
        ("int16_compute_activations", None, "int16"),
    )
    for label, weight_precision, compute_precision in variants:
        command = [
            str(args.binary), "lm", "evaluate", "--model", str(args.model),
            "--text", str(args.text), "--qa", str(args.qa),
            "--qa-label", "oui", "--qa-label", "non",
        ]
        if weight_precision:
            command.extend(("--weight-precision", weight_precision))
        if compute_precision:
            command.extend(("--compute-precision", compute_precision))
        output, wall_seconds, max_rss_kib = timed(command)
        values = metrics(output)
        values["test_utf8_bytes"] = str(test_utf8_bytes)
        if test_utf8_bytes and "cross_entropy" in values and "tokens" in values:
            values["bits_per_byte"] = f"{float(values['cross_entropy']) * int(values['tokens']) / test_utf8_bytes / math.log(2.0):.9g}"
        values["variant"] = label
        values["wall_seconds"] = f"{wall_seconds:.9g}"
        values["max_rss_kib"] = str(max_rss_kib)
        rows.append(values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "weight_precision_bits", "weight_max_abs_error", "weight_mean_abs_error",
              "compute_precision_bits", "activation_precision_bits",
              "cross_entropy", "perplexity", "tokens", "test_utf8_bytes", "bits_per_byte", "qa_exact_match", "qa_token_f1",
              "qa_closed_set_accuracy", "qa_generation_ms_per_example",
              "qa_generation_tokens_per_second", "wall_seconds", "max_rss_kib"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    compiler = subprocess.run(["g++", "--version"], check=True, text=True,
                              stdout=subprocess.PIPE).stdout.splitlines()[0]
    manifest = {
        "format": "gloomy-transformer-weight-precision-pilot-v1",
        "method": {
            "int8_fake_weights/int16_fake_weights": "Per-matrix symmetric fake quantization followed by dequantization into float64 parameters.",
            "float32_compute": "Float32-rounded graph arithmetic and dot products.",
            "int8_compute_activations/int16_compute_activations": "Inputs and weights quantized per input vector/output channel; integer dot products accumulate in int64 and dequantize outputs. Q/K/V, attention outputs, and feed-forward activations are quantized per vector.",
            "all_modes": "Biases and normalization parameters stay float64; integer modes use float64 biases and normalization math.",
        },
        "memory_interpretation": "Checkpoint parameters remain float64 and integer operands are quantized on the fly, not stored in a packed checkpoint. Reduced inference skips autodiff graph nodes for linear layers; RSS measures this implementation path and is not a packed-weight storage comparison.",
        "timing_interpretation": "Reduced modes use direct linear kernels while the float64 reference builds autodiff nodes; observed speed differences include this inference-path change and cannot be attributed only to numeric precision.",
        "model": {"path": str(args.model), "bytes": args.model.stat().st_size, "sha256": sha256(args.model)},
        "text_test": {"path": str(args.text), "bytes": args.text.stat().st_size, "sha256": sha256(args.text)},
        "qa_test": {"path": str(args.qa), "bytes": args.qa.stat().st_size, "sha256": sha256(args.qa)},
        "machine": platform.platform(),
        "cpu": " / ".join(dict.fromkeys(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines()
            if line.lower().startswith("model name")
        )),
        "compiler": compiler,
        "results": args.output.name,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} and {manifest_path}")


if __name__ == "__main__":
    main()
