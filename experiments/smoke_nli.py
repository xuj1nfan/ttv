"""Run a few transparent three-way probes against the configured real NLI model."""

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.data import file_sha256, read_jsonl, write_json
from framework.nli_verifier import NLIVerifier
from framework.schema import Claim


def probe_premise(probe: dict, cases: dict[str, str]) -> str:
    if "case_id" in probe:
        try:
            return cases[probe["case_id"]]
        except KeyError as error:
            raise ValueError(f"Unknown probe case_id: {probe['case_id']}") from error
    return probe["premise"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/first_round.json")
    parser.add_argument("--probes", type=Path, default=ROOT / "examples/nli_probe.jsonl")
    parser.add_argument("--cases", type=Path, help="Prepared cases whose case_text is used when probes specify case_id")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/nli_probe" / uuid4().hex[:8])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    verifier = NLIVerifier(**config["verifier"])
    cases = {row["case_id"]: row["case_text"] for _, row in read_jsonl(args.cases)} if args.cases else {}
    rows = []
    for _, probe in read_jsonl(args.probes):
        premise = probe_premise(probe, cases)
        decision = verifier.verify(premise, Claim(probe["hypothesis"], 1))
        row = {
            "probe_id": probe["probe_id"],
            "predicted_label": decision.label, "probabilities": decision.probabilities,
            "metadata": decision.metadata,
        }
        if probe.get("expected_label") is not None:
            row["expected_label"] = probe["expected_label"]
        rows.append(row)
        expected = f" expected={row['expected_label']}" if "expected_label" in row else ""
        print(f"{row['probe_id']}:{expected} predicted={row['predicted_label']} probability={decision.confidence:.4f}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "results.json", {
        "model": verifier.backend.metadata,
        "probe_file": str(args.probes.resolve()), "probe_sha256": file_sha256(args.probes),
        "case_file": str(args.cases.resolve()) if args.cases else None,
        "case_sha256": file_sha256(args.cases) if args.cases else None,
        "probes": rows,
        "correct": sum(row["expected_label"] == row["predicted_label"] for row in rows if "expected_label" in row)
        if any("expected_label" in row for row in rows) else None,
        "labeled_total": sum("expected_label" in row for row in rows),
        "total": len(rows),
        "note": "Accuracy is computed only when probes include expected_label. Unlabeled probes are predictions, not an accuracy benchmark.",
    })
    print(f"Probe results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
