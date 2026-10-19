"""Recompute metrics from saved predictions without rerunning any model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.data import read_jsonl, write_json
from framework.labels import validate_labels
from experiments.run_first_round import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    if args.resamples < 0:
        parser.error("resamples cannot be negative")
    labels = validate_labels(json.loads((args.run_dir / "labels.json").read_text(encoding="utf-8")))
    snapshot = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    expected = set(snapshot["selected_case_ids"])
    groups = {}
    for name in snapshot["groups"]:
        path = args.run_dir / name / "predictions.jsonl"
        rows = [row for _, row in read_jsonl(path)]
        if {row["case_id"] for row in rows} != expected:
            raise ValueError(f"Incomplete group {name}; do not score a partially interrupted run as complete")
        groups[name] = rows
    summary = summarize(groups, labels, resamples=args.resamples, seed=snapshot["config"].get("seed", 42))
    write_json(args.run_dir / "summary.json", summary)
    print(json.dumps({name: {key: value for key, value in metrics.items() if key != "per_class"} for name, metrics in summary["groups"].items()}, ensure_ascii=False, indent=2))
    print(json.dumps(summary["paired_vs_baseline"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
