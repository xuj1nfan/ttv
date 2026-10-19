"""Export blinded factual claims, then score human annotations against NLI traces."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import isclose, log
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.data import read_jsonl, text_fingerprint, write_json, write_jsonl

LABELS = ("support", "conflict", "insufficient")


def read_trace(path: Path) -> tuple[dict, dict]:
    cases, checkpoints = {}, {}
    for _, row in read_jsonl(path):
        if row.get("record_type") == "case_result":
            cases[row["case_id"]] = row["case_text"]
        elif row.get("record_type") == "checkpoint":
            annotation_id = row["metadata"]["checkpoint_id"]
            if annotation_id in checkpoints:
                raise ValueError("Repeated checkpoint IDs; use a trace from one unique batch run")
            checkpoints[annotation_id] = row
    return cases, checkpoints


def export_annotations(trace: Path, *, limit: int | None = None, seed: int = 42, unique: bool = False) -> list[dict]:
    cases, checkpoints = read_trace(trace)
    rows = [{
        "annotation_id": key, "case_id": row["case_id"],
        "case_text": cases[row["case_id"]], "claim": row["claim"]["text"],
        "gold_label": None, "evidence": None, "notes": "",
    } for key, row in checkpoints.items() if row["case_id"] in cases]
    if unique:
        deduplicated = {}
        for row in rows:
            key = (row["case_id"], row["claim"])
            if key not in deduplicated:
                deduplicated[key] = row
                row["duplicate_annotation_ids"] = []
            else:
                deduplicated[key]["duplicate_annotation_ids"].append(row["annotation_id"])
        rows = list(deduplicated.values())
    if limit is not None and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    # Deliberately omit predictions, probabilities, interventions and gold charge.
    return rows


def score_annotations(trace: Path, annotations: Path, thresholds: list[float]) -> dict:
    cases, checkpoints = read_trace(trace)
    seen, labeled, unannotated = set(), [], 0
    for _, item in read_jsonl(annotations):
        key = item["annotation_id"]
        if key in seen or key not in checkpoints:
            raise ValueError(f"Duplicate or unknown annotation_id: {key}")
        seen.add(key)
        row = checkpoints[key]
        if item.get("case_id") != row["case_id"] or item.get("claim") != row["claim"]["text"]:
            raise ValueError(f"Changed annotation claim or case ID: {key}")
        if row["case_id"] not in cases or text_fingerprint(item.get("case_text", "")) != text_fingerprint(cases[row["case_id"]]):
            raise ValueError(f"Changed annotation fact: {key}")
        gold = item.get("gold_label")
        if gold in (None, ""):
            unannotated += 1
            continue
        if gold not in LABELS:
            raise ValueError(f"Invalid human label for {key}: {gold}")
        decision = row["decision"]
        if decision["metadata"].get("evaluated"):
            probabilities = decision["probabilities"]
            if set(probabilities) != set(LABELS) or not all(0 <= p <= 1 for p in probabilities.values()) or not isclose(sum(probabilities.values()), 1, abs_tol=1e-6):
                raise ValueError(f"Invalid three-way NLI probabilities: {key}")
        labeled.append((gold, decision))
    if not labeled:
        raise ValueError("No human labels yet; fill gold_label with support/conflict/insufficient")
    evaluated = [(gold, decision) for gold, decision in labeled if decision["metadata"].get("evaluated")]
    confusion = {gold: dict(Counter(decision["label"] for actual, decision in labeled if actual == gold)) for gold in LABELS}
    sweeps = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be in [0, 1]")
        triggered = [(gold, decision) for gold, decision in evaluated if decision["label"] == "conflict" and decision["probabilities"]["conflict"] >= threshold]
        sweeps.append({
            "threshold": threshold, "triggered": len(triggered),
            "conflict_precision": sum(gold == "conflict" for gold, _ in triggered) / len(triggered) if triggered else None,
            "coverage_over_annotated": len(triggered) / len(labeled),
        })
    result = {
        "annotated": len(labeled), "unannotated": unannotated, "evaluated_by_nli": len(evaluated),
        "nli_coverage": len(evaluated) / len(labeled),
        "gold_counts": dict(Counter(gold for gold, _ in labeled)), "confusion": confusion,
        "accuracy_on_evaluated": sum(gold == decision["label"] for gold, decision in evaluated) / len(evaluated) if evaluated else None,
        "conflict_thresholds": sweeps,
        "note": "Threshold selection is development-only. Model confidence is not calibrated by this report. Stratified/synthetic annotations may not reflect deployment precision.",
    }
    if evaluated:
        result["nll_on_evaluated"] = -sum(log(max(decision["probabilities"][gold], 1e-15)) for gold, decision in evaluated) / len(evaluated)
        result["brier_on_evaluated"] = sum(sum((probability - int(label == gold)) ** 2 for label, probability in decision["probabilities"].items()) for gold, decision in evaluated) / len(evaluated)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--trace", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--limit", type=int)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--unique", action="store_true", help="Export each distinct case and claim once, retaining duplicate checkpoint IDs")
    score = commands.add_parser("score")
    score.add_argument("--trace", type=Path, required=True)
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--thresholds", nargs="+", type=float, default=[0.8, 0.9, 0.95])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    if args.command == "export":
        if args.limit is not None and args.limit < 1:
            parser.error("limit must be positive")
        rows = export_annotations(args.trace, limit=args.limit, seed=args.seed, unique=args.unique)
        if not rows:
            parser.error("No checkpoints from completed cases to annotate")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output, rows)
        print(f"Exported {len(rows)} blinded claims to {args.output}")
    else:
        report = score_annotations(args.trace, args.annotations, args.thresholds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
