"""Run paired real-model CAIL pilots through a local Qwen chat endpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework import FixedCheckpointScheduler, ProcessVerificationRunner, ThresholdInterventionPolicy
from framework.data import file_sha256, load_cases, write_json
from framework.evaluation import classification_metrics, paired_comparison
from framework.intervention import NoInterventionPolicy, SelfCheckPolicy
from framework.labels import validate_labels
from framework.local_generator import FINAL_OUTPUT_PROTOCOL, LocalChatClient, LocalChatGenerator
from framework.nli_verifier import NLIVerifier, UncheckedVerifier


GROUPS = ("baseline", "shadow", "self_check", "nli_conflict", "nli_all")


def make_runner(group: str, client, labels: list[str], config: dict, nli=None, reporter=None):
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group}")
    if group in {"shadow", "nli_conflict", "nli_all"} and nli is None:
        raise ValueError(f"Group {group} requires an NLI verifier")
    verifier = nli if group in {"shadow", "nli_conflict", "nli_all"} else UncheckedVerifier()
    if group in {"baseline", "shadow"}:
        policy = NoInterventionPolicy()
    elif group == "self_check":
        policy = SelfCheckPolicy()
    else:
        settings = dict(config.get("intervention", {}))
        settings["enabled_labels"] = ["conflict"] if group == "nli_conflict" else ["conflict", "insufficient"]
        policy = ThresholdInterventionPolicy(**settings)
    return ProcessVerificationRunner(
        LocalChatGenerator(client, labels, seed=config.get("seed", 42)), verifier,
        FixedCheckpointScheduler(step_checkpoints=[1, 2]), policy, max_steps=3, reporter=reporter,
    )


def run_group(group: str, cases: list[dict], labels: list[str], client, config: dict, output: Path, *, nli=None, verbose: bool = False) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    runner = make_runner(group, client, labels, config, nli, print if verbose else None)
    trace_path, predictions = output / "traces.jsonl", []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            started = perf_counter()
            row = {
                "case_id": case["case_id"], "text_sha256": case["text_sha256"],
                "gold_charge": case["gold_charge"], "split": case.get("split"), "group": group,
            }
            try:
                # Gold labels and the prepared record are NEVER passed to models.
                result = runner.run(case["case_id"], case["case_text"], trace_path=trace_path)
                row.update(status="ok", prediction=result.answer, intervention_count=sum(item.intervene for item in result.trace))
            except Exception as error:
                row.update(status="error", prediction=None, error_type=type(error).__name__, error=str(error))
                row["intervention_count"] = len((runner.last_state.metadata if runner.last_state else {}).get("feedback_history", []))
            metadata = runner.last_state.metadata if runner.last_state else {}
            row.update(
                elapsed_seconds=perf_counter() - started, usage=metadata.get("usage"),
                usage_missing_calls=metadata.get("usage_missing_calls", 1),
                verification_usage=metadata.get("verification_usage", {}),
                valid_charge=row.get("prediction") in labels if row["status"] == "ok" else False,
            )
            predictions.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"[{group}] {index}/{len(cases)} {case['case_id']} {row['status']}", flush=True)
    write_json(output / "metrics.json", classification_metrics(predictions, labels))
    return predictions


def summarize(groups: dict[str, list[dict]], labels: list[str], *, resamples: int = 1000, seed: int = 42, expected_groups: list[str] | None = None) -> dict:
    result = {
        "complete": set(groups) == set(expected_groups if expected_groups is not None else groups),
        "groups": {name: classification_metrics(rows, labels) for name, rows in groups.items()},
        "paired_vs_baseline": {},
    }
    if "baseline" in groups:
        result["paired_vs_baseline"] = {
            name: paired_comparison(groups["baseline"], rows, resamples=resamples, seed=seed)
            for name, rows in groups.items() if name != "baseline"
        }
    return result


def resolve_config_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def package_versions() -> dict:
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "sentencepiece"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/first_round.json")
    parser.add_argument("--data", type=Path, help="Prepared JSONL; explicit relative paths use current directory")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=["baseline", "shadow"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if len(set(args.groups)) != len(args.groups):
        parser.error("Duplicate groups are not allowed")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_path = args.data or resolve_config_path(config["data"]["cases"])
    labels_path = args.labels or resolve_config_path(config["data"]["labels"])
    labels = validate_labels(json.loads(labels_path.read_text(encoding="utf-8")))
    cases = load_cases(data_path, labels)
    if args.limit is not None and args.limit < len(cases):
        cases = random.Random(config.get("seed", 42)).sample(cases, args.limit)
    client = LocalChatClient(**config["generator"])
    client.check_model()
    nli = NLIVerifier(**config["verifier"]) if set(args.groups) & {"shadow", "nli_conflict", "nli_all"} else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = args.output_dir or ROOT / "results/first_round" / run_id
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {
        "config": config, "groups": args.groups, "limit": args.limit, "case_count": len(cases),
        "final_output_protocol": FINAL_OUTPUT_PROTOCOL,
        "python": sys.version, "platform": platform.platform(), "packages": package_versions(),
        "cases_file": str(data_path.resolve()), "cases_sha256": file_sha256(data_path),
        "labels_file": str(labels_path.resolve()), "labels_sha256": file_sha256(labels_path),
        "selected_case_ids": [case["case_id"] for case in cases],
        "status": "running",
        "nli_backend": nli.backend.metadata if nli else None,
        "notes": "Client environment versions only; record the deployed Qwen/vLLM revision separately. Timings include warm-up; no hidden model retries.",
    }
    write_json(output / "run.json", snapshot)
    write_json(output / "labels.json", labels)
    completed = {}
    for group in args.groups:
        completed[group] = run_group(group, cases, labels, client, config, output / group, nli=nli, verbose=args.verbose)
        write_json(output / "summary.json", summarize(completed, labels, seed=config.get("seed", 42), expected_groups=args.groups))
    snapshot["status"] = "complete_with_errors" if any(row["status"] == "error" for rows in completed.values() for row in rows) else "complete"
    write_json(output / "run.json", snapshot)
    print(f"Results: {output.resolve()}")
    for group, rows in completed.items():
        metrics = classification_metrics(rows, labels)
        print(f"{group}: accuracy={metrics['accuracy']:.4f}, errors={metrics['errors']}, invalid={metrics['invalid_outputs']}")
    # Finished with an auditable report, but signal operational failures to scripts.
    if any(row["status"] == "error" for rows in completed.values() for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
