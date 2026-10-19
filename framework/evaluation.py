"""Exact-label evaluation and paired corrections, including failed cases."""

from __future__ import annotations

from collections import Counter
from math import ceil, floor
import random

from .labels import validate_labels


def index_predictions(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        if not isinstance(row.get("case_id"), str) or row["case_id"] in indexed:
            raise ValueError("Missing or duplicate case_id in predictions")
        if row.get("status") not in {"ok", "error"}:
            raise ValueError("Prediction status must be ok or error")
        indexed[row["case_id"]] = row
    if not indexed:
        raise ValueError("No prediction records")
    return indexed


def is_correct(row: dict) -> bool:
    return row["status"] == "ok" and row.get("prediction") == row["gold_charge"]


def classification_metrics(rows: list[dict], labels: list[str]) -> dict:
    labels = validate_labels(labels)
    index_predictions(rows)
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        gold, predicted = row["gold_charge"], row.get("prediction")
        if gold not in labels:
            raise ValueError(f"Unknown gold charge: {gold}")
        support[gold] += 1
        if is_correct(row):
            tp[gold] += 1
        else:
            fn[gold] += 1
            if row["status"] == "ok" and predicted in labels:
                fp[predicted] += 1
    per_class = {}
    for label in labels:
        denominator = 2 * tp[label] + fp[label] + fn[label]
        per_class[label] = {
            "support": support[label], "tp": tp[label], "fp": fp[label], "fn": fn[label],
            "f1": 2 * tp[label] / denominator if denominator else 0.0,
        }
    observed = [value["f1"] for value in per_class.values() if value["support"]]
    complete_usage = all(row.get("usage_missing_calls", 1) == 0 and row.get("usage") is not None for row in rows)
    available_usage = {key: sum((row.get("usage") or {}).get(key, 0) for row in rows) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return {
        "n": len(rows), "accuracy": sum(is_correct(row) for row in rows) / len(rows),
        "macro_f1_all_labels": sum(value["f1"] for value in per_class.values()) / len(labels),
        "macro_f1_present_gold_labels": sum(observed) / len(observed),
        "classes_with_gold_support": len(observed), "vocabulary_size": len(labels),
        "errors": sum(row["status"] == "error" for row in rows),
        "invalid_outputs": sum(row["status"] == "ok" and row.get("prediction") not in labels for row in rows),
        "interventions": sum(row.get("intervention_count", 0) for row in rows),
        "elapsed_seconds": sum(row.get("elapsed_seconds", 0) for row in rows),
        "mean_case_seconds": sum(row.get("elapsed_seconds", 0) for row in rows) / len(rows),
        "nli_model_calls": sum(row.get("verification_usage", {}).get("model_calls", 0) for row in rows),
        "nli_abstentions": sum(row.get("verification_usage", {}).get("abstentions", 0) for row in rows),
        "nli_input_tokens": sum(row.get("verification_usage", {}).get("input_tokens", 0) for row in rows),
        "verification_seconds": sum(row.get("verification_usage", {}).get("elapsed_seconds", 0) for row in rows),
        "usage_complete": complete_usage,
        "tokens": available_usage if complete_usage else None,
        "reported_tokens_partial": available_usage,
        "per_class": per_class,
    }


def paired_comparison(baseline: list[dict], treatment: list[dict], *, resamples: int = 1000, seed: int = 42) -> dict:
    before, after = index_predictions(baseline), index_predictions(treatment)
    if before.keys() != after.keys():
        raise ValueError("Paired evaluation requires identical case ID sets, including failures")
    counts, differences = Counter(), []
    for case_id in sorted(before):
        left, right = before[case_id], after[case_id]
        for key in ("gold_charge", "text_sha256"):
            if not left.get(key) or left[key] != right.get(key):
                raise ValueError(f"Paired case mismatch for {case_id}: {key}")
        a, b = is_correct(left), is_correct(right)
        counts[(a, b)] += 1
        differences.append(int(b) - int(a))
    n = len(differences)
    result = {
        "n": n, "wrong_to_right": counts[(False, True)], "right_to_wrong": counts[(True, False)],
        "both_correct": counts[(True, True)], "both_wrong": counts[(False, False)],
        "delta_accuracy": sum(differences) / n,
        "correction_rate_among_baseline_wrong": counts[(False, True)] / sum(not is_correct(row) for row in baseline) if any(not is_correct(row) for row in baseline) else None,
        "harm_rate_among_baseline_correct": counts[(True, False)] / sum(is_correct(row) for row in baseline) if any(is_correct(row) for row in baseline) else None,
    }
    if resamples > 0:
        rng = random.Random(seed)
        estimates = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(resamples))
        result["paired_bootstrap_95ci"] = [estimates[floor(0.025 * (resamples - 1))], estimates[ceil(0.975 * (resamples - 1))]]
        result["bootstrap_resamples"] = resamples
    return result
