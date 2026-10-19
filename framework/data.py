"""Streaming, reproducible CAIL preparation without exposing labels to models."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re

from .labels import normalize_charge


# Conservative screening. Export excluded line numbers for manual inspection;
# no rewriting/deletion inside a fact, and never match against the gold charge.
ANSWER_CUES = re.compile(r"本院认为|判决如下|裁定如下|(?:构成|涉嫌)[^。；\n]{1,30}罪|以[^。；\n]{1,30}罪(?:提起公诉|起诉|追究)|犯[^。；\n]{1,20}罪(?:[，。、；\s]|$)")


def text_fingerprint(text: str) -> str:
    return sha256("".join(text.split()).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be objects")
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            yield line_number, row


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def scan_training(path: Path) -> tuple[list[str], set[str]]:
    labels, hashes = set(), set()
    for line, row in read_jsonl(path):
        fact = row.get("fact")
        meta = row.get("meta", {})
        accusations = meta.get("accusation") if isinstance(meta, dict) else None
        if not isinstance(fact, str) or not fact.strip() or not isinstance(accusations, list) or not accusations:
            raise ValueError(f"Invalid CAIL training row at {path}:{line}")
        labels.update(normalize_charge(value) for value in accusations)
        hashes.add(text_fingerprint(fact))
    if not labels:
        raise ValueError("Training data is empty")
    return sorted(labels), hashes


def sample_cail(
    path: Path, *, split: str, labels: list[str], seen: set[str],
    tokenizer, sample_size: int, seed: int, max_fact_tokens: int = 400,
    exclude_answer_cues: bool = True,
) -> tuple[list[dict], dict, list[dict]]:
    """Reservoir sample eligible cases. `seen` includes ALL earlier split facts.

    Even rejected/unselected dev facts enter seen, preventing their appearance in
    the frozen test set. Membership and order depend only on files and fixed seed.
    """
    if sample_size < 1 or max_fact_tokens < 1:
        raise ValueError("sample_size and max_fact_tokens must be positive")
    rng, counts, sample, audit = random.Random(seed), Counter(), [], []
    vocabulary = set(labels)
    for line, row in read_jsonl(path):
        counts["total"] += 1
        fact = row.get("fact")
        meta = row.get("meta", {})
        if not isinstance(fact, str) or not fact.strip() or not isinstance(meta, dict):
            counts["invalid"] += 1
            continue
        fingerprint = text_fingerprint(fact)
        duplicate = fingerprint in seen
        seen.add(fingerprint)
        accusations, criminals = meta.get("accusation"), meta.get("criminals")
        reason = None
        if duplicate:
            reason = "duplicate_or_earlier_split"
        elif not isinstance(accusations, list) or len(accusations) != 1:
            reason = "not_single_charge"
        elif not isinstance(criminals, list) or len(criminals) != 1:
            reason = "not_single_defendant"
        else:
            charge = normalize_charge(accusations[0])
            if charge not in vocabulary:
                reason = "charge_absent_from_training"
            elif exclude_answer_cues and ANSWER_CUES.search(fact):
                reason = "possible_answer_cue"
        if reason is None:
            tokens = len(tokenizer.encode(fact, add_special_tokens=False))
            if tokens > max_fact_tokens:
                reason = "fact_too_long"
        if reason:
            counts[reason] += 1
            # Only possible leakage exclusions need text inspection; reference
            # raw source line instead of duplicating the entire dataset.
            if reason == "possible_answer_cue":
                audit.append({"split": split, "source_line": line, "reason": reason, "text_sha256": fingerprint})
            continue
        counts["eligible"] += 1
        prepared = {
            "case_id": f"cail-{split}-{line}", "case_text": fact, "gold_charge": charge,
            "split": split, "source_line": line, "text_sha256": fingerprint, "fact_tokens": tokens,
        }
        if len(sample) < sample_size:
            sample.append(prepared)
        else:
            index = rng.randrange(counts["eligible"])
            if index < sample_size:
                sample[index] = prepared
    sample.sort(key=lambda item: item["source_line"])
    counts["selected"] = len(sample)
    return sample, dict(counts), audit


def load_cases(path: Path, labels: list[str], *, limit: int | None = None) -> list[dict]:
    cases, ids, hashes = [], set(), set()
    for line, row in read_jsonl(path):
        if limit is not None and len(cases) >= limit:
            break
        case_id, text, gold = row.get("case_id"), row.get("case_text"), row.get("gold_charge")
        if not isinstance(case_id, str) or not case_id or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid prepared case at {path}:{line}")
        gold = normalize_charge(gold)
        if gold not in labels:
            raise ValueError(f"Gold charge absent from fixed vocabulary at {path}:{line}")
        fingerprint = text_fingerprint(text)
        if row.get("text_sha256", fingerprint) != fingerprint:
            raise ValueError(f"Changed fact text at {path}:{line}")
        if case_id in ids or fingerprint in hashes:
            raise ValueError(f"Duplicate case ID or text at {path}:{line}")
        ids.add(case_id)
        hashes.add(fingerprint)
        cases.append({**row, "gold_charge": gold, "text_sha256": fingerprint})
    if not cases:
        raise ValueError("No prepared cases to run")
    return cases
