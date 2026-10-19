"""Freeze a short-fact, single-defendant, single-charge CAIL pilot subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.data import file_sha256, sample_cail, scan_training, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "dev", "test"):
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nli-tokenizer", default="joeddav/xlm-roberta-large-xnli")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-fact-tokens", type=int, default=400)
    parser.add_argument("--keep-answer-cues", action="store_true", help="Keep suspected answer cues; default excludes them conservatively")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory to preserve frozen samples")
    if min(args.dev_size, args.test_size, args.max_fact_tokens) < 1:
        parser.error("Sample sizes and token limit must be positive")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.nli_tokenizer, revision=args.revision, local_files_only=args.local_files_only)
    labels, seen = scan_training(args.train)
    manifest = {
        "dataset": "CAIL2018 filtered pilot; not the full official benchmark",
        "seed": args.seed, "tokenizer": args.nli_tokenizer, "revision": args.revision,
        "tokenizer_resolved_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "max_fact_tokens": args.max_fact_tokens, "exclude_answer_cues": not args.keep_answer_cues,
        "deduplication": "whitespace-normalized SHA256; train -> all dev -> test",
        "source_files": {split: {"path": str(getattr(args, split).resolve()), "sha256": file_sha256(getattr(args, split))} for split in ("train", "dev", "test")},
        "label_source": "all accusations in training only", "label_count": len(labels), "splits": {},
    }
    outputs, audit = {}, []
    for split, size in (("dev", args.dev_size), ("test", args.test_size)):
        rows, counts, exclusions = sample_cail(
            getattr(args, split), split=split, labels=labels, seen=seen, tokenizer=tokenizer,
            sample_size=size, seed=args.seed, max_fact_tokens=args.max_fact_tokens,
            exclude_answer_cues=not args.keep_answer_cues,
        )
        if not rows:
            parser.error(f"No eligible {split} cases; inspect source files and filtering settings")
        outputs[split] = rows
        manifest["splits"][split] = {**counts, "requested": size}
        audit.extend(exclusions)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "labels.json", labels)
    for split, rows in outputs.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    write_jsonl(args.output_dir / "answer_cue_exclusions.jsonl", audit)
    manifest["prepared_sha256"] = {name: file_sha256(args.output_dir / name) for name in ("labels.json", "dev.jsonl", "test.jsonl")}
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    print(f"Frozen data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
