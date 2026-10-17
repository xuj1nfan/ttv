"""Run a portable mock process-verification experiment using only stdlib."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

# Support the requested direct invocation without an installation step.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework import (  # noqa: E402
    FixedCheckpointScheduler, MockGenerator, MockVerifier,
    ProcessVerificationRunner, ThresholdInterventionPolicy,
)


def main() -> None:
    """Load configuration, assemble replaceable components and run one case."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/process_verification.json",
    )
    parser.add_argument("--trace-path", type=Path, help="Append JSONL to this path")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    generator_config = dict(config["generator"])
    verifier_config = dict(config["verifier"])
    scheduler_config = dict(config["scheduler"])
    for component, settings, supported in (
        ("generator", generator_config, "mock"),
        ("verifier", verifier_config, "mock"),
        ("scheduler", scheduler_config, "fixed"),
    ):
        if settings.pop("type") != supported:
            raise ValueError(f"Unsupported {component} type; this demo supports {supported}")
    generator_config["feedback_responses"] = {
        int(step): text for step, text in generator_config.get("feedback_responses", {}).items()
    }
    runner = ProcessVerificationRunner(
        MockGenerator(**generator_config), MockVerifier(**verifier_config),
        FixedCheckpointScheduler(**scheduler_config),
        ThresholdInterventionPolicy(**config["intervention"]),
        **config.get("runner", {}), reporter=print,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    trace_dir = Path(config["output"]["trace_dir"])
    if not trace_dir.is_absolute():
        trace_dir = PROJECT_ROOT / trace_dir
    trace_path = args.trace_path or trace_dir / f"process-verification-{run_id}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the exact experiment settings beside the trace for reproducibility.
    config_path = (
        trace_path.with_name(f"{trace_path.stem}.{run_id}.config.json")
        if args.trace_path else trace_path.with_suffix(".config.json")
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    result = runner.run(config["case"]["id"], config["case"]["text"], trace_path=trace_path)
    print(f"Interventions: {sum(record.intervene for record in result.trace)}")
    print(f"Trace: {trace_path.resolve()}")
    print(f"Config snapshot: {config_path.resolve()}")


if __name__ == "__main__":
    main()
