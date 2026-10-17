"""Synchronous reasoning -> claim -> verification -> policy -> continuation loop."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

from .generator import Generator
from .intervention import InterventionPolicy
from .scheduler import Scheduler
from .schema import RunResult, TraceRecord
from .verifier import Verifier


class ProcessVerificationRunner:
    """Coordinate injected components without knowing their models or thresholds."""

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        scheduler: Scheduler,
        intervention_policy: InterventionPolicy,
        *,
        max_steps: int = 1000,
        reporter: Callable[[str], None] | None = None,
    ) -> None:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.generator = generator
        self.verifier = verifier
        self.scheduler = scheduler
        self.policy = intervention_policy
        self.max_steps = max_steps
        self.reporter = reporter

    def _report(self, text: str) -> None:
        if self.reporter is not None:
            self.reporter(text)

    def run(
        self, case_id: str, case_text: str, *, trace_path: str | Path | None = None,
    ) -> RunResult:
        """Run one case and optionally append checkpoint/final records to JSONL.

        Each checkpoint is written after one continuation step. On an exception,
        a pending checkpoint is still saved, with null reasoning_after, followed
        by a run_error record. Use a distinct path per run if case IDs may repeat
        across experimental configurations. Files are appended, never truncated.
        """
        state = self.generator.start(case_text)
        trace: list[TraceRecord] = []
        pending: TraceRecord | None = None
        path = Path(trace_path) if trace_path is not None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

        def save(record: dict[str, Any]) -> None:
            if path is not None:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        def advance() -> None:
            nonlocal pending
            if state.current_step >= self.max_steps:
                raise RuntimeError("Generator exceeded max_steps before finishing")
            previous_step = state.current_step
            previous_length = len(state.steps)
            self.generator.step(state)
            if state.current_step != previous_step + 1 or len(state.steps) != previous_length + 1:
                raise RuntimeError("Generator.step must advance one step and append one text")
            self._report(f"Step {state.current_step}: {state.steps[-1]}")
            if pending is not None:
                pending.reasoning_after = deepcopy(state.steps)
                pending.metadata["observed_at_step"] = state.current_step
                save(pending.to_dict())
                pending = None

            # A finished generator has no remaining opportunity to consume feedback.
            if state.finished:
                return
            checkpoint = self.scheduler.get_checkpoint(state)
            if checkpoint is None:
                return
            claim = self.generator.extract_claim(state)
            decision = self.verifier.verify(state.case_text, claim)
            action = self.policy.decide(decision)
            record = TraceRecord(
                case_id=case_id, checkpoint=deepcopy(checkpoint),
                reasoning_before=deepcopy(state.steps), claim=deepcopy(claim),
                decision=deepcopy(decision), intervene=action.intervene,
                feedback=action.message if action.intervene else None,
                intervention_reason=action.reason,
                metadata={"checkpoint_id": f"{case_id}:{len(trace) + 1}"},
            )
            trace.append(record)
            pending = record
            self._report(
                f"Checkpoint {checkpoint.name}\nClaim: {claim.text}\n"
                f"Verifier: {decision.label}\nProbability: {decision.probabilities}\n"
                f"Intervention: {'yes' if action.intervene else 'no'}\n"
                f"Feedback: {record.feedback or '(none)'}"
            )
            if action.intervene:
                # Intervention validates that feedback is nonempty when enabled.
                assert action.message is not None
                self.generator.feedback(state, action.message)

        try:
            while not state.finished:
                advance()
        except Exception as error:
            if pending is not None:
                pending.metadata["continuation_status"] = "unobserved_due_to_error"
                save(pending.to_dict())
                pending = None
            save({
                "schema_version": "1.0", "record_type": "run_error", "case_id": case_id,
                "step": state.current_step, "error_type": type(error).__name__,
                "message": str(error),
            })
            raise
        finally:
            if pending is not None:
                save(pending.to_dict())

        save({
            "schema_version": "1.0", "record_type": "case_result", "case_id": case_id,
            "case_text": state.case_text, "answer": state.answer, "finished": state.finished,
            "steps": state.steps, "intervention_count": sum(x.intervene for x in trace),
            "metadata": deepcopy(state.metadata),
        })
        self._report(f"Final Answer: {state.answer}")
        return RunResult(case_id, state, trace)
