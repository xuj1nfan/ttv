"""Behavioral tests for timing, feedback delivery, trace observations and contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from framework import (
    Checkpoint, Claim, Decision, FixedCheckpointScheduler, MockGenerator, MockVerifier,
    ProcessVerificationRunner, ReasoningState, Scheduler, ThresholdInterventionPolicy,
    Verifier,
)


def decision(label: str, probability: float) -> Decision:
    """Create a normalized three-class fixture."""
    probabilities = dict.fromkeys(("support", "conflict", "insufficient"), (1 - probability) / 2)
    probabilities[label] = probability
    return Decision(label, probability, probabilities)


class FrameworkTests(unittest.TestCase):
    def test_fractional_scheduler_and_finished_state(self) -> None:
        scheduler = FixedCheckpointScheduler(total_steps=10)
        triggered = []
        for step in range(1, 11):
            checkpoint = scheduler.get_checkpoint(ReasoningState("case", current_step=step))
            if checkpoint:
                triggered.append((step, checkpoint.name))
        self.assertEqual(triggered, [(2, "20%"), (5, "50%"), (8, "80%")])
        self.assertIsNone(scheduler.get_checkpoint(
            ReasoningState("case", current_step=2, finished=True),
        ))

    def test_step_scheduler_and_collisions(self) -> None:
        scheduler = FixedCheckpointScheduler(step_checkpoints=[2, 4])
        self.assertIsNone(scheduler.get_checkpoint(ReasoningState("case", current_step=1)))
        self.assertEqual(scheduler.get_checkpoint(ReasoningState("case", current_step=2)).step, 2)
        merged = FixedCheckpointScheduler([0.2, 0.3], total_steps=3)
        checkpoint = merged.get_checkpoint(ReasoningState("case", current_step=1))
        self.assertEqual(checkpoint.metadata["targets"], ["20%", "30%"])

    def test_scheduler_requires_horizon_only_for_fractions(self) -> None:
        with self.assertRaises(ValueError):
            FixedCheckpointScheduler().get_checkpoint(ReasoningState("case", current_step=1))
        with self.assertRaises(ValueError):
            FixedCheckpointScheduler([1.0])
        with self.assertRaises(ValueError):
            FixedCheckpointScheduler([0.2], step_checkpoints=[2])

    def test_mock_verifier_schema_and_optional_evidence(self) -> None:
        result = MockVerifier().verify("case", Claim("local fact", 1))
        self.assertEqual(result.label, "insufficient")
        self.assertAlmostEqual(sum(result.probabilities.values()), 1)
        self.assertEqual(result.confidence, result.probabilities[result.label])
        self.assertIsNone(result.evidence)

    def test_schema_rejects_invalid_probabilities_and_allows_future_labels(self) -> None:
        for probabilities in (
            {"support": 0.2}, {"support": float("nan")},
            {"support": 1.1, "conflict": -0.1},
        ):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                Decision("support", 0.2, probabilities)
        with self.assertRaises(ValueError):
            Decision("support", 0.9, {"support": 1.0})
        with self.assertRaises(ValueError):
            Decision("support", 1.0000001, {"support": 1.0})
        result = Decision("irrelevant", 1.0, {"irrelevant": 1.0})
        self.assertFalse(ThresholdInterventionPolicy().decide(result).intervene)

    def test_conflict_threshold_is_inclusive_and_configurable(self) -> None:
        policy = ThresholdInterventionPolicy(conflict_threshold=0.8)
        self.assertTrue(policy.decide(decision("conflict", 0.9)).intervene)
        self.assertTrue(policy.decide(decision("conflict", 0.8)).intervene)
        self.assertFalse(policy.decide(decision("conflict", 0.79)).intervene)
        self.assertFalse(ThresholdInterventionPolicy(0.95).decide(decision("conflict", 0.9)).intervene)

    def test_insufficient_and_support(self) -> None:
        policy = ThresholdInterventionPolicy(insufficient_threshold=0.85)
        self.assertTrue(policy.decide(decision("insufficient", 0.9)).intervene)
        self.assertFalse(policy.decide(decision("insufficient", 0.84)).intervene)
        action = policy.decide(decision("support", 0.99))
        self.assertFalse(action.intervene)
        self.assertIsNone(action.message)

    def make_runner(self, generator=None, **kwargs):
        return ProcessVerificationRunner(
            generator or MockGenerator(
                ["supported fact", "wrong fact", "uncorrected continuation", "last step"],
                "baseline", feedback_responses={2: "corrected continuation"},
                answer_after_feedback="revised answer",
            ),
            MockVerifier({
                "supported fact": {"support": 0.95, "conflict": 0.03, "insufficient": 0.02},
                "wrong fact": {"support": 0.03, "conflict": 0.92, "insufficient": 0.05},
            }),
            FixedCheckpointScheduler(step_checkpoints=[1, 2]),
            ThresholdInterventionPolicy(), **kwargs,
        )

    def test_runner_and_trace_include_actual_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/trace.jsonl"
            runner = self.make_runner()
            result = runner.run("case-1", "case facts", trace_path=path)
            self.assertTrue(result.state.finished)
            self.assertEqual(result.answer, "revised answer")
            self.assertEqual(len(result.trace), 2)
            self.assertEqual(len(result.state.metadata["feedback_history"]), 1)
            self.assertEqual(result.trace[0].decision.label, "support")
            self.assertFalse(result.trace[0].intervene)
            conflict = result.trace[1]
            self.assertTrue(conflict.intervene)
            self.assertEqual(conflict.reasoning_before[-1], "wrong fact")
            self.assertEqual(conflict.reasoning_after[-1], "corrected continuation")
            self.assertEqual(len(conflict.reasoning_before), 2)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["record_type"] for row in rows], ["checkpoint", "checkpoint", "case_result"])
            self.assertEqual(rows[1], conflict.to_dict())
            self.assertEqual(rows[-1]["intervention_count"], 1)
            # Components and their schedules must remain reusable across cases.
            repeat = runner.run("case-2", "case facts", trace_path=path)
            self.assertEqual(len(repeat.trace), 2)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 6)

    def test_no_intervention_changes_neither_continuation_nor_answer(self) -> None:
        runner = self.make_runner()
        runner.policy = ThresholdInterventionPolicy(1.0, 1.0)
        result = runner.run("baseline", "case facts")
        self.assertEqual(result.answer, "baseline")
        self.assertEqual(result.trace[1].reasoning_after[-1], "uncorrected continuation")
        self.assertFalse(any(record.intervene for record in result.trace))

    def test_generation_error_preserves_pending_checkpoint(self) -> None:
        class FailingGenerator(MockGenerator):
            def step(self, state):
                if state.current_step == 1:
                    raise RuntimeError("fixture failure")
                super().step(state)

        generator = FailingGenerator(["supported fact", "next"], "answer")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.make_runner(generator).run("case", "facts", trace_path=path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            checkpoint = next(row for row in rows if row["record_type"] == "checkpoint")
            self.assertIsNone(checkpoint["reasoning_after"])
            self.assertEqual(checkpoint["decision"]["label"], "support")
            self.assertTrue(any(row["record_type"] == "run_error" for row in rows))

    def test_max_steps_bounds_unfinished_generation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "max_steps"):
            self.make_runner(max_steps=2).run("case", "facts")

    def test_custom_verifier_and_scheduler_without_runner_changes(self) -> None:
        class CustomVerifier(Verifier):
            def verify(self, case_text, claim):
                self.last_input = (case_text, claim.text)
                return Decision("partially_supported", 1.0, {"partially_supported": 1.0})

        class CustomScheduler(Scheduler):
            def get_checkpoint(self, state):
                # Deliberately selects every step; runner still protects termination.
                return Checkpoint("custom", state.current_step)

        verifier = CustomVerifier()
        runner = ProcessVerificationRunner(
            MockGenerator(["local proposition", "finish"], "final"),
            verifier, CustomScheduler(), ThresholdInterventionPolicy(),
        )
        result = runner.run("custom", "original facts")
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(verifier.last_input, ("original facts", "local proposition"))
        self.assertEqual(result.trace[0].decision.label, "partially_supported")
        self.assertFalse(result.trace[0].intervene)

    def test_demo_runs_outside_repository(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments/run_process_verification.py"
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "demo.jsonl"
            process = subprocess.run(
                [sys.executable, str(script), "--trace-path", str(trace_path)],
                cwd=directory, capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("Checkpoint 20%", process.stdout)
            self.assertIn("Final Answer:", process.stdout)
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            checks = [row for row in rows if row["record_type"] == "checkpoint"]
            self.assertEqual([row["decision"]["label"] for row in checks], ["support", "conflict", "insufficient"])
            self.assertEqual([row["intervene"] for row in checks], [False, True, True])
            self.assertTrue(all(row["reasoning_after"] is not None for row in checks))


if __name__ == "__main__":
    unittest.main()
