"""Shared data contracts; model-specific extensions belong in metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isclose, isfinite
from typing import Any

Metadata = dict[str, Any]


@dataclass(frozen=True)
class Claim:
    """A local proposition extracted from a particular reasoning step."""

    text: str
    source_step: int
    claim_type: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip() or self.source_step < 1:
            raise ValueError("A claim requires text and a positive source_step")


@dataclass(frozen=True)
class Decision:
    """Verifier judgment; confidence is the probability of the chosen label.

    Probability keys are extensible strings. The initial labels are support,
    conflict and insufficient. Extra scores, calibration details and alternative
    decision formats can be stored in metadata without changing the runner.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]
    evidence: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probabilities or self.label not in self.probabilities:
            raise ValueError("Decision label must have a probability")
        for label, value in self.probabilities.items():
            if not label or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Probabilities require nonempty labels and finite values in [0, 1]")
        if not isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to 1")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1 or not isclose(
            self.confidence, self.probabilities[self.label], abs_tol=1e-6
        ):
            raise ValueError("Confidence must match the selected label probability")


@dataclass(frozen=True)
class Intervention:
    """A policy action, separate from the verifier's judgment."""

    intervene: bool
    message: str | None
    reason: str
    decision: Decision

    def __post_init__(self) -> None:
        if self.intervene and (not self.message or not self.message.strip()):
            raise ValueError("An intervention requires a feedback message")


@dataclass
class ReasoningState:
    """Mutable state owned by a generator; steps are observable analysis text."""

    case_text: str
    steps: list[str] = field(default_factory=list)
    current_step: int = 0
    finished: bool = False
    answer: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Checkpoint:
    """A scheduler-selected verification point after a reasoning step."""

    name: str
    step: int
    progress: float | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass
class TraceRecord:
    """One checkpoint, completed after the next step or at termination.

    reasoning_after is the full snapshot after exactly one subsequent generator
    step, including when the policy abstains. A null value explicitly means no
    continuation was observed. Neither field is a causal correction label.
    """

    case_id: str
    checkpoint: Checkpoint
    reasoning_before: list[str]
    claim: Claim
    decision: Decision
    intervene: bool
    feedback: str | None
    intervention_reason: str
    reasoning_after: list[str] | None = None
    schema_version: str = "1.0"
    record_type: str = "checkpoint"
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready snapshot (metadata must also be JSON-compatible)."""
        return asdict(self)


@dataclass
class RunResult:
    """Final state and ordered checkpoint trace for a single case."""

    case_id: str
    state: ReasoningState
    trace: list[TraceRecord]

    @property
    def answer(self) -> str | None:
        """Return the generator's final answer."""
        return self.state.answer
