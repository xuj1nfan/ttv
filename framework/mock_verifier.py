"""Deterministic fixtures for exercising orchestration, not legal inference."""

from copy import deepcopy
from typing import Mapping

from .schema import Claim, Decision
from .verifier import Verifier


class MockVerifier(Verifier):
    """Look up probabilities by exact claim text; unknown claims are insufficient.

    Fixtures deliberately live outside generator metadata, so tests do not pass
    a purported correct label from the generator to the verifier.
    """

    def __init__(self, responses: Mapping[str, Mapping[str, float]] | None = None) -> None:
        self.responses = deepcopy(dict(responses or {}))

    def verify(self, case_text: str, claim: Claim) -> Decision:
        """Return a fresh fixture judgment with optional evidence left empty."""
        probabilities = dict(self.responses.get(
            claim.text, {"support": 0.05, "conflict": 0.05, "insufficient": 0.90}
        ))
        label = max(probabilities, key=probabilities.__getitem__)
        return Decision(
            label=label, confidence=probabilities[label], probabilities=probabilities,
            metadata={"verifier": "mock", "fixture": True},
        )
