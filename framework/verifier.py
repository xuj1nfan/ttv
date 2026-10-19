"""Model-independent local-claim verification contract."""

from abc import ABC, abstractmethod

from .schema import Claim, Decision


class Verifier(ABC):
    """Adapters map a model's native labels into the shared decision schema."""

    @abstractmethod
    def verify(self, case_text: str, claim: Claim) -> Decision:
        """Judge only the local claim against the original case material."""
        raise NotImplementedError
