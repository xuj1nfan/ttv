"""Convert probabilistic judgments into controlled feedback."""

from abc import ABC, abstractmethod
from math import isfinite

from .schema import Decision, Intervention


class InterventionPolicy(ABC):
    """Choose whether and how a verifier result should affect generation."""

    @abstractmethod
    def decide(self, decision: Decision) -> Intervention:
        """Return an action without calling a verifier or generator."""
        raise NotImplementedError


class ThresholdInterventionPolicy(InterventionPolicy):
    """Intervene on selected conflict/insufficient judgments at inclusive thresholds.

    Support and unknown future labels abstain. These probability thresholds do
    not imply that an underlying model has actually been calibrated.
    """

    def __init__(
        self, conflict_threshold: float = 0.80, insufficient_threshold: float = 0.85,
    ) -> None:
        self.thresholds = {
            "conflict": conflict_threshold, "insufficient": insufficient_threshold,
        }
        if any(not isfinite(x) or not 0 <= x <= 1 for x in self.thresholds.values()):
            raise ValueError("Intervention thresholds must be finite values in [0, 1]")
        self.messages = {
            "conflict": "当前命题与案件材料存在明显冲突，请重新检查这一判断，并优先依据原始案件事实修正后续分析。",
            "insufficient": "当前材料不足以支持这一命题，请不要将其作为既定事实，并在后续分析中保持该问题未决。",
        }

    def decide(self, decision: Decision) -> Intervention:
        """Apply the threshold for the selected label; otherwise abstain."""
        threshold = self.thresholds.get(decision.label)
        intervene = threshold is not None and decision.confidence >= threshold
        reason = (
            f"{decision.label}: {decision.confidence:.4f} >= {threshold:.4f}"
            if intervene else
            f"{decision.label}: below threshold or no intervention rule"
        )
        return Intervention(
            intervene=intervene,
            message=self.messages[decision.label] if intervene else None,
            reason=reason, decision=decision,
        )
