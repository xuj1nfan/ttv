"""Verification timing, independent of model and intervention policy."""

from abc import ABC, abstractmethod
from math import ceil, isfinite
from typing import Sequence

from .schema import Checkpoint, ReasoningState


class Scheduler(ABC):
    """Select intermediate checkpoints from the current generator state."""

    @abstractmethod
    def get_checkpoint(self, state: ReasoningState) -> Checkpoint | None:
        """Return a checkpoint or None; implementations may inspect uncertainty."""
        raise NotImplementedError


class FixedCheckpointScheduler(Scheduler):
    """Trigger at fractional progress or explicit step numbers.

    Fractions use ceil(fraction * total_steps). Supply a planned total_steps or
    publish it in state.metadata. Coincident targets are combined into one check.
    Finished states are excluded: verification must allow continued reasoning.
    This scheduler is stateless and reusable across cases.
    """

    def __init__(
        self,
        checkpoints: Sequence[float] | None = None,
        *,
        step_checkpoints: Sequence[int] | None = None,
        total_steps: int | None = None,
    ) -> None:
        if checkpoints is not None and step_checkpoints is not None:
            raise ValueError("Choose fractional checkpoints OR step_checkpoints")
        self.fractions = tuple(
            (0.2, 0.5, 0.8) if checkpoints is None and step_checkpoints is None
            else checkpoints or ()
        )
        self.step_checkpoints = tuple(step_checkpoints or ())
        if any(not isfinite(x) or not 0 < x < 1 for x in self.fractions):
            raise ValueError("Fractional checkpoints must be strictly between 0 and 1")
        if any(type(x) is not int or x < 1 for x in self.step_checkpoints):
            raise ValueError("Step checkpoints must be positive integers")
        if total_steps is not None and (type(total_steps) is not int or total_steps < 1):
            raise ValueError("total_steps must be a positive integer")
        self.total_steps = total_steps

    def get_checkpoint(self, state: ReasoningState) -> Checkpoint | None:
        """Select one check at the current step, merging rounded collisions."""
        if state.finished or state.current_step < 1:
            return None
        names: list[str] = []
        if self.fractions:
            total = self.total_steps or state.metadata.get("total_steps")
            if type(total) is not int or total < 1:
                raise ValueError("Fractional scheduling requires a positive total_steps")
            names = [f"{x:.0%}" for x in sorted(set(self.fractions))
                     if ceil(x * total) == state.current_step]
            progress = state.current_step / total
        else:
            if state.current_step in self.step_checkpoints:
                names = [f"step-{state.current_step}"]
            progress = None
        if not names:
            return None
        return Checkpoint(
            name=", ".join(names), step=state.current_step, progress=progress,
            metadata={"scheduler": "fixed", "targets": names},
        )
