"""Incremental generator interface and a deterministic scripted implementation."""

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from .schema import Claim, ReasoningState


class Generator(ABC):
    """Produce one observable analysis step at a time, accepting feedback in between.

    Adapters mutate the supplied state in place. step increments current_step by
    one, appends one text step, and eventually sets finished and answer. feedback
    queues context for subsequent generation; it must not advance the step.
    """

    @abstractmethod
    def start(self, case_text: str) -> ReasoningState:
        """Initialize independent state for one case."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: ReasoningState) -> None:
        """Append one analysis step, consuming any pending feedback."""
        raise NotImplementedError

    @abstractmethod
    def extract_claim(self, state: ReasoningState) -> Claim:
        """Extract a local claim from the latest analysis, before the final answer."""
        raise NotImplementedError

    @abstractmethod
    def feedback(self, state: ReasoningState, message: str) -> None:
        """Make a policy message available to subsequent generation."""
        raise NotImplementedError


class MockGenerator(Generator):
    """A transparent fixture, with optional scripted responses after feedback.

    Each step text is itself the local claim. A correction is keyed by the step
    receiving feedback and is emitted at the next step only if feedback arrived.
    Answers are scripted, so this mock cannot measure real correction efficacy.
    """

    def __init__(
        self,
        steps: Sequence[str],
        answer: str,
        *,
        feedback_responses: Mapping[int, str] | None = None,
        answer_after_feedback: str | None = None,
    ) -> None:
        if not steps or any(not text.strip() for text in steps):
            raise ValueError("MockGenerator requires nonempty reasoning steps")
        self.script = tuple(steps)
        self.final_answer = answer
        self.feedback_responses = dict(feedback_responses or {})
        self.answer_after_feedback = answer_after_feedback

    def start(self, case_text: str) -> ReasoningState:
        """Create fresh state with a known planned reasoning length."""
        return ReasoningState(case_text, metadata={
            "total_steps": len(self.script), "feedback_history": [],
        })

    def step(self, state: ReasoningState) -> None:
        """Emit the next script entry or a queued fixture correction."""
        if state.finished:
            raise ValueError("Cannot advance a finished generator")
        pending = state.metadata.pop("pending_feedback", None)
        text = self.script[state.current_step]
        if pending is not None:
            text = self.feedback_responses.get(
                state.current_step, f"收到验证反馈：{pending} 后续分析：{text}",
            )
        state.steps.append(text)
        state.current_step += 1
        if state.current_step == len(self.script):
            state.finished = True
            state.answer = (
                self.answer_after_feedback
                if state.metadata["feedback_history"] and self.answer_after_feedback is not None
                else self.final_answer
            )

    def extract_claim(self, state: ReasoningState) -> Claim:
        """Treat the latest fixture step as a fact proposition."""
        return Claim(state.steps[-1], state.current_step, claim_type="fact")

    def feedback(self, state: ReasoningState, message: str) -> None:
        """Queue feedback for the next step and retain its delivery history."""
        state.metadata["pending_feedback"] = message
        state.metadata["feedback_history"].append({
            "step": state.current_step, "message": message,
        })
