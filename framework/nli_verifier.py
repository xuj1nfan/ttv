"""Three-way NLI with explicit label mapping and no silent input truncation."""

from __future__ import annotations

from math import exp, isfinite
from time import perf_counter

from .schema import Claim, Decision
from .verifier import Verifier


NLI_LABELS = {"entailment": "support", "contradiction": "conflict", "neutral": "insufficient"}


def label_mapping(id2label: dict, override: dict | None = None) -> dict[int, str]:
    """An explicit override is mandatory for models with generic LABEL_0 names."""
    mapping = {}
    for key, label in (override if override is not None else id2label).items():
        name = str(label).lower()
        mapping[int(key)] = NLI_LABELS.get(name, name)
    if set(mapping) != {0, 1, 2} or set(mapping.values()) != set(NLI_LABELS.values()):
        raise ValueError("NLI must map exactly three output indices to support/conflict/insufficient; inspect id2label or supply label_map")
    return mapping


def probabilities_from_logits(logits: list[float], mapping: dict[int, str], temperature: float = 1.0) -> dict[str, float]:
    if len(logits) != 3 or not all(isfinite(value) for value in logits):
        raise ValueError("Expected three finite NLI logits")
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("NLI temperature must be finite and positive")
    shifted = [(value - max(logits)) / temperature for value in logits]
    weights = [exp(value) for value in shifted]
    return {mapping[index]: weight / sum(weights) for index, weight in enumerate(weights)}


class PairTooLong(ValueError):
    def __init__(self, tokens: int, limit: int) -> None:
        super().__init__(f"NLI pair has {tokens} tokens, exceeding {limit}")
        self.tokens, self.limit = tokens, limit


class HuggingFaceNLIBackend:
    """Optional torch/transformers dependencies are loaded only for real NLI."""

    def __init__(
        self, model_name: str, *, device: str = "cpu", max_length: int = 512,
        revision: str | None = None, local_files_only: bool = False,
        label_map: dict | None = None,
    ) -> None:
        if type(max_length) is not int or max_length < 4:
            raise ValueError("NLI max_length must be an integer >= 4")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Real NLI requires torch and transformers; see requirements-nli.txt") from error
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NLI requested CUDA, but this process cannot see a CUDA device")
        options = {"revision": revision, "local_files_only": local_files_only}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **options)
        self.model.to(self.device).eval()
        self.mapping = label_mapping(self.model.config.id2label, label_map)
        self.max_length = min(max_length, self.tokenizer.model_max_length)
        self.metadata = {
            "model": model_name, "revision": revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": str(self.device), "dtype": str(self.model.dtype),
            "max_length": self.max_length, "id2label": self.model.config.id2label,
        }

    def predict(self, premise: str, hypothesis: str) -> tuple[list[float], dict]:
        encoded = self.tokenizer(premise, hypothesis, truncation=False, return_tensors="pt")
        tokens = encoded["input_ids"].shape[-1]
        if tokens > self.max_length:
            raise PairTooLong(tokens, self.max_length)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0].float().cpu().tolist()
        return logits, {"input_tokens": tokens, "truncated": False}


class NLIVerifier(Verifier):
    def __init__(
        self, model_name: str = "joeddav/xlm-roberta-large-xnli", *,
        temperature: float = 1.0, backend=None, **backend_options,
    ) -> None:
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("NLI temperature must be finite and positive")
        self.backend = backend if backend is not None else HuggingFaceNLIBackend(model_name, **backend_options)
        self.temperature = temperature

    def verify(self, case_text: str, claim: Claim) -> Decision:
        started = perf_counter()
        metadata = {**self.backend.metadata, "verifier": "nli", "temperature": self.temperature}
        try:
            logits, details = self.backend.predict(case_text, claim.text)
        except PairTooLong as error:
            # This extensible label is a control status, not fabricated NLI
            # confidence. It must be excluded from NLI classification metrics.
            return Decision("unverifiable", 1.0, {"unverifiable": 1.0}, metadata={
                **metadata, "evaluated": False, "abstained": True, "reason": "pair_too_long",
                "input_tokens": error.tokens, "max_length": error.limit,
                "elapsed_seconds": perf_counter() - started, "model_probabilities": None,
            })
        probabilities = probabilities_from_logits(logits, self.backend.mapping, self.temperature)
        label = max(probabilities, key=probabilities.__getitem__)
        return Decision(label, probabilities[label], probabilities, metadata={
            **metadata, **details, "evaluated": True, "abstained": False,
            "logits": logits, "elapsed_seconds": perf_counter() - started,
        })


class UncheckedVerifier(Verifier):
    """Control status for baseline/self-review; does not invoke an NLI model."""

    def verify(self, case_text: str, claim: Claim) -> Decision:
        return Decision("not_evaluated", 1.0, {"not_evaluated": 1.0}, metadata={
            "verifier": "none", "evaluated": False, "model_probabilities": None,
        })
