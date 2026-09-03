"""Replaceable generation backends used by the execution harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

from .contracts import GenerationSpec


class ModelAdapter(Protocol):
    adapter_id: str

    def generate(self, spec: GenerationSpec) -> str:
        ...


@dataclass
class CallableModelAdapter:
    """Adapter for the existing ``call_model`` function and test doubles."""

    model_call: Callable[..., str]
    adapter_id: str = "prompt-model"

    def generate(self, spec: GenerationSpec) -> str:
        kwargs: Dict[str, object] = {"json_mode": spec.json_mode}
        model = spec.metadata.get("model")
        if model:
            kwargs["model"] = str(model)
        reasoning_effort = spec.metadata.get("reasoning_effort")
        if reasoning_effort:
            kwargs["reasoning_effort"] = str(reasoning_effort)
        return self.model_call(spec.prompt, **kwargs)


@dataclass
class HttpModelAdapter:
    """Adapter for a local/remote learned model exposing ``generate(prompt)``."""

    client: object
    adapter_id: str = "http-model"
    max_new_tokens: int = 256
    temperature: float = 0.7

    def generate(self, spec: GenerationSpec) -> str:
        generate = getattr(self.client, "generate")
        return str(
            generate(
                spec.prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        )


@dataclass
class SFTModelAdapter(HttpModelAdapter):
    adapter_id: str = "sft-model"


@dataclass
class OPSDModelAdapter(HttpModelAdapter):
    adapter_id: str = "opsd-model"


__all__ = [
    "CallableModelAdapter",
    "HttpModelAdapter",
    "ModelAdapter",
    "OPSDModelAdapter",
    "SFTModelAdapter",
]
