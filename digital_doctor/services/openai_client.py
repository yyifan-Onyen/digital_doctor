from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Iterator, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass(frozen=True)
class ModelCallRecord:
    group: str
    requested_model: str
    served_model: str
    json_mode: bool
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    latency_seconds: float
    error_type: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_MODEL_OVERRIDE: ContextVar[Optional[str]] = ContextVar(
    "digital_doctor_model_override", default=None
)
_REASONING_OVERRIDE: ContextVar[Optional[str]] = ContextVar(
    "digital_doctor_reasoning_override", default=None
)
_CALL_CAPTURE: ContextVar[Optional[tuple[str, List[ModelCallRecord]]]] = ContextVar(
    "digital_doctor_call_capture", default=None
)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to call the main GPT model.")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


@contextmanager
def model_config(
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Iterator[None]:
    """Temporarily override model settings for all calls in the current context."""
    model_token = _MODEL_OVERRIDE.set(model)
    reasoning_token = _REASONING_OVERRIDE.set(reasoning_effort)
    try:
        yield
    finally:
        _REASONING_OVERRIDE.reset(reasoning_token)
        _MODEL_OVERRIDE.reset(model_token)


@contextmanager
def capture_model_calls(group: str) -> Iterator[List[ModelCallRecord]]:
    """Collect token and latency metadata without changing callers' return values."""
    records: List[ModelCallRecord] = []
    token = _CALL_CAPTURE.set((group, records))
    try:
        yield records
    finally:
        _CALL_CAPTURE.reset(token)


def _usage_value(value: object, name: str) -> int:
    raw = getattr(value, name, 0) if value is not None else 0
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def _append_call_record(
    *,
    requested_model: str,
    served_model: str,
    json_mode: bool,
    usage: object,
    latency_seconds: float,
    error_type: str = "",
) -> None:
    capture = _CALL_CAPTURE.get()
    if capture is None:
        return
    group, records = capture
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    records.append(
        ModelCallRecord(
            group=group,
            requested_model=requested_model,
            served_model=served_model,
            json_mode=json_mode,
            input_tokens=_usage_value(usage, "prompt_tokens"),
            cached_input_tokens=_usage_value(prompt_details, "cached_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens"),
            reasoning_tokens=_usage_value(completion_details, "reasoning_tokens"),
            total_tokens=_usage_value(usage, "total_tokens"),
            latency_seconds=round(latency_seconds, 6),
            error_type=error_type,
        )
    )


def call_model(
    prompt_text: str,
    json_mode: bool = False,
    model: Optional[str] = None,
    *,
    reasoning_effort: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
    response_schema: Optional[Dict[str, object]] = None,
    response_schema_name: str = "structured_response",
) -> str:
    selected_model = model or _MODEL_OVERRIDE.get() or DEFAULT_MODEL
    selected_reasoning = reasoning_effort or _REASONING_OVERRIDE.get()
    payload = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    if selected_reasoning:
        payload["reasoning_effort"] = selected_reasoning
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema_name,
                "strict": True,
                "schema": response_schema,
            },
        }
    elif json_mode:
        payload["response_format"] = {"type": "json_object"}

    started = time.perf_counter()
    try:
        response = _get_client().chat.completions.create(**payload)
    except Exception as exc:
        _append_call_record(
            requested_model=selected_model,
            served_model="",
            json_mode=json_mode or response_schema is not None,
            usage=None,
            latency_seconds=time.perf_counter() - started,
            error_type=type(exc).__name__,
        )
        raise
    _append_call_record(
        requested_model=selected_model,
        served_model=str(getattr(response, "model", "") or selected_model),
        json_mode=json_mode or response_schema is not None,
        usage=getattr(response, "usage", None),
        latency_seconds=time.perf_counter() - started,
    )
    return response.choices[0].message.content or ""
