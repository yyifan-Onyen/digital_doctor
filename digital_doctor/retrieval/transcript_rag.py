from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..paths import DEFAULT_SEGMENT_CACHE_PATH, resolve_repo_path
from ..prompt import build_summary_prompt
from ..services.openai_client import call_model
from ..core.text_utils import extract_final


@dataclass
class Summary:
    core: str
    context: str
    signals: str


@dataclass
class Segment:
    raw_text: str
    summary: Summary


def parse_summary(raw: str) -> Summary:
    cleaned = extract_final(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        obj = {}
    return Summary(
        core=str(obj.get("core", "")).strip(),
        context=str(obj.get("context", "")).strip(),
        signals=str(obj.get("signals", "")).strip(),
    )


def summarize_text(text: str) -> Summary:
    raw = call_model(build_summary_prompt(text), json_mode=True)
    return parse_summary(raw)


def summary_to_text(summary: Summary) -> str:
    return " ".join(part for part in [summary.core, summary.context, summary.signals] if part).lower()


def is_treatment_relevant(summary: Summary) -> bool:
    return bool(summary.core.strip())


def _load_segment_cache(cache_path: str, max_segments: int) -> List[Segment]:
    segments: List[Segment] = []
    resolved = str(resolve_repo_path(cache_path))
    if not os.path.exists(resolved):
        return segments
    with open(resolved, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            summary = Summary(
                core=str(item.get("core", "")).strip(),
                context=str(item.get("context", "")).strip(),
                signals=str(item.get("signals", "")).strip(),
            )
            if not is_treatment_relevant(summary):
                continue
            raw_text = str(item.get("raw_text", "")).strip()
            if not raw_text:
                continue
            segments.append(Segment(raw_text=raw_text, summary=summary))
            if len(segments) >= max_segments:
                break
    return segments


def _write_segment_cache(cache_path: str, segments: Sequence[Segment]) -> None:
    resolved = str(resolve_repo_path(cache_path))
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        for segment in segments:
            record = {
                "raw_text": segment.raw_text,
                "core": segment.summary.core,
                "context": segment.summary.context,
                "signals": segment.summary.signals,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_segments(
    path: str,
    max_segments: int = 120,
    cache_path: str = DEFAULT_SEGMENT_CACHE_PATH,
) -> List[Segment]:
    resolved_path = str(resolve_repo_path(path))
    if not os.path.exists(resolved_path):
        return []
    if cache_path:
        cached = _load_segment_cache(cache_path, max_segments)
        if cached:
            return cached
    with open(resolved_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    segments: List[Segment] = []
    for transcript in data.get("Transcripts", []):
        for raw_segment in transcript.get("segments_deid", []):
            summary = summarize_text(raw_segment)
            if not is_treatment_relevant(summary):
                continue
            segments.append(Segment(raw_text=raw_segment, summary=summary))
            if len(segments) >= max_segments:
                if cache_path:
                    _write_segment_cache(cache_path, segments)
                return segments
    if cache_path:
        _write_segment_cache(cache_path, segments)
    return segments


def pick_refs(query: str, segments: Sequence[Segment], top_k: int = 3) -> List[str]:
    user_summary = summarize_text(query)
    user_text = summary_to_text(user_summary)
    tokens = set(user_text.split())
    scored: List[Tuple[int, str]] = []
    for segment in segments:
        lower = summary_to_text(segment.summary)
        score = sum(1 for token in tokens if token and token in lower)
        if score > 0:
            scored.append((score, segment.raw_text[:400]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [segment for _, segment in scored[:top_k]]
