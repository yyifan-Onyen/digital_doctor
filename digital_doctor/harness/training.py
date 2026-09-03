"""Convert harness turn traces into SFT or skill-conditioned OPSD records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Literal, Mapping


ExportFormat = Literal["sft", "opsd"]


def iter_distillation_records(trace_path: str | Path) -> Iterator[Dict[str, object]]:
    with Path(trace_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on trace line {line_number}") from exc
            if event.get("event") != "distillation_record":
                continue
            yield {
                key: value
                for key, value in event.items()
                if key not in {"timestamp", "event", "session_id", "episode_id"}
            }


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def to_sft_record(record: Mapping[str, object]) -> Dict[str, object]:
    student = _as_mapping(record.get("student_input"))
    target = _as_mapping(record.get("teacher_target"))
    history = str(student.get("history", "")).strip()
    patient_message = str(student.get("patient_message", "")).strip()
    user_content = (
        f"Recent dialogue:\n{history}\n\nLatest patient message:\n{patient_message}"
        if history
        else patient_message
    )
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(target.get("response", ""))},
        ],
        "clinical_supervision": {
            "identity": record.get("identity", {}),
            "action_plan": _as_mapping(record.get("privileged_skill_context")).get(
                "action_plan", {}
            ),
            "state_delta": _as_mapping(record.get("privileged_skill_context")).get(
                "state_delta", {}
            ),
            "safety": target.get("safety", {}),
        },
    }


def to_opsd_record(record: Mapping[str, object]) -> Dict[str, object]:
    return {
        "identity": record.get("identity", {}),
        "student_input": record.get("student_input", {}),
        "privileged_teacher_context": record.get("privileged_skill_context", {}),
        "teacher_target": record.get("teacher_target", {}),
    }


def export_distillation_records(
    trace_path: str | Path,
    output_path: str | Path,
    output_format: ExportFormat = "opsd",
) -> int:
    if output_format not in {"sft", "opsd"}:
        raise ValueError(f"Unsupported export format: {output_format}")
    converter = to_sft_record if output_format == "sft" else to_opsd_record
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in iter_distillation_records(trace_path):
            handle.write(json.dumps(converter(record), ensure_ascii=False) + "\n")
            count += 1
    return count


__all__ = [
    "export_distillation_records",
    "iter_distillation_records",
    "to_opsd_record",
    "to_sft_record",
]
