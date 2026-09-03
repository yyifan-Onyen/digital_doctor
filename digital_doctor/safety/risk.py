"""Compatibility facade for the default OCD/ERP skill risk assessor."""

from __future__ import annotations

from typing import Optional

from ..services.openai_client import call_model
from ..skills.ocd_erp.risk import (
    MoodAssessment,
    TraceWriter,
    critical_risk_response,
    stopped_conversation_response,
)
from ..skills.ocd_erp.risk import assess_patient_state as _assess_patient_state


def assess_patient_state(
    user_text: str,
    history: str = "",
    trace: Optional[TraceWriter] = None,
) -> MoodAssessment:
    return _assess_patient_state(
        user_text,
        history,
        trace,
        model_call=call_model,
    )


__all__ = [
    "MoodAssessment",
    "assess_patient_state",
    "critical_risk_response",
    "stopped_conversation_response",
]
