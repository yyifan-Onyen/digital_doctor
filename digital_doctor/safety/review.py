"""Compatibility facade for the default OCD/ERP response reviewer."""

from __future__ import annotations

from typing import Dict, Optional

from ..services.openai_client import call_model
from ..skills.ocd_erp.review import SafetyVerdict, TraceWriter
from ..skills.ocd_erp.review import review_final_response as _review_final_response


def review_final_response(
    user_text: str,
    draft_reply: str,
    history: str = "",
    formulation_context: str = "",
    treatment_allowed: bool = True,
    mood_assessment: Optional[Dict[str, object]] = None,
    trace: Optional[TraceWriter] = None,
) -> SafetyVerdict:
    return _review_final_response(
        user_text=user_text,
        draft_reply=draft_reply,
        history=history,
        formulation_context=formulation_context,
        treatment_allowed=treatment_allowed,
        mood_assessment=mood_assessment,
        trace=trace,
        model_call=call_model,
    )


__all__ = ["SafetyVerdict", "review_final_response"]
