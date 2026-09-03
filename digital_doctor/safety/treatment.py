"""Compatibility facade for the default OCD/ERP treatment policies."""

from ..skills.ocd_erp.treatment import (
    TreatmentReadiness,
    assess_treatment_readiness,
    contains_treatment_advice,
    enforce_high_risk_treatment_limit,
    enforce_treatment_buffer,
    treatment_policy_block,
)

__all__ = [
    "TreatmentReadiness",
    "assess_treatment_readiness",
    "contains_treatment_advice",
    "enforce_high_risk_treatment_limit",
    "enforce_treatment_buffer",
    "treatment_policy_block",
]
