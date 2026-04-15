"""
api/models.py
─────────────
Request and response schemas for the recommendation API.
Pydantic v2 — used for validation in the Azure Function handler.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class AlzheimerStage(str, Enum):
    VERY_MILD  = "very_mild"    # CDR 0.5 — subjective / very mild
    MILD       = "mild"         # CDR 1   — mild dementia
    MODERATE   = "moderate"     # CDR 2   — moderate dementia
    SEVERE     = "severe"       # CDR 3   — severe dementia


class SleepQuality(str, Enum):
    GOOD    = "good"
    FAIR    = "fair"
    POOR    = "poor"
    UNKNOWN = "unknown"


# ── Request ───────────────────────────────────────────────────────────────────

class Vitals(BaseModel):
    sleep_hours:    Optional[float]        = Field(None, ge=0, le=24,  description="Average nightly sleep in hours")
    sleep_quality:  Optional[SleepQuality] = Field(None,               description="Subjective sleep quality")
    heart_rate_bpm: Optional[int]          = Field(None, ge=20, le=300,description="Resting heart rate")
    systolic_bp:    Optional[int]          = Field(None, ge=60, le=300,description="Systolic blood pressure mmHg")
    diastolic_bp:   Optional[int]          = Field(None, ge=40, le=200,description="Diastolic blood pressure mmHg")
    weight_kg:      Optional[float]        = Field(None, ge=20, le=300)
    other_notes:    Optional[str]          = Field(None, max_length=500)


class RecommendRequest(BaseModel):
    patient_id:         str              = Field(..., min_length=1, max_length=100)
    stage:              AlzheimerStage
    age:                int              = Field(..., ge=18, le=120)
    vitals:             Vitals           = Field(default_factory=Vitals)
    current_medications:list[str]        = Field(default_factory=list, description="Current medications for contraindication awareness")
    comorbidities:      list[str]        = Field(default_factory=list, description="e.g. ['hypertension', 'diabetes']")
    top_k:              int              = Field(default=5, ge=1, le=10, description="Number of trial sources to retrieve")

    @field_validator("current_medications", "comorbidities", mode="before")
    @classmethod
    def strip_list_items(cls, v):
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return v


# ── Response ──────────────────────────────────────────────────────────────────

class TrialSource(BaseModel):
    doc_id:          str
    title:           str
    source:          str
    source_url:      str
    relevance_score: float
    phases:          str


class TreatmentRecommendation(BaseModel):
    treatment:       str   = Field(..., description="Treatment name or class")
    dosage:          str   = Field(..., description="Recommended dosage and schedule")
    rationale:       str   = Field(..., description="Evidence-based rationale from retrieved trials")
    cautions:        list[str] = Field(default_factory=list, description="Drug interactions and monitoring notes")
    monitoring:      list[str] = Field(default_factory=list, description="Vital signs / labs to monitor")
    lifestyle_notes: list[str] = Field(default_factory=list, description="Sleep, exercise, diet notes based on vitals")


class RecommendResponse(BaseModel):
    patient_id:      str
    stage:           AlzheimerStage
    recommendation:  TreatmentRecommendation
    sources:         list[TrialSource]
    disclaimer:      str = (
        "This recommendation is AI-generated from clinical trial data and is intended "
        "to assist clinicians only. It does not replace professional medical judgment. "
        "Always verify against current clinical guidelines and the patient's full history."
    )


class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None
    code:    int