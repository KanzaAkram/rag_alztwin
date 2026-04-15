"""
api/patient_context.py
───────────────────────
Converts the incoming patient payload into:
  1. A natural-language search query for Azure AI Search
  2. A structured context block injected into the GPT-4o system prompt
"""

from models import AlzheimerStage, RecommendRequest

# ── Stage descriptions ────────────────────────────────────────────────────────

STAGE_DESCRIPTIONS = {
    AlzheimerStage.VERY_MILD: (
        "very mild dementia (CDR 0.5) — subjective memory complaints, "
        "slight forgetfulness, largely independent in daily activities"
    ),
    AlzheimerStage.MILD: (
        "mild dementia (CDR 1) — noticeable memory loss, difficulty with "
        "complex tasks, may need some assistance in daily activities"
    ),
    AlzheimerStage.MODERATE: (
        "moderate dementia (CDR 2) — significant memory loss, confusion, "
        "needs help with basic daily activities, behavioral symptoms common"
    ),
    AlzheimerStage.SEVERE: (
        "severe dementia (CDR 3) — profound memory loss, minimal verbal "
        "communication, fully dependent, late-stage complications"
    ),
}

# Stage → typical first-line treatment focus (used to enrich the search query)
STAGE_TREATMENT_FOCUS = {
    AlzheimerStage.VERY_MILD: "mild cognitive impairment early intervention prevention",
    AlzheimerStage.MILD:      "cholinesterase inhibitor donepezil rivastigmine galantamine mild dementia",
    AlzheimerStage.MODERATE:  "memantine cholinesterase inhibitor combination moderate dementia behavioral symptoms",
    AlzheimerStage.SEVERE:    "memantine severe dementia palliative care behavioral management",
}


# ── Heart rate alerts ─────────────────────────────────────────────────────────

def _heart_rate_flag(hr: int | None) -> str | None:
    if hr is None:
        return None
    if hr < 50:
        return f"bradycardia (HR {hr} bpm) — caution with cholinesterase inhibitors"
    if hr > 100:
        return f"tachycardia (HR {hr} bpm) — monitor cardiac status"
    return None


def _sleep_flag(hours: float | None, quality: str | None) -> str | None:
    if hours is not None and hours < 5:
        return f"severely disrupted sleep ({hours}h/night) — consider sleep-targeted interventions"
    if hours is not None and hours < 6:
        return f"reduced sleep ({hours}h/night) — may exacerbate cognitive symptoms"
    if quality in ("poor", "fair"):
        return f"poor sleep quality — evaluate for sleep disorders common in Alzheimer's"
    return None


def _bp_flag(systolic: int | None, diastolic: int | None) -> str | None:
    if systolic is not None and systolic > 140:
        return f"hypertension ({systolic}/{diastolic} mmHg) — check for drug interactions"
    if systolic is not None and systolic < 90:
        return f"hypotension ({systolic}/{diastolic} mmHg) — fall risk, caution with sedating agents"
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def build_search_query(req: RecommendRequest) -> str:
    """
    Returns a rich natural-language query to send to Azure AI Search.
    Combines stage, vitals-derived symptoms, and treatment focus keywords.
    """
    parts = [STAGE_TREATMENT_FOCUS[req.stage]]

    v = req.vitals
    if v.sleep_hours is not None and v.sleep_hours < 6:
        parts.append("sleep disorder treatment")
    if v.sleep_quality in ("poor", "fair"):
        parts.append("insomnia circadian rhythm")
    if v.heart_rate_bpm is not None:
        if v.heart_rate_bpm < 60:
            parts.append("bradycardia cardiac monitoring donepezil")
        if v.heart_rate_bpm > 90:
            parts.append("tachycardia cardiovascular risk")
    if req.comorbidities:
        parts.extend(req.comorbidities)

    return " ".join(parts)


def build_stage_filter(req: RecommendRequest) -> str | None:
    """
    Returns an OData filter string to bias search towards stage-relevant chunks.
    We do soft filtering (boost) rather than hard filter to avoid zero results.
    """
    stage_map = {
        AlzheimerStage.VERY_MILD: "very_mild",
        AlzheimerStage.MILD:      "mild",
        AlzheimerStage.MODERATE:  "moderate",
        AlzheimerStage.SEVERE:    "severe",
    }
    stage_val = stage_map[req.stage]
    # Return None to skip hard filtering — let semantic search handle relevance
    # Uncomment below to hard-filter (will miss cross-stage trials):
    # return f"search.ismatch('{stage_val}', 'alzheimer_stages')"
    return None


def build_system_prompt(req: RecommendRequest, retrieved_context: str) -> str:
    """
    Returns the full system prompt for GPT-4o including patient context
    and retrieved trial excerpts.
    """
    v = req.vitals
    stage_desc = STAGE_DESCRIPTIONS[req.stage]

    # Build vital sign flags
    flags = []
    for flag_fn, args in [
        (_heart_rate_flag, (v.heart_rate_bpm,)),
        (_sleep_flag, (v.sleep_hours, v.sleep_quality)),
        (_bp_flag, (v.systolic_bp, v.diastolic_bp)),
    ]:
        flag = flag_fn(*args)
        if flag:
            flags.append(f"  • {flag}")

    vitals_block = f"""
Patient Vitals:
  • Age: {req.age} years
  • Sleep: {f"{v.sleep_hours}h/night" if v.sleep_hours else "not recorded"} | Quality: {v.sleep_quality or "unknown"}
  • Heart rate: {f"{v.heart_rate_bpm} bpm" if v.heart_rate_bpm else "not recorded"}
  • Blood pressure: {f"{v.systolic_bp}/{v.diastolic_bp} mmHg" if v.systolic_bp else "not recorded"}
  • Weight: {f"{v.weight_kg} kg" if v.weight_kg else "not recorded"}
  • Notes: {v.other_notes or "none"}
""".strip()

    flags_block = (
        "Clinical Flags:\n" + "\n".join(flags)
        if flags
        else "Clinical Flags: none"
    )

    meds_block = (
        "Current Medications: " + ", ".join(req.current_medications)
        if req.current_medications
        else "Current Medications: none reported"
    )

    comorbidities_block = (
        "Comorbidities: " + ", ".join(req.comorbidities)
        if req.comorbidities
        else "Comorbidities: none reported"
    )

    return f"""You are a clinical decision-support AI specializing in Alzheimer's disease treatment.
Your role is to assist clinicians — not replace them.
Always base recommendations on the retrieved clinical trial evidence provided below.

════════════════════════════════════════
PATIENT PROFILE
════════════════════════════════════════
Alzheimer's Stage: {stage_desc}
{vitals_block}
{flags_block}
{meds_block}
{comorbidities_block}

════════════════════════════════════════
RETRIEVED CLINICAL TRIAL EVIDENCE
════════════════════════════════════════
{retrieved_context}

════════════════════════════════════════
INSTRUCTIONS
════════════════════════════════════════
Based ONLY on the retrieved evidence above:

1. Recommend the most appropriate treatment(s) for this patient's stage and vitals.
2. Specify the exact dosage, schedule, and titration if available in the evidence.
3. Provide a clear rationale citing the trials.
4. List any cautions specific to this patient's vitals (especially heart rate and sleep).
5. List vital signs or labs to monitor.
6. Add any lifestyle recommendations relevant to the vitals (sleep hygiene, cardiac).

Respond ONLY in the following JSON format — no markdown, no extra text:

{{
  "treatment": "<treatment name or class>",
  "dosage": "<dosage and schedule>",
  "rationale": "<evidence-based rationale, 2-4 sentences>",
  "cautions": ["<caution 1>", "<caution 2>"],
  "monitoring": ["<monitor 1>", "<monitor 2>"],
  "lifestyle_notes": ["<note 1>", "<note 2>"]
}}"""


def build_user_message(req: RecommendRequest) -> str:
    return (
        f"Generate a treatment recommendation for the patient described above "
        f"at {req.stage.value.replace('_', ' ')} Alzheimer's stage. "
        f"Focus on evidence from the retrieved trials."
    )