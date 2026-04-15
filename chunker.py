"""
ingestion/chunker.py
─────────────────────
Splits raw trial/abstract documents into overlapping chunks
suitable for embedding and vector search.

Strategy:
  - Combine title + summary + description into one text blob
  - Split on ~400 tokens with 80-token overlap
  - Each chunk keeps full metadata so retrieval is self-contained
"""

import re
from typing import Generator

import tiktoken

ENCODER      = tiktoken.get_encoding("cl100k_base")  # same as text-embedding-3-large
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 80

# Alzheimer stage keywords — used to tag chunks for stage-aware retrieval
STAGE_KEYWORDS = {
    "very_mild":  ["very mild", "preclinical", "subjective cognitive", "MCI", "mild cognitive impairment"],
    "mild":       ["mild dementia", "early stage", "mild alzheimer", "mild AD"],
    "moderate":   ["moderate dementia", "moderate alzheimer", "moderate AD", "moderate stage"],
    "severe":     ["severe dementia", "severe alzheimer", "late stage", "advanced AD"],
}

# Symptom keywords for secondary tagging
SYMPTOM_KEYWORDS = {
    "sleep":       ["sleep", "insomnia", "circadian", "REM", "sleep disturbance", "sleep disorder"],
    "cardiac":     ["heart rate", "cardiac", "cardiovascular", "bradycardia", "tachycardia", "blood pressure"],
    "cognitive":   ["cognitive", "memory", "cognition", "MMSE", "ADAS-cog"],
    "behavioral":  ["agitation", "aggression", "depression", "anxiety", "behavioral"],
    "mobility":    ["gait", "falls", "mobility", "balance", "motor"],
}

BIOMARKER_KEYWORDS = {
    "amyloid": ["amyloid", "abeta", "aβ", "beta amyloid", "plaque"],
    "tau": ["tau", "p-tau", "phosphorylated tau", "neurofibrillary"],
    "neuroinflammation": ["neuroinflammation", "microglia", "inflammatory cytokine"],
    "neurodegeneration": ["neurodegeneration", "atrophy", "hippocampal volume"],
}

OUTCOME_KEYWORDS = {
    "cognitive": ["mmse", "adas-cog", "cognitive", "memory", "recall"],
    "functional": ["activities of daily living", "adl", "functional status", "iadl"],
    "behavioral": ["agitation", "depression", "anxiety", "behavioral"],
    "sleep": ["sleep", "insomnia", "sleep quality", "circadian"],
    "safety": ["adverse event", "side effect", "safety", "tolerability"],
    "cardiovascular": ["heart rate", "blood pressure", "cardiac", "bradycardia", "tachycardia"],
}

EVIDENCE_KEYWORDS = {
    "randomized_controlled_trial": ["randomized", "randomised", "controlled trial", "rct"],
    "double_blind": ["double blind", "double-blind"],
    "placebo_controlled": ["placebo"],
    "open_label": ["open-label", "open label"],
    "meta_analysis": ["meta-analysis", "systematic review"],
}

INTERVENTION_CLASS_KEYWORDS = {
    "cholinesterase_inhibitor": ["donepezil", "rivastigmine", "galantamine", "cholinesterase inhibitor"],
    "nmda_antagonist": ["memantine", "nmda antagonist"],
    "anti_amyloid": ["aducanumab", "lecanemab", "donanemab", "anti-amyloid", "monoclonal antibody"],
    "antidepressant": ["citalopram", "mirtazapine", "ssri", "antidepressant"],
    "sleep_targeted": ["melatonin", "trazodone", "sleep intervention", "sleep hygiene"],
    "non_pharmacologic": ["exercise", "cognitive training", "diet", "caregiver intervention"],
}

RISK_KEYWORDS = {
    "bradycardia_risk": ["bradycardia", "low heart rate"],
    "hypotension_risk": ["hypotension", "orthostatic"],
    "sedation_risk": ["sedation", "somnolence", "drowsiness"],
    "fall_risk": ["fall", "falls"],
    "bleeding_risk": ["bleeding", "hemorrhage", "haemorrhage"],
}


def _tokenize(text: str) -> list[int]:
    return ENCODER.encode(text)


def _detokenize(tokens: list[int]) -> str:
    return ENCODER.decode(tokens)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def _build_full_text(doc: dict) -> str:
    """Concatenate all meaningful text fields from a document."""
    parts = []

    if doc.get("title"):
        parts.append(f"Title: {doc['title']}")

    if doc.get("phases"):
        parts.append(f"Phase: {', '.join(doc['phases'])}")

    if doc.get("status"):
        parts.append(f"Status: {doc['status']}")

    if doc.get("summary"):
        parts.append(f"Summary: {doc['summary']}")

    if doc.get("description"):
        parts.append(f"Description: {doc['description']}")

    if doc.get("interventions"):
        for iv in doc["interventions"]:
            name = iv.get("name", "")
            desc = iv.get("description", "")
            iv_type = iv.get("type", "")
            parts.append(f"Intervention ({iv_type}): {name}. {desc}")

    if doc.get("eligibility"):
        parts.append(f"Eligibility: {doc['eligibility']}")

    return _clean(" ".join(parts))


def _detect_stages(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        stage
        for stage, keywords in STAGE_KEYWORDS.items()
        if any(kw.lower() in text_lower for kw in keywords)
    ]


def _detect_symptoms(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        symptom
        for symptom, keywords in SYMPTOM_KEYWORDS.items()
        if any(kw.lower() in text_lower for kw in keywords)
    ]


def _detect_labels(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    text_lower = text.lower()
    return [
        label
        for label, keywords in keyword_map.items()
        if any(kw.lower() in text_lower for kw in keywords)
    ]


def _detect_interventions(doc: dict, text: str) -> list[str]:
    names = [iv.get("name", "") for iv in doc.get("interventions", [])]
    # Also scan full text for common drug names
    from scraper import ALZHEIMER_INTERVENTIONS
    found = set()
    text_lower = text.lower()
    for drug in ALZHEIMER_INTERVENTIONS:
        if drug.lower() in text_lower:
            found.add(drug)
    for name in names:
        if name:
            found.add(name)
    return list(found)


def chunk_document(doc: dict) -> Generator[dict, None, None]:
    """
    Yields chunk dicts from a single document.
    Each chunk has all the metadata needed to answer a patient query.
    """
    full_text = _build_full_text(doc)
    if not full_text:
        return

    tokens = _tokenize(full_text)

    # detect metadata once for the whole doc
    detected_stages      = _detect_stages(full_text)
    detected_symptoms    = _detect_symptoms(full_text)
    detected_drugs       = _detect_interventions(doc, full_text)
    detected_biomarkers  = _detect_labels(full_text, BIOMARKER_KEYWORDS)
    detected_outcomes    = _detect_labels(full_text, OUTCOME_KEYWORDS)
    detected_evidence    = _detect_labels(full_text, EVIDENCE_KEYWORDS)
    intervention_classes = _detect_labels(full_text, INTERVENTION_CLASS_KEYWORDS)
    detected_risks       = _detect_labels(full_text, RISK_KEYWORDS)

    # slide a window across tokens
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text   = _detokenize(chunk_tokens).strip()

        if len(chunk_text) > 20:  # skip trivially short tails
            yield {
                # ── identity ──────────────────────────────────────────
                "chunk_id":     f"{doc['id']}_chunk{chunk_index}",
                "doc_id":       doc["id"],
                "source":       doc.get("source", ""),
                "chunk_index":  chunk_index,

                # ── searchable text ───────────────────────────────────
                "content":      chunk_text,

                # ── structured metadata (used for filter queries) ─────
                "title":        doc.get("title", ""),
                "phases":       doc.get("phases", []),
                "status":       doc.get("status", ""),
                "alzheimer_stages":    detected_stages,
                "relevant_symptoms":   detected_symptoms,
                "drugs_mentioned":     detected_drugs,
                "biomarker_tags":      detected_biomarkers,
                "outcome_domains":     detected_outcomes,
                "evidence_types":      detected_evidence,
                "intervention_classes": intervention_classes,
                "safety_risk_tags":    detected_risks,

                # ── source URL for citation ───────────────────────────
                "source_url": (
                    f"https://clinicaltrials.gov/study/{doc['id']}"
                    if doc.get("source") == "clinicaltrials.gov"
                    else f"https://pubmed.ncbi.nlm.nih.gov/{doc['id'].replace('pmid-', '')}"
                ),
            }

        chunk_index += 1
        # move forward by chunk size minus overlap
        start += CHUNK_TOKENS - OVERLAP_TOKENS

        if end == len(tokens):
            break


def chunk_all_documents(docs: list[dict]) -> Generator[dict, None, None]:
    for doc in docs:
        yield from chunk_document(doc)