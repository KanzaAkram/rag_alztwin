"""
api/rag_handler.py
───────────────────
Core RAG logic:
  1. Embed the patient query
  2. Hybrid search (vector + keyword) against Azure AI Search
  3. Build context from top-k chunks
  4. Call GPT-4o for the recommendation
  5. Return structured response
"""

import json
import logging
import os
import re

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from models import (
    RecommendRequest,
    RecommendResponse,
    TrialSource,
    TreatmentRecommendation,
)
from patient_context import (
    build_search_query,
    build_system_prompt,
    build_user_message,
)

logger = logging.getLogger(__name__)

# ── Clients (module-level singletons — reused across warm invocations) ────────

_embedding_client: AzureOpenAI | None = None
_chat_client: OpenAI | None = None
_search_client: SearchClient | None = None


def _get_embedding_client() -> AzureOpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-12-01-preview"),
        )
    return _embedding_client


def _get_chat_client() -> OpenAI:
    global _chat_client
    if _chat_client is None:
        _chat_client = OpenAI(
            base_url=os.environ["AZURE_OPENAI_CHAT_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_CHAT_API_KEY"],
        )
    return _chat_client


def _get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:
        _search_client = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=os.environ.get("AZURE_SEARCH_INDEX_NAME", "alzheimer-trials"),
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_QUERY_KEY"]),
        )
    return _search_client


# ── Embedding ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _embed_query(text: str) -> list[float]:
    client = _get_embedding_client()
    response = client.embeddings.create(
        input=text,
        model=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NEW", "text-embedding-3-small"),
    )
    return response.data[0].embedding


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _hybrid_search(query_text: str, query_vector: list[float], top_k: int) -> list[dict]:
    """
    Hybrid search: combines BM25 keyword score with HNSW vector score.
    Azure AI Search merges them via Reciprocal Rank Fusion (RRF).
    """
    client = _get_search_client()

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k * 2,   # over-fetch then re-rank
        fields="content_vector",
    )

    results = client.search(
        search_text=query_text,           # keyword arm
        vector_queries=[vector_query],    # vector arm
        select=[
            "chunk_id", "doc_id", "source", "source_url",
            "title", "content", "phases", "alzheimer_stages",
            "relevant_symptoms", "drugs_mentioned", "status",
            "biomarker_tags", "outcome_domains", "evidence_types",
            "intervention_classes", "safety_risk_tags",
        ],
        top=top_k,
    )

    chunks = []
    for r in results:
        chunks.append({
            "chunk_id":        r["chunk_id"],
            "doc_id":          r["doc_id"],
            "source":          r.get("source", ""),
            "source_url":      r.get("source_url", ""),
            "title":           r.get("title", ""),
            "content":         r.get("content", ""),
            "phases":          r.get("phases", ""),
            "alzheimer_stages":r.get("alzheimer_stages", ""),
            "drugs_mentioned": r.get("drugs_mentioned", ""),
            "biomarker_tags":  r.get("biomarker_tags", ""),
            "outcome_domains": r.get("outcome_domains", ""),
            "evidence_types":  r.get("evidence_types", ""),
            "intervention_classes": r.get("intervention_classes", ""),
            "safety_risk_tags": r.get("safety_risk_tags", ""),
            "status":          r.get("status", ""),
            "score":           r.get("@search.score", 0.0),
        })

    return chunks


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Keep highest-scoring chunk per doc_id."""
    seen: dict[str, dict] = {}
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        if doc_id not in seen or chunk["score"] > seen[doc_id]["score"]:
            seen[doc_id] = chunk
    return list(seen.values())


def _build_context_string(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for GPT-4o."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Trial {i}] {chunk['title']}\n"
            f"Source: {chunk['source']} | Status: {chunk['status']} | Phase: {chunk['phases']}\n"
            f"Relevant stages: {chunk['alzheimer_stages']} | Drugs: {chunk['drugs_mentioned']}\n"
            f"Biomarkers: {chunk['biomarker_tags']} | Outcomes: {chunk['outcome_domains']}\n"
            f"Evidence: {chunk['evidence_types']} | Risk tags: {chunk['safety_risk_tags']}\n"
            f"Excerpt:\n{chunk['content']}\n"
            f"URL: {chunk['source_url']}"
        )
    return "\n\n" + "─" * 60 + "\n\n".join(parts)


# ── Generation ────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
def _call_gpt4o(system_prompt: str, user_message: str) -> str:
    client = _get_chat_client()
    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NEW", "gpt-oss-120b"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,       # low temperature for clinical consistency
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _parse_gpt_response(raw: str) -> dict:
    """Parse GPT-4o JSON response, with fallback for malformed output."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # strip markdown code fences if present
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("GPT-4o returned non-JSON, using fallback")
            return {
                "treatment":       "Unable to parse recommendation",
                "dosage":          "Please consult clinical guidelines",
                "rationale":       raw[:500],
                "cautions":        [],
                "monitoring":      [],
                "lifestyle_notes": [],
            }


# ── Main entry point ──────────────────────────────────────────────────────────

def get_recommendation(req: RecommendRequest) -> RecommendResponse:
    """
    Full RAG pipeline:
    embed → search → build context → generate → return structured response
    """

    # 1. Build search query from patient data
    search_query = build_search_query(req)
    logger.info(f"Search query: {search_query[:100]}")

    # 2. Embed the query
    query_vector = _embed_query(search_query)

    # 3. Hybrid retrieval
    raw_chunks = _hybrid_search(search_query, query_vector, top_k=req.top_k * 2)
    chunks     = _deduplicate_chunks(raw_chunks)[: req.top_k]

    if not chunks:
        raise ValueError("No relevant clinical trial data found for this patient profile.")

    logger.info(f"Retrieved {len(chunks)} unique trial chunks")

    # 4. Build context + prompts
    context_str   = _build_context_string(chunks)
    system_prompt = build_system_prompt(req, context_str)
    user_message  = build_user_message(req)

    # 5. Generate recommendation
    raw_response = _call_gpt4o(system_prompt, user_message)
    parsed       = _parse_gpt_response(raw_response)

    # 6. Build structured response
    recommendation = TreatmentRecommendation(
        treatment       = parsed.get("treatment", ""),
        dosage          = parsed.get("dosage", ""),
        rationale       = parsed.get("rationale", ""),
        cautions        = parsed.get("cautions", []),
        monitoring      = parsed.get("monitoring", []),
        lifestyle_notes = parsed.get("lifestyle_notes", []),
    )

    sources = [
        TrialSource(
            doc_id          = c["doc_id"],
            title           = c["title"],
            source          = c["source"],
            source_url      = c["source_url"],
            relevance_score = round(c["score"], 4),
            phases          = c["phases"],
        )
        for c in chunks
    ]

    return RecommendResponse(
        patient_id     = req.patient_id,
        stage          = req.stage,
        recommendation = recommendation,
        sources        = sources,
    )