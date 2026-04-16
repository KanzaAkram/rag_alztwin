"""
ingestion/indexer.py
─────────────────────
1. Creates (or updates) the Azure AI Search index with vector + keyword fields
2. Embeds chunks using Azure OpenAI text-embedding-3-large
3. Uploads chunks in batches to the index

Run:
    python indexer.py

Environment variables required (see .env.example):
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT
"""

import asyncio
import logging
import os
from itertools import islice
from typing import Generator

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from chunker import chunk_document
from scraper import fetch_all

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SEARCH_ENDPOINT  = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME       = os.environ.get("AZURE_SEARCH_INDEX_NAME", "alzheimer-trials")
EMBEDDING_MODEL  = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NEW", "text-embedding-3-small")
EMBEDDING_DIMS   = 1536   # text-embedding-3-small output size
BATCH_SIZE       = 16     # documents per embedding API call
UPLOAD_BATCH     = 100    # documents per Search upload batch

openai_client = AzureOpenAI(
    azure_endpoint = os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
    api_key        = os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
    api_version    = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-12-01-preview"),
)


# ── Index schema ──────────────────────────────────────────────────────────────

def _build_index() -> SearchIndex:
    """
    Hybrid index: keyword (BM25) + vector (HNSW) on the content field.
    Metadata fields are filterable so we can narrow by stage / symptom.
    """
    fields = [
        SimpleField(name="chunk_id",   type=SearchFieldDataType.String, key=True,        filterable=True),
        SimpleField(name="doc_id",     type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source",     type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="status",     type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_url", type=SearchFieldDataType.String),
        SimpleField(name="chunk_index",type=SearchFieldDataType.Int32),

        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="title",   type=SearchFieldDataType.String, analyzer_name="en.microsoft"),

        # Collections stored as comma-separated strings (Search doesn't support string arrays in Basic tier)
        SimpleField(name="phases",             type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="alzheimer_stages",   type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="relevant_symptoms",  type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="drugs_mentioned",    type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="biomarker_tags",     type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="outcome_domains",    type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="evidence_types",     type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="intervention_classes", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="safety_risk_tags",   type=SearchFieldDataType.String, filterable=True),

        # Vector field
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-config")],
    )

    return SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)


def create_or_update_index() -> None:
    client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
    )

    # Vector dims are immutable on an existing index. If the dims changed
    # (e.g. switching embedding models), delete and recreate.
    try:
        existing = client.get_index(INDEX_NAME)
        existing_dims = next(
            (f.vector_search_dimensions for f in existing.fields
             if f.name == "content_vector"),
            None,
        )
        if existing_dims and existing_dims != EMBEDDING_DIMS:
            logger.warning(
                f"Index '{INDEX_NAME}' has dims={existing_dims}, "
                f"need {EMBEDDING_DIMS}. Deleting and recreating."
            )
            client.delete_index(INDEX_NAME)
    except Exception as e:
        logger.info(f"No existing index to check ({e.__class__.__name__}); creating fresh.")

    index = _build_index()
    result = client.create_or_update_index(index)
    logger.info(f"Index '{result.name}' ready (dims={EMBEDDING_DIMS})")


# ── Embedding ─────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _embed_batch(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def _batched(iterable, n: int) -> Generator:
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


# ── Upload ────────────────────────────────────────────────────────────────────

def _prepare_doc(chunk: dict, embedding: list[float]) -> dict:
    """Convert chunk dict to the shape Azure Search expects."""
    return {
        "chunk_id":            chunk["chunk_id"],
        "doc_id":              chunk["doc_id"],
        "source":              chunk["source"],
        "status":              chunk["status"],
        "source_url":          chunk["source_url"],
        "chunk_index":         chunk["chunk_index"],
        "content":             chunk["content"],
        "title":               chunk["title"],
        "phases":              ", ".join(chunk.get("phases", [])),
        "alzheimer_stages":    ", ".join(chunk.get("alzheimer_stages", [])),
        "relevant_symptoms":   ", ".join(chunk.get("relevant_symptoms", [])),
        "drugs_mentioned":     ", ".join(chunk.get("drugs_mentioned", [])),
        "biomarker_tags":      ", ".join(chunk.get("biomarker_tags", [])),
        "outcome_domains":     ", ".join(chunk.get("outcome_domains", [])),
        "evidence_types":      ", ".join(chunk.get("evidence_types", [])),
        "intervention_classes": ", ".join(chunk.get("intervention_classes", [])),
        "safety_risk_tags":    ", ".join(chunk.get("safety_risk_tags", [])),
        "content_vector":      embedding,
    }


def upload_chunks(chunks: list[dict]) -> None:
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
    )

    upload_buffer: list[dict] = []
    total_uploaded = 0

    for embed_batch in _batched(chunks, BATCH_SIZE):
        texts      = [c["content"] for c in embed_batch]
        embeddings = _embed_batch(texts)

        for chunk, embedding in zip(embed_batch, embeddings):
            upload_buffer.append(_prepare_doc(chunk, embedding))

        if len(upload_buffer) >= UPLOAD_BATCH:
            result = search_client.upload_documents(documents=upload_buffer)
            succeeded = sum(1 for r in result if r.succeeded)
            total_uploaded += succeeded
            logger.info(f"Uploaded {total_uploaded} chunks so far...")
            upload_buffer.clear()

    # flush remainder
    if upload_buffer:
        result = search_client.upload_documents(documents=upload_buffer)
        succeeded = sum(1 for r in result if r.succeeded)
        total_uploaded += succeeded

    logger.info(f"Indexing complete. Total chunks uploaded: {total_uploaded}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_ingestion_pipeline() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Step 1/3  — Creating/verifying Azure AI Search index...")
    create_or_update_index()

    logger.info("Step 2/3  — Fetching documents from ClinicalTrials.gov + PubMed...")
    all_chunks: list[dict] = []
    doc_count = 0

    async for doc in fetch_all():
        doc_count += 1
        chunks = list(chunk_document(doc))
        all_chunks.extend(chunks)
        if doc_count % 50 == 0:
            logger.info(f"  Processed {doc_count} docs → {len(all_chunks)} chunks")

    logger.info(f"  Total: {doc_count} docs → {len(all_chunks)} chunks")

    logger.info("Step 3/3  — Embedding and uploading to Azure AI Search...")
    upload_chunks(all_chunks)


if __name__ == "__main__":
    asyncio.run(run_ingestion_pipeline())