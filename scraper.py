"""
ingestion/scraper.py
────────────────────
Fetches Alzheimer-related clinical trials from ClinicalTrials.gov v2 API
and research abstracts from PubMed/NCBI.

Run:
    python -m ingestion.scraper
"""

import asyncio
import logging
from typing import AsyncGenerator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

CLINICAL_TRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

ALZHEIMER_INTERVENTIONS = [
    "Donepezil", "Rivastigmine", "Galantamine", "Memantine",
    "Aducanumab", "Lecanemab", "Donanemab", "Brexpiprazole",
    "Citalopram", "Mirtazapine", "Melatonin", "Trazodone",
]


# ── ClinicalTrials.gov ────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _get_page(client: httpx.AsyncClient, params: dict) -> dict:
    r = await client.get(CLINICAL_TRIALS_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


async def fetch_clinical_trials(
    max_results: int = 500,
) -> AsyncGenerator[dict, None]:
    """
    Yields flattened trial dicts from ClinicalTrials.gov.
    Covers: interventional studies, all phases, Alzheimer condition.
    """
    params = {
        "query.cond": "Alzheimer Disease",
        "query.term": "treatment OR therapy OR intervention OR dosage",
        "filter.overallStatus": "COMPLETED,ACTIVE_NOT_RECRUITING,RECRUITING",
        "pageSize": 100,
        "format": "json",
        "fields": (
            "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,"
            "InterventionName,InterventionType,ArmGroupDescription,"
            "EligibilityCriteria,Phase,OverallStatus,StudyType,"
            "StartDate,CompletionDate,EnrollmentCount"
        ),
    }

    async with httpx.AsyncClient() as client:
        next_token = None
        fetched = 0

        while fetched < max_results:
            if next_token:
                params["pageToken"] = next_token
            elif "pageToken" in params:
                del params["pageToken"]

            data = await _get_page(client, params)
            studies = data.get("studies", [])

            if not studies:
                break

            for study in studies:
                yield _flatten_trial(study)
                fetched += 1
                if fetched >= max_results:
                    break

            next_token = data.get("nextPageToken")
            if not next_token:
                break

    logger.info(f"Fetched {fetched} clinical trials")


def _flatten_trial(study: dict) -> dict:
    proto   = study.get("protocolSection", {})
    id_mod  = proto.get("identificationModule", {})
    desc    = proto.get("descriptionModule", {})
    arms    = proto.get("armsInterventionsModule", {})
    elig    = proto.get("eligibilityModule", {})
    status  = proto.get("statusModule", {})
    design  = proto.get("designModule", {})

    interventions = [
        {
            "name": i.get("name", ""),
            "type": i.get("type", ""),
            "description": i.get("description", ""),
        }
        for i in arms.get("interventions", [])
    ]

    phases = design.get("phaseList", {}).get("phase", [])

    return {
        "id":             id_mod.get("nctId", ""),
        "source":         "clinicaltrials.gov",
        "title":          id_mod.get("officialTitle") or id_mod.get("briefTitle", ""),
        "summary":        desc.get("briefSummary", "").strip(),
        "description":    desc.get("detailedDescription", "").strip(),
        "interventions":  interventions,
        "eligibility":    elig.get("eligibilityCriteria", "").strip(),
        "phases":         phases,
        "status":         status.get("overallStatus", ""),
        "study_type":     design.get("studyType", ""),
        "enrollment":     design.get("enrollmentInfo", {}).get("count"),
        "start_date":     status.get("startDateStruct", {}).get("date", ""),
        "completion_date":status.get("completionDateStruct", {}).get("date", ""),
    }


# ── PubMed ────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _pubmed_search(client: httpx.AsyncClient, query: str, retmax: int) -> list[str]:
    r = await client.get(PUBMED_SEARCH_URL, params={
        "db":     "pubmed",
        "term":   query,
        "retmax": retmax,
        "retmode":"json",
    }, timeout=20)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _pubmed_fetch(client: httpx.AsyncClient, pmids: list[str]) -> str:
    r = await client.post(PUBMED_FETCH_URL, data={
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }, timeout=30)
    r.raise_for_status()
    return r.text


async def fetch_pubmed_abstracts(
    max_results: int = 200,
) -> AsyncGenerator[dict, None]:
    """
    Yields abstract dicts from PubMed for Alzheimer treatment research.
    """
    queries = [
        "Alzheimer disease drug treatment dosage randomized controlled trial",
        "Alzheimer dementia cholinesterase inhibitor clinical trial",
        "Alzheimer disease memantine treatment outcome",
        "mild cognitive impairment Alzheimer intervention",
        "Alzheimer disease sleep disorder treatment",
    ]

    import xml.etree.ElementTree as ET

    async with httpx.AsyncClient() as client:
        seen_pmids: set[str] = set()
        yielded = 0

        for query in queries:
            if yielded >= max_results:
                break

            pmids = await _pubmed_search(client, query, retmax=40)
            new_pmids = [p for p in pmids if p not in seen_pmids]
            if not new_pmids:
                continue

            seen_pmids.update(new_pmids)

            # fetch in batches of 20
            for batch_start in range(0, len(new_pmids), 20):
                batch = new_pmids[batch_start:batch_start + 20]
                xml_text = await _pubmed_fetch(client, batch)

                try:
                    root = ET.fromstring(xml_text)
                except ET.ParseError:
                    continue

                for article in root.findall(".//PubmedArticle"):
                    pmid_el = article.find(".//PMID")
                    pmid    = pmid_el.text if pmid_el is not None else ""

                    title_el   = article.find(".//ArticleTitle")
                    title      = title_el.text or "" if title_el is not None else ""

                    abstract_parts = article.findall(".//AbstractText")
                    abstract = " ".join(
                        (el.text or "") for el in abstract_parts if el.text
                    ).strip()

                    if not abstract:
                        continue

                    pub_date = article.find(".//PubDate")
                    year = ""
                    if pub_date is not None:
                        year_el = pub_date.find("Year")
                        year = year_el.text if year_el is not None else ""

                    yield {
                        "id":          f"pmid-{pmid}",
                        "source":      "pubmed",
                        "title":       title.strip(),
                        "summary":     abstract[:500],
                        "description": abstract,
                        "phases":      [],
                        "interventions": [],
                        "eligibility": "",
                        "status":      "published",
                        "year":        year,
                    }

                    yielded += 1
                    if yielded >= max_results:
                        break

    logger.info(f"Fetched {yielded} PubMed abstracts")


# ── Entry point ───────────────────────────────────────────────────────────────

async def fetch_all() -> AsyncGenerator[dict, None]:
    """Yields all documents from both sources."""
    async for doc in fetch_clinical_trials(max_results=500):
        yield doc
    async for doc in fetch_pubmed_abstracts(max_results=200):
        yield doc


if __name__ == "__main__":
    async def _preview():
        count = 0
        async for doc in fetch_all():
            count += 1
            if count <= 2:
                print(doc["id"], "|", doc["title"][:80])
        print(f"Total: {count}")

    asyncio.run(_preview())