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
import time
from typing import AsyncGenerator

import httpx
import requests
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


# ── ClinicalTrials.gov (uses requests — httpx gets blocked by Cloudflare) ────

CT_SESSION = requests.Session()

BASE_FIELDS = (
    "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,"
    "InterventionName,InterventionType,ArmGroupDescription,"
    "EligibilityCriteria,Phase,OverallStatus,StudyType,"
    "StartDate,CompletionDate,EnrollmentCount"
)

TRIAL_QUERIES = [
    # ── Drug treatments (core) ───────────────────────────────────────
    {"cond": "Alzheimer Disease", "term": "donepezil OR rivastigmine OR galantamine OR cholinesterase inhibitor"},
    {"cond": "Alzheimer Disease", "term": "memantine OR NMDA antagonist"},
    {"cond": "Alzheimer Disease", "term": "lecanemab OR aducanumab OR donanemab OR anti-amyloid antibody"},
    # ── Sleep & circadian ────────────────────────────────────────────
    {"cond": "Alzheimer Disease", "term": "sleep OR insomnia OR circadian OR melatonin OR trazodone OR light therapy"},
    # ── Behavioral & neuropsychiatric ────────────────────────────────
    {"cond": "Alzheimer Disease", "term": "agitation OR aggression OR depression OR anxiety OR brexpiprazole OR citalopram"},
    # ── Cardiovascular / vital signs ─────────────────────────────────
    {"cond": "Alzheimer Disease", "term": "heart rate OR blood pressure OR cardiovascular OR bradycardia OR hypertension"},
    # ── Biomarkers & progression ─────────────────────────────────────
    {"cond": "Alzheimer Disease", "term": "amyloid PET OR tau PET OR biomarker OR CSF OR neurodegeneration"},
    # ── Non-pharmacologic interventions ──────────────────────────────
    {"cond": "Alzheimer Disease", "term": "exercise OR physical activity OR cognitive training OR music therapy"},
    {"cond": "Alzheimer Disease", "term": "diet OR nutrition OR Mediterranean OR ketogenic OR omega-3"},
    # ── Caregiver & quality of life ──────────────────────────────────
    {"cond": "Alzheimer Disease", "term": "caregiver OR quality of life OR activities of daily living OR ADL"},
    # ── Mild cognitive impairment (early stage) ──────────────────────
    {"cond": "Mild Cognitive Impairment", "term": "Alzheimer OR amyloid OR prevention OR early intervention"},
    # ── Stage-specific: moderate to severe ───────────────────────────
    {"cond": "Alzheimer Disease", "term": "moderate dementia OR severe dementia OR late stage OR palliative"},
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=3, max=15))
def _ct_get_page(params: dict) -> dict:
    """Sync request to ClinicalTrials.gov via requests (bypasses Cloudflare)."""
    r = CT_SESSION.get(CLINICAL_TRIALS_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


async def fetch_clinical_trials(
    max_results: int = 600,
) -> AsyncGenerator[dict, None]:
    """
    Yields flattened trial dicts from ClinicalTrials.gov.
    Runs multiple diverse queries to cover sleep, behavioral, cardiac,
    biomarker, non-drug, and stage-specific Alzheimer trials.
    Uses requests (sync) in a thread because httpx gets blocked by Cloudflare.
    """
    seen_ids: set[str] = set()
    fetched = 0
    per_query_limit = max(max_results // len(TRIAL_QUERIES), 30)
    loop = asyncio.get_event_loop()

    for qi, q in enumerate(TRIAL_QUERIES):
        if fetched >= max_results:
            break

        logger.info(f"  Query {qi+1}/{len(TRIAL_QUERIES)}: {q['term'][:60]}...")
        query_fetched = 0
        next_token = None

        try:
            while query_fetched < per_query_limit and fetched < max_results:
                params = {
                    "query.cond": q["cond"],
                    "query.term": q["term"],
                    "filter.overallStatus": "COMPLETED,ACTIVE_NOT_RECRUITING,RECRUITING",
                    "pageSize": 100,
                    "format": "json",
                    "fields": BASE_FIELDS,
                }
                if next_token:
                    params["pageToken"] = next_token

                # Run sync requests call in a thread to not block the event loop
                data = await loop.run_in_executor(None, _ct_get_page, params)
                studies = data.get("studies", [])
                if not studies:
                    break

                for study in studies:
                    trial = _flatten_trial(study)
                    if trial["id"] in seen_ids:
                        continue
                    seen_ids.add(trial["id"])
                    yield trial
                    fetched += 1
                    query_fetched += 1
                    if fetched >= max_results or query_fetched >= per_query_limit:
                        break

                next_token = data.get("nextPageToken")
                if not next_token:
                    break

                await asyncio.sleep(0.5)

        except Exception as e:
            logger.warning(f"  Query {qi+1} failed: {e}. Skipping to next query.")
            continue

        await asyncio.sleep(1.0)

    logger.info(f"Fetched {fetched} clinical trials across {len(TRIAL_QUERIES)} diverse queries")


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


# ── PubMed (httpx works fine for this domain) ───────────────────────────────

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
    max_results: int = 400,
) -> AsyncGenerator[dict, None]:
    """
    Yields abstract dicts from PubMed for Alzheimer treatment research.
    Diverse queries covering drugs, sleep, behavioral, biomarkers, non-drug, and staging.
    """
    queries = [
        # Drug treatments
        "Alzheimer disease drug treatment dosage randomized controlled trial",
        "Alzheimer dementia cholinesterase inhibitor clinical trial",
        "Alzheimer disease memantine treatment outcome",
        "Alzheimer lecanemab OR aducanumab OR donanemab anti-amyloid",
        # Sleep & circadian
        "Alzheimer disease sleep disorder treatment melatonin trazodone",
        "Alzheimer dementia circadian rhythm light therapy",
        # Behavioral symptoms
        "Alzheimer agitation aggression treatment brexpiprazole",
        "Alzheimer depression anxiety antidepressant",
        # Cardiovascular / vitals
        "Alzheimer disease heart rate bradycardia cholinesterase",
        "Alzheimer disease hypertension blood pressure cardiovascular",
        # Biomarkers & disease progression
        "Alzheimer amyloid tau biomarker disease progression",
        # Non-drug interventions
        "Alzheimer exercise physical activity cognitive outcome",
        "Alzheimer cognitive training intervention",
        "Alzheimer nutrition Mediterranean diet ketogenic",
        # MCI / early stage
        "mild cognitive impairment Alzheimer intervention prevention",
        # Stage-specific
        "moderate severe Alzheimer dementia treatment palliative",
    ]

    import xml.etree.ElementTree as ET

    async with httpx.AsyncClient(follow_redirects=True) as client:
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
    """Yields all documents from both sources. Continues if one source fails."""
    try:
        async for doc in fetch_clinical_trials(max_results=600):
            yield doc
    except Exception as e:
        logger.error(f"ClinicalTrials.gov failed: {e}. Continuing with PubMed...")

    try:
        async for doc in fetch_pubmed_abstracts(max_results=400):
            yield doc
    except Exception as e:
        logger.error(f"PubMed failed: {e}.")


if __name__ == "__main__":
    async def _preview():
        count = 0
        async for doc in fetch_all():
            count += 1
            if count <= 2:
                print(doc["id"], "|", doc["title"][:80])
        print(f"Total: {count}")

    asyncio.run(_preview())
