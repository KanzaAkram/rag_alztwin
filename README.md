# Alzheimer Clinical-Trial RAG Backend (Azure Prototype)

This backend provides:
- ingestion of Alzheimer clinical-trial and PubMed evidence
- vector + keyword index in Azure AI Search
- recommendation API for clinician-facing treatment suggestions using patient vitals and dementia stage

It is designed for **prototype use on Azure services** and can be consumed from a web frontend.

## Architecture

1. `scraper.py`
   - pulls data from ClinicalTrials.gov + PubMed
2. `chunker.py`
   - creates metadata-rich text chunks
3. `indexer.py`
   - embeds chunks with Azure OpenAI
   - uploads to Azure AI Search
4. `function_app.py` + `rag_handler.py`
   - exposes HTTP API via Azure Functions
   - performs RAG retrieval + generation

## API Endpoints

- `POST /api/recommend`
- `GET /api/health`
- `GET /api/stages`

Example request for `POST /api/recommend`:

```json
{
  "patient_id": "p-123",
  "stage": "mild",
  "age": 72,
  "vitals": {
    "sleep_hours": 5.1,
    "sleep_quality": "poor",
    "heart_rate_bpm": 88,
    "systolic_bp": 138,
    "diastolic_bp": 84
  },
  "current_medications": ["metformin"],
  "comorbidities": ["hypertension"],
  "top_k": 5
}
```

## Local Setup

1. Create virtual environment and install deps
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
   - `pip install -r requirements.txt`
2. Copy `.env.example` values into `local_settings.json` (`Values` object)
3. Run ingestion:
   - `python indexer.py`
4. Run Azure Functions locally:
   - `func start`

## Frontend Consumption

Call:
- local: `http://localhost:7071/api/recommend`
- deployed: `https://<function-app>.azurewebsites.net/api/recommend`

Use function key:
- header: `x-functions-key: <key>`
- or query: `?code=<key>`

## Best Azure Services for Prototype (Free/Low-cost first)

- **Azure Functions (Consumption plan)**: API hosting, pay-per-use
- **Azure AI Search (Free tier)**: vector + keyword retrieval for prototype data size
- **Azure OpenAI**: embeddings + chat recommendation generation
- **Application Insights** (basic): request/latency/error telemetry
- **Azure Key Vault** (optional for prototype, recommended before production): secret storage

## Firebase Integration Pattern

Keep Firebase as your main backend and call this Azure API as a clinical recommendation microservice.

Suggested flow:
1. Frontend authenticates with Firebase
2. Frontend calls your Firebase backend
3. Firebase backend forwards curated patient payload to Azure `/api/recommend`
4. Firebase returns recommendation + trial citations to frontend

This keeps Azure keys off the client.

## Important Safety Note

This project is clinical decision support only. It should not replace clinician judgment. Always validate against current treatment guidelines and full patient context.
