"""
api/function_app.py
────────────────────
Azure Functions v2 (Python) entry point.

Endpoints:
  POST /api/recommend      — main RAG recommendation
  GET  /api/health         — health check
  GET  /api/stages         — list valid Alzheimer stages

Authentication: Function-level API key
  • Pass as header:    x-functions-key: <key>
  • Or as query param: ?code=<key>
  Azure enforces this automatically when authLevel = "function".

CORS: configured in host.json — your frontend origin goes there.
"""

import json
import logging
import os

import azure.functions as func
from pydantic import ValidationError

from models import AlzheimerStage, ErrorResponse, RecommendRequest
from rag_handler import get_recommendation

logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


# ── Helper ────────────────────────────────────────────────────────────────────

def _json_response(data: dict | list, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps(data, default=str),
        status_code=status_code,
        mimetype="application/json",
        headers={
            "Content-Type":                "application/json",
            "X-Content-Type-Options":      "nosniff",
            "Strict-Transport-Security":   "max-age=31536000",
        },
    )


def _error(message: str, detail: str | None = None, code: int = 400) -> func.HttpResponse:
    body = ErrorResponse(error=message, detail=detail, code=code)
    return _json_response(body.model_dump(), status_code=code)


# ── POST /api/recommend ───────────────────────────────────────────────────────

@app.route(route="recommend", methods=["POST"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main recommendation endpoint.

    Request body (JSON):
    {
        "patient_id": "abc123",
        "stage": "mild",                   // very_mild | mild | moderate | severe
        "age": 72,
        "vitals": {
            "sleep_hours": 4.5,
            "sleep_quality": "poor",       // good | fair | poor | unknown
            "heart_rate_bpm": 88,
            "systolic_bp": 138,
            "diastolic_bp": 85
        },
        "current_medications": ["metformin"],
        "comorbidities": ["hypertension", "type2_diabetes"],
        "top_k": 5
    }
    """
    logger.info("POST /api/recommend called")

    # Parse body
    try:
        body = req.get_json()
    except ValueError:
        return _error("Request body must be valid JSON", code=400)

    # Validate with Pydantic
    try:
        recommend_req = RecommendRequest.model_validate(body)
    except ValidationError as e:
        errors = e.errors()
        detail = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in errors
        )
        return _error("Validation failed", detail=detail, code=422)

    # Run RAG pipeline
    try:
        response = get_recommendation(recommend_req)
        return _json_response(response.model_dump())

    except ValueError as e:
        logger.warning(f"Business error: {e}")
        return _error(str(e), code=404)

    except Exception as e:
        logger.exception("Unexpected error in recommend handler")
        return _error(
            "Internal server error",
            detail=str(e) if os.environ.get("FUNCTIONS_ENVIRONMENT") == "Development" else None,
            code=500,
        )


# ── GET /api/health ───────────────────────────────────────────────────────────

@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check. Returns 200 with service status.
    You can call this without an API key from monitoring tools
    by changing authLevel to ANONYMOUS for this route only.
    """
    return _json_response({
        "status":  "healthy",
        "service": "alzheimer-rag-api",
        "version": "1.0.0",
    })


# ── GET /api/stages ───────────────────────────────────────────────────────────

@app.route(route="stages", methods=["GET"])
def stages(req: func.HttpRequest) -> func.HttpResponse:
    """Returns valid Alzheimer stage values for frontend dropdowns."""
    stage_info = {
        AlzheimerStage.VERY_MILD: "Very mild dementia (CDR 0.5) — subjective / MCI",
        AlzheimerStage.MILD:      "Mild dementia (CDR 1) — noticeable memory loss",
        AlzheimerStage.MODERATE:  "Moderate dementia (CDR 2) — needs daily assistance",
        AlzheimerStage.SEVERE:    "Severe dementia (CDR 3) — fully dependent",
    }
    return _json_response([
        {"value": stage.value, "label": label}
        for stage, label in stage_info.items()
    ])