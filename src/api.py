from datetime import datetime, timedelta
import os
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.service import build_store

app = FastAPI(title="Visa Processing Prediction API", version="0.1.0")
store = build_store()
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "0") == "1"

cors_origins = [
    origin.strip()
    for origin in ("https://jingran-zhou.com,https://www.jingran-zhou.com,http://localhost:4321").split(",")
    if origin.strip()
]
extra_origins = [
    origin.strip()
    for origin in (str((__import__("os").getenv("CORS_ORIGINS") or ""))).split(",")
    if origin.strip()
]
allowed_origins = list(dict.fromkeys(cors_origins + extra_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


class PredictRequest(BaseModel):
    check_date: str = Field(..., description="YYYY-MM-DD")
    consulate: str = Field(default="GuangZhou")
    visa_type: str = Field(default="H1")
    major: str = Field(default="CS")
    as_of: Optional[str] = Field(default=None, description="YYYY-MM-DD")


class PredictResponse(BaseModel):
    profile: dict
    forecast: dict
    metadata: dict


@app.on_event("startup")
def on_startup() -> None:
    store.ensure_model()


@app.get("/health")
def health(check_model: int = 0) -> dict:
    payload = {"status": "ok", "data_path": store.data_path}
    if check_model:
        try:
            store.ensure_model()
            payload["model_status"] = "ok"
        except Exception as exc:
            payload["model_status"] = "error"
            payload["error"] = str(exc) if DEBUG_ERRORS else "model_init_failed"
    return payload

@app.head("/health")
def health_head() -> None:
    return None

@app.get("/options")
def options() -> dict:
    try:
        store.ensure_model()
        return store.get_options()
    except Exception as exc:
        detail = str(exc) if DEBUG_ERRORS else "options_failed"
        raise HTTPException(status_code=500, detail=detail) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, background_tasks: BackgroundTasks) -> PredictResponse:
    t_start = datetime.now()
    try:
        store.ensure_model()
    except Exception as exc:
        detail = str(exc) if DEBUG_ERRORS else "model_init_failed"
        raise HTTPException(status_code=500, detail=detail) from exc
    store.maybe_refresh_background()

    check_date = _parse_date(body.check_date)
    reference = _parse_date(body.as_of) if body.as_of else datetime.now()
    t0 = (reference - check_date).days

    if t0 < 0:
        raise HTTPException(status_code=400, detail="check_date is in the future")

    consulate = body.consulate.strip() or "GuangZhou"
    visa_type = body.visa_type.strip() or "H1"
    major_bucket = store.bucket_major(body.major)

    forecast, perf = store.predict(consulate, visa_type, major_bucket, t0)

    estimated_completion = None
    if "ExpectedValue" in forecast and forecast["ExpectedValue"] is not None:
        estimated_completion = check_date + _days(int(round(forecast["ExpectedValue"])))

    percentile_dates = {}
    for key in ("P25", "P50", "P75", "P90"):
        if key in forecast and forecast[key] is not None:
            percentile_dates[f"{key}Date"] = (check_date + _days(int(round(forecast[key])))).date().isoformat()

    response = PredictResponse(
        profile={
            "consulate": consulate,
            "visa_type": visa_type,
            "major_bucket": major_bucket,
            "check_date": check_date.date().isoformat(),
            "as_of": reference.date().isoformat(),
            "elapsed_days": t0,
        },
        forecast={
            **forecast,
            "EstimatedCompletion": estimated_completion.date().isoformat() if estimated_completion else None,
            **percentile_dates,
        },
        metadata={
            "data_mtime": store.model_mtime,
            "refresh_ttl_seconds": store.refresh_ttl_seconds,
            "params_cached": perf.get("params_cached"),
            "forecast_cached": perf.get("forecast_cached"),
            "fit_ms": perf.get("fit_ms"),
            "predict_ms": perf.get("predict_ms"),
            "server_time": t_start.isoformat(),
        },
    )

    if store.should_refresh():
        background_tasks.add_task(store.refresh_now)

    return response


@app.post("/refresh")
def refresh(request: Request) -> dict:
    api_key = __import__("os").getenv("VISA_REFRESH_KEY")
    if api_key:
        provided = request.headers.get("x-api-key")
        if not provided or provided != api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    count = store.refresh_now()
    return {"status": "refreshed", "count": count}


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid date format, use YYYY-MM-DD") from exc


def _days(num: int) -> timedelta:
    return timedelta(days=num)
