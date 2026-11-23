from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import CONTENT_TYPE_LATEST
from .metrics import REQUESTS, LATENCY, PREDICTIONS, MODEL_LOADED, metrics_payload
from .config import FEATURES, MODELS_DIR, MODEL_VERSION, PIPELINE_VERSION
from .rules import ClinicianRuleResult, ClinicianRulesPort

DISCLAIMER = ("Research/demo output only. This is not a diagnosis, not a clinically validated probability, "
              "and must not be used to make care decisions. Emergency criteria must come from a separate, "
              "qualified clinician-approved rules service.")

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(..., ge=1, le=120)
    trestbps: float = Field(..., ge=40, le=300)
    chol: float = Field(..., ge=50, le=800)
    thalach: float = Field(..., ge=40, le=250)
    oldpeak: float = Field(..., ge=-10, le=20)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    exang: int = Field(..., ge=0, le=1)
    slope: int = Field(..., ge=1, le=3)
    ca: int | None = Field(default=None, ge=0, le=3)
    thal: int | None = Field(default=None, ge=3, le=7)
    # This field is a MOCK fixture for demonstrating the separate rules port only.
    mock_clinician_emergency_flag: bool | None = None

class PredictionOutput(BaseModel):
    risk_probability: float = Field(..., ge=0, le=1)
    illustrative_band: str
    model_version: str
    pipeline_version: str
    disclaimer: str
    emergency_flag: bool | None = None
    emergency_rule_source: str = "independent clinician-rules interface; mock only"

app=FastAPI(title="DJP Heart-Risk Research Demo API",version="1.0.0",description=DISCLAIMER)

@lru_cache(maxsize=1)
def load_pipeline():
    path=MODELS_DIR/"heart_risk_pipeline.joblib"
    if not path.exists(): raise FileNotFoundError("Model artefact not found. Run `djp-train` first.")
    return joblib.load(path)

def illustrative_band(p: float) -> str:
    # Provisional assignment bands only; never a clinical threshold.
    if p < 0.25: return "LOW"
    if p <= 0.75: return "MEDIUM"
    return "HIGH"

@app.get("/metrics", include_in_schema=False)
def metrics():
    payload, content_type = metrics_payload()
    from fastapi.responses import Response
    return Response(content=payload, media_type=content_type)

@app.get("/health")
def health():
    loaded = (MODELS_DIR/"heart_risk_pipeline.joblib").exists()
    MODEL_LOADED.set(1 if loaded else 0)
    return {"status":"ok","mode":"research_demo_only","model_loaded":loaded}

@app.post("/predict",response_model=PredictionOutput)
def predict(payload: PredictionInput):
    with LATENCY.labels(endpoint="predict").time():
        return _predict(payload)

def _predict(payload: PredictionInput):
    try: pipeline=load_pipeline()
    except FileNotFoundError as exc: raise HTTPException(status_code=503,detail=str(exc)) from exc
    values=payload.model_dump(exclude={"mock_clinician_emergency_flag"})
    # Explicitly preserve model feature order and nulls for fitted imputers.
    X=pd.DataFrame([{name:values.get(name) for name in FEATURES}],columns=FEATURES)
    probability=float(pipeline.predict_proba(X)[0][1])
    PREDICTIONS.inc()
    REQUESTS.labels(method="POST", endpoint="predict", status="success").inc()
    rule_result=None
    if payload.mock_clinician_emergency_flag is not None:
        rule_result=ClinicianRulesPort().accept_mock_result(ClinicianRuleResult(payload.mock_clinician_emergency_flag))
    return PredictionOutput(risk_probability=probability,illustrative_band=illustrative_band(probability),model_version=MODEL_VERSION,pipeline_version=PIPELINE_VERSION,disclaimer=DISCLAIMER,emergency_flag=None if rule_result is None else rule_result.emergency_flag)
