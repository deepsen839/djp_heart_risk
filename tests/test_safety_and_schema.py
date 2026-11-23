from fastapi.testclient import TestClient
from djp_heart_risk.api import app, illustrative_band, PredictionInput
from djp_heart_risk.rules import ClinicianRuleResult, ClinicianRulesPort

client=TestClient(app)

def test_probability_band_boundaries_are_project_only():
    assert illustrative_band(0.0)=="LOW"
    assert illustrative_band(0.2499)=="LOW"
    assert illustrative_band(0.25)=="MEDIUM"
    assert illustrative_band(0.75)=="MEDIUM"
    assert illustrative_band(0.7501)=="HIGH"
    assert illustrative_band(1.0)=="HIGH"

def test_emergency_rule_true_is_not_overridden_by_low_ml_score():
    # ML probability is deliberately absent from the rules interface.
    supplied=ClinicianRuleResult(emergency_flag=True)
    result=ClinicianRulesPort().accept_mock_result(supplied)
    assert result.emergency_flag is True

def test_prediction_schema_rejects_extra_fields():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PredictionInput(age=50,trestbps=120,chol=200,thalach=150,oldpeak=1,sex=1,cp=2,fbs=0,restecg=0,exang=0,slope=2,ca=0,thal=3,unexpected=1)

def test_health_endpoint():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["mode"]=="research_demo_only"

def test_low_model_probability_does_not_clear_independent_emergency_flag(monkeypatch):
    class LowScorePipeline:
        def predict_proba(self, X):
            return [[0.90, 0.10]]
    from djp_heart_risk import api
    monkeypatch.setattr(api, "load_pipeline", lambda: LowScorePipeline())
    payload={"age":52,"sex":1,"cp":2,"trestbps":128,"chol":210,"fbs":0,"restecg":1,"thalach":155,"exang":0,"oldpeak":1.2,"slope":2,"ca":0,"thal":3,"mock_clinician_emergency_flag":True}
    response=client.post("/predict",json=payload)
    assert response.status_code==200
    body=response.json()
    assert body["risk_probability"] < 0.25
    assert body["emergency_flag"] is True

def test_prediction_output_probability_bounds():
    from djp_heart_risk.api import PredictionOutput, DISCLAIMER
    import pytest
    with pytest.raises(Exception):
        PredictionOutput(risk_probability=1.2,illustrative_band="HIGH",model_version="x",pipeline_version="y",disclaimer=DISCLAIMER)


def test_model_loader_reports_missing_artifact(monkeypatch, tmp_path):
    from djp_heart_risk import api
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path)
    api.load_pipeline.cache_clear()
    import pytest
    with pytest.raises(FileNotFoundError, match="Run `djp-train` first"):
        api.load_pipeline()
    api.load_pipeline.cache_clear()


def test_prometheus_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "djp_model_loaded" in response.text
