from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# Make the src-layout package importable on Streamlit Community Cloud without
# requiring an editable package installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from djp_heart_risk.config import (  # noqa: E402
    FEATURES,
    MODEL_VERSION,
    MODELS_DIR,
    PIPELINE_VERSION,
)

DISCLAIMER = (
    "Research/demo output only. This is not a diagnosis, not a clinically validated "
    "probability, and must not be used to make care decisions. Emergency criteria "
    "must come from a separate, qualified clinician-approved rules service."
)

FIELD_LIMITS = {
    "age": (1, 120),
    "trestbps": (40, 300),
    "chol": (50, 800),
    "thalach": (40, 250),
    "oldpeak": (-10.0, 20.0),
    "sex": (0, 1),
    "cp": (1, 4),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "exang": (0, 1),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}


@st.cache_resource(show_spinner="Loading the trained research model...")
def load_pipeline():
    """Load the same fitted preprocessing + calibrated estimator used by FastAPI."""
    model_path = MODELS_DIR / "heart_risk_pipeline.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Copy the trained "
            "heart_risk_pipeline.joblib into the repository's models/ directory."
        )
    return joblib.load(model_path)


def illustrative_band(probability: float) -> str:
    """Assignment-only descriptive band; not a clinical threshold."""
    if probability < 0.25:
        return "LOW"
    if probability <= 0.75:
        return "MEDIUM"
    return "HIGH"


def validate_payload(payload: dict) -> list[str]:
    errors = []
    for field, (minimum, maximum) in FIELD_LIMITS.items():
        value = payload.get(field)
        if value is None:
            if field in {"ca", "thal"}:
                continue
            errors.append(f"{field} is required")
        elif not minimum <= value <= maximum:
            errors.append(f"{field} must be between {minimum} and {maximum}")
    return errors


st.set_page_config(page_title="DJP Heart-Risk Demo", page_icon="🫀", layout="wide")
st.title("DJP · Heart-Risk Decision-Support Demo")
st.caption("HackBase ML Intern Assignment | Research/demo mode only")
st.warning(DISCLAIMER, icon="⚠️")
st.info(
    "Use synthetic/non-identifiable demonstration values only. This interface is not "
    "for clinical use, diagnosis, triage, or patient-facing decisions."
)

with st.sidebar:
    st.subheader("Model status")
    st.caption("Inference runs directly inside this Streamlit app; no FastAPI URL is required.")
    try:
        pipeline = load_pipeline()
        st.success("Trained model loaded")
        st.caption(f"Model version: {MODEL_VERSION}")
        st.caption(f"Pipeline version: {PIPELINE_VERSION}")
    except Exception as exc:  # show a clear setup issue instead of a broken page
        pipeline = None
        st.error("Model unavailable")
        st.caption(str(exc))

st.subheader("Synthetic example input")
st.caption("Defaults are illustrative values for demonstrating the model contract, not medical advice.")
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=54)
        sex = st.selectbox(
            "Sex code (dataset encoding)", [0, 1], index=1,
            help="Dataset coding only; not a statement about gender identity.",
        )
        cp = st.selectbox("Chest-pain type code", [1, 2, 3, 4], index=0)
        trestbps = st.number_input("Resting blood pressure (mmHg)", 40, 300, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 50, 800, 240)
    with c2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (code)", [0, 1])
        restecg = st.selectbox("Resting ECG code", [0, 1, 2])
        thalach = st.number_input("Maximum heart rate achieved (bpm)", 40, 250, 150)
        exang = st.selectbox("Exercise-induced angina (code)", [0, 1])
        oldpeak = st.number_input(
            "ST depression (oldpeak)", min_value=-10.0, max_value=20.0,
            value=1.0, step=0.1,
        )
    with c3:
        slope = st.selectbox("ST segment slope code", [1, 2, 3], index=1)
        ca_choice = st.selectbox("Major vessels (ca)", ["Unknown / missing", "0", "1", "2", "3"])
        thal_choice = st.selectbox("Thalassemia code", ["Unknown / missing", "3", "6", "7"])
        mock_emergency = st.selectbox(
            "Independent mock clinician-rule result",
            ["Not supplied", "False", "True"],
            help=("Test fixture only. This is not a clinical rule and is never "
                  "inferred from ML probability."),
        )
    submitted = st.form_submit_button("Run research demo prediction", type="primary")

if submitted:
    payload = {
        "age": float(age), "sex": int(sex), "cp": int(cp),
        "trestbps": float(trestbps), "chol": float(chol), "fbs": int(fbs),
        "restecg": int(restecg), "thalach": float(thalach),
        "exang": int(exang), "oldpeak": float(oldpeak), "slope": int(slope),
        "ca": None if ca_choice.startswith("Unknown") else int(ca_choice),
        "thal": None if thal_choice.startswith("Unknown") else int(thal_choice),
    }
    errors = validate_payload(payload)
    if errors:
        for error in errors:
            st.error(error)
    elif pipeline is None:
        st.error(
            "Cannot run prediction because the trained model is missing. "
            "Add models/heart_risk_pipeline.joblib to the deployed repository."
        )
    else:
        try:
            # Keep the exact feature order expected by the training pipeline.
            features = pd.DataFrame([{name: payload.get(name) for name in FEATURES}], columns=FEATURES)
            probability = float(pipeline.predict_proba(features)[0][1])
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Model returned a probability outside [0, 1].")

            # This is a separate mock fixture, never inferred from the model output.
            emergency_flag = None if mock_emergency == "Not supplied" else mock_emergency == "True"

            st.subheader("Model output")
            m1, m2, m3 = st.columns(3)
            m1.metric("Model estimate", f"{probability:.3f}")
            m2.metric("Illustrative demo band", illustrative_band(probability))
            m3.metric(
                "Emergency flag (independent)",
                "Not supplied" if emergency_flag is None else str(emergency_flag),
            )
            st.caption(
                "The LOW / MEDIUM / HIGH band is provisional and unvalidated; "
                "it is not a clinical threshold."
            )
            st.json({
                "model_version": MODEL_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "inference_mode": "local_streamlit_model",
                "emergency_rule_source": (
                    "not supplied" if emergency_flag is None
                    else "independent mock fixture; not a clinical rules engine"
                ),
            })
            st.warning(DISCLAIMER)
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

st.divider()
st.subheader("What this demo deliberately does not do")
st.markdown(
    "- It does not diagnose heart disease or recommend treatment.\n"
    "- It does not infer emergency status from the model score.\n"
    "- It does not accept patient identifiers or upload medical reports.\n"
    "- The mock emergency flag only demonstrates the independent rules boundary.\n"
    "- Real emergency rules, thresholds, messaging, and escalation require qualified clinician approval."
)
