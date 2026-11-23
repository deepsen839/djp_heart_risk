from __future__ import annotations

import os
import requests
import streamlit as st

# API_URL = os.getenv("DJP_API_URL", "http://127.0.0.1:8000").rstrip("/")
def get_api_url():
    default_url = os.getenv("DJP_API_URL", "http://127.0.0.1:8000")

    try:
        return st.secrets.get("DJP_API_URL", default_url).rstrip("/")
    except Exception:
        return default_url.rstrip("/")


API_URL = get_api_url()


DISCLAIMER = (
    "Research/demo output only. This is not a diagnosis, not a clinically validated "
    "probability, and must not be used to make care decisions. Emergency criteria "
    "must come from a separate, qualified clinician-approved rules service."
)

st.set_page_config(page_title="DJP Heart-Risk Demo", page_icon="🫀", layout="wide")
st.title("DJP · Heart-Risk Decision-Support Demo")
st.caption("HackBase ML Intern Assignment | Research/demo mode only")
st.warning(DISCLAIMER, icon="⚠️")
st.info("Use synthetic/non-identifiable demonstration values only. This interface is not for clinical use, diagnosis, triage, or patient-facing decisions.")

with st.sidebar:
    st.subheader("API connection")
    st.code(API_URL)
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        health.raise_for_status()
        health_data = health.json()
        st.success(f"API: {health_data.get('status')} · {health_data.get('mode')}")
        if not health_data.get("model_loaded"):
            st.warning("Model artefact is not present. Run the training pipeline first.")
    except requests.RequestException as exc:
        st.error(f"API unavailable: {exc}")

st.subheader("Synthetic example input")
st.caption("Defaults are illustrative values for demonstrating the API contract, not medical advice.")
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=54)
        sex = st.selectbox("Sex code (dataset encoding)", [0, 1], index=1, help="Dataset coding only; not a statement about gender identity.")
        cp = st.selectbox("Chest-pain type code", [1, 2, 3, 4], index=0)
        trestbps = st.number_input("Resting blood pressure (mmHg)", 40, 300, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 50, 800, 240)
    with c2:
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (code)", [0, 1])
        restecg = st.selectbox("Resting ECG code", [0, 1, 2])
        thalach = st.number_input("Maximum heart rate achieved (bpm)", 40, 250, 150)
        exang = st.selectbox("Exercise-induced angina (code)", [0, 1])
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=-10.0, max_value=20.0, value=1.0, step=0.1)
    with c3:
        slope = st.selectbox("ST segment slope code", [1, 2, 3], index=1)
        ca_choice = st.selectbox("Major vessels (ca)", ["Unknown / missing", "0", "1", "2", "3"])
        thal_choice = st.selectbox("Thalassemia code", ["Unknown / missing", "3", "6", "7"])
        mock_emergency = st.selectbox("Independent mock clinician-rule result", ["Not supplied", "False", "True"], help="Test fixture only. This is not a clinical rule and is never inferred from ML probability.")
    submitted = st.form_submit_button("Run research demo prediction", type="primary")

if submitted:
    payload = {
        "age": float(age), "sex": int(sex), "cp": int(cp), "trestbps": float(trestbps),
        "chol": float(chol), "fbs": int(fbs), "restecg": int(restecg),
        "thalach": float(thalach), "exang": int(exang), "oldpeak": float(oldpeak),
        "slope": int(slope), "ca": None if ca_choice.startswith("Unknown") else int(ca_choice),
        "thal": None if thal_choice.startswith("Unknown") else int(thal_choice),
    }
    if mock_emergency != "Not supplied":
        payload["mock_clinician_emergency_flag"] = mock_emergency == "True"
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        st.subheader("Model output")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model estimate", f"{result['risk_probability']:.3f}")
        m2.metric("Illustrative demo band", result["illustrative_band"])
        m3.metric("Emergency flag (independent)", "Not supplied" if result["emergency_flag"] is None else str(result["emergency_flag"]))
        st.caption("The LOW / MEDIUM / HIGH band is provisional and unvalidated; it is not a clinical threshold.")
        st.json({"model_version": result["model_version"], "pipeline_version": result["pipeline_version"], "emergency_rule_source": result["emergency_rule_source"]})
        st.warning(result["disclaimer"])
    except requests.HTTPError:
        st.error(f"Prediction request failed: {response.text}")
    except requests.RequestException as exc:
        st.error(f"Could not reach prediction API: {exc}")

st.divider()
st.subheader("What this demo deliberately does not do")
st.markdown("- It does not diagnose heart disease or recommend treatment.\n- It does not infer emergency status from the model score.\n- It does not accept patient identifiers or upload medical reports.\n- The mock emergency flag only demonstrates the independent rules boundary.\n- Real emergency rules, thresholds, messaging, and escalation require qualified clinician approval.")
