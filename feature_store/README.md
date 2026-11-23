# Feast Feature Store demo (synthetic data only)

This repository contains a runnable local Feast demonstration. It deliberately does **not** invent patient identifiers or event times for the public UCI Cleveland dataset. That dataset remains the source for the existing model-training pipeline; it is cross-sectional and does not provide the longitudinal entity history required for a meaningful feature-store demonstration.

## Data design

`generate_synthetic_data.py` deterministically generates 200 synthetic entities with 12 monthly snapshots each (2,400 rows). The synthetic fields are age, systolic blood pressure, cholesterol, resting heart rate, weekly activity minutes, smoking flag, and BMI. Entity IDs and timestamps exist only to demonstrate Feast mechanics. They do not represent real people, real patient records, clinical outcomes, or medically validated distributions.

## Feast components

- Entity: `synthetic_person_id`
- FeatureView: `synthetic_heart_risk_features`
- Event time: `event_timestamp`
- Offline source: Parquet file
- Online store: SQLite
- Registry: local SQLite
- Historical retrieval: point-in-time feature retrieval via `get_historical_features`
- Online retrieval: `get_online_features` after materialization

## Run locally

Install the optional dependencies:

```bash
pip install -e '.[features]'
python feature_store/run_feature_store_demo.py
```

Or, after building the project image:

```bash
docker compose --profile features run --rm feature-store-demo
```

The generated Parquet file and Feast SQLite databases are kept under `feature_store/data/` and are ignored by Git. The feature-store demo is independent of the Cleveland model training pipeline and does not feed synthetic data into that model.

**Safety:** synthetic values are for software engineering demonstrations only. Do not use them for diagnosis, triage, treatment, or clinical decisions.
