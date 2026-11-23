"""Build and query the local Feast feature store using synthetic data."""
from pathlib import Path
import os
import subprocess
import sys
import pandas as pd
from feast import FeatureStore

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "feature_repo"


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def main() -> None:
    os.environ.setdefault("FEAST_USAGE", "False")
    run(str(ROOT / "generate_synthetic_data.py"))
    subprocess.run(["feast", "apply"], cwd=REPO, check=True)
    store = FeatureStore(repo_path=str(REPO))
    source = pd.read_parquet(ROOT / "data" / "heart_risk_features.parquet")
    end = source["event_timestamp"].max().to_pydatetime()
    start = source["event_timestamp"].min().to_pydatetime()
    store.materialize(start_date=start, end_date=end)
    entity_rows = [{"synthetic_person_id": 1}, {"synthetic_person_id": 2}]
    feature_refs = [
        "synthetic_heart_risk_features:age",
        "synthetic_heart_risk_features:systolic_bp",
        "synthetic_heart_risk_features:cholesterol",
        "synthetic_heart_risk_features:resting_heart_rate",
        "synthetic_heart_risk_features:activity_minutes_7d",
        "synthetic_heart_risk_features:smoking_flag",
        "synthetic_heart_risk_features:bmi",
    ]
    historical_entities = pd.DataFrame([
        {"synthetic_person_id": entity["synthetic_person_id"], "event_timestamp": end}
        for entity in entity_rows
    ])
    historical = store.get_historical_features(entity_df=historical_entities, features=feature_refs).to_df()
    print("Historical feature retrieval (synthetic entities):")
    print(historical.tail().to_string(index=False))
    online = store.get_online_features(entity_rows=entity_rows, features=feature_refs).to_dict()
    print("\nOnline feature retrieval (synthetic entities):")
    print(online)
    print("\nFeature Store demo completed. These values are synthetic and are not for clinical use.")


if __name__ == "__main__":
    main()
