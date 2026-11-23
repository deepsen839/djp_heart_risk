"""Generate explicitly synthetic, longitudinal records for the Feast demo."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "heart_risk_features.parquet"


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for person_id in range(1, 201):
        age = int(rng.integers(30, 81))
        base_bp = float(rng.normal(124, 13))
        base_chol = float(rng.normal(205, 28))
        base_hr = float(rng.normal(72, 9))
        bmi = float(np.clip(rng.normal(26, 4), 16, 42))
        smoking = int(rng.random() < 0.22)
        for visit in range(12):
            event_time = start + pd.Timedelta(days=30 * visit + int(rng.integers(0, 5)))
            rows.append({
                "synthetic_person_id": person_id,
                "event_timestamp": event_time,
                "created_timestamp": event_time + pd.Timedelta(hours=1),
                "age": age,
                "systolic_bp": np.float32(np.clip(base_bp + rng.normal(0, 5), 85, 210)),
                "cholesterol": np.float32(np.clip(base_chol + rng.normal(0, 12), 100, 360)),
                "resting_heart_rate": np.float32(np.clip(base_hr + rng.normal(0, 4), 40, 130)),
                "activity_minutes_7d": np.float32(np.clip(rng.normal(150, 65), 0, 500)),
                "smoking_flag": smoking,
                "bmi": np.float32(np.clip(bmi + rng.normal(0, 0.5), 16, 42)),
            })
    frame = pd.DataFrame(rows).sort_values(["event_timestamp", "synthetic_person_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    print(f"Wrote {len(frame):,} synthetic feature rows to {OUT}")
    print("Synthetic only: no real people, patient records, or clinical outcomes are represented.")


if __name__ == "__main__":
    main()
