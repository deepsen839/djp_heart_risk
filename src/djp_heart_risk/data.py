from __future__ import annotations
import pandas as pd
from .config import FEATURES, TARGET, RAW_DIR

UCI_DATASET_ID = 45
SOURCE_URL = "https://archive.ics.uci.edu/dataset/45/heart+disease"


def fetch_data() -> pd.DataFrame:
    """Fetch the UCI Heart Disease dataset; use Cleveland subset only."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("Install project dependencies (`pip install -e .`) to fetch the UCI dataset") from exc
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if X is None or y is None:
        raise RuntimeError("UCI dataset did not return both features and target")
    # ucimlrepo exposes the combined repository; restrict to Cleveland rows.
    # The original Cleveland subset is the first 303 rows in the canonical UCI file.
    frame = X.join(y)
    frame = frame.iloc[:303].copy()
    target_col = y.columns[0]
    frame = frame.rename(columns={target_col: TARGET})
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame[TARGET] = (frame[TARGET] > 0).astype("int64")
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Unexpected UCI schema; missing features: {missing}")
    frame = frame[FEATURES + [TARGET]].copy()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_DIR / "cleveland_heart_disease.csv", index=False)
    return frame


def load_local_data(path: str | None = None) -> pd.DataFrame:
    if path:
        frame = pd.read_csv(path)
    else:
        candidate = RAW_DIR / "cleveland_heart_disease.csv"
        frame = pd.read_csv(candidate) if candidate.exists() else fetch_data()
    if TARGET not in frame:
        raise ValueError(f"Dataset must contain target column '{TARGET}'")
    return frame
