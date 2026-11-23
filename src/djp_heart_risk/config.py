from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"
SEED = 42
TARGET = "target"
POSITIVE_LABEL = 1
MODEL_VERSION = "djp-heart-risk-1.0.0"
PIPELINE_VERSION = "preprocess-train-v1"

# The public UCI Cleveland subset has no stable patient identifier.
# Discrete clinical measurements are treated as categorical model inputs.
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
