from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data import load_local_data, SOURCE_URL
from .config import FEATURES, TARGET, REPORTS_DIR

DICTIONARY = {
 "age": ("Age", "numeric", "years"), "sex": ("Recorded sex code (0=female, 1=male in source)", "categorical", "0/1"),
 "cp": ("Chest-pain type code", "categorical", "1-4 source codes"), "trestbps": ("Resting blood pressure", "numeric", "mm Hg"),
 "chol": ("Serum cholesterol", "numeric", "mg/dL"), "fbs": ("Fasting blood sugar >120 mg/dL indicator", "categorical", "0/1"),
 "restecg": ("Resting ECG category", "categorical", "0-2 source codes"), "thalach": ("Maximum heart rate achieved", "numeric", "beats/min"),
 "exang": ("Exercise-induced angina indicator", "categorical", "0/1"), "oldpeak": ("ST depression induced by exercise relative to rest", "numeric", "source measurement"),
 "slope": ("Peak exercise ST segment slope code", "categorical", "1-3 source codes"), "ca": ("Number of major vessels (0-3) coloured by fluoroscopy", "categorical", "0-3; missing in source"),
 "thal": ("Thalassemia test category code", "categorical", "source codes; missing in source"),
 "target": ("Binary benchmark label: 0=no disease; 1=any positive original disease-severity value", "binary", "0/1")
}


def build_audit(df: pd.DataFrame) -> dict:
    missing_cols = [c for c in FEATURES if c not in df]
    if missing_cols: raise ValueError(f"Missing expected columns: {missing_cols}")
    out_of_range = {}
    ranges = {"age": (1, 120), "sex": (0, 1), "cp": (1, 4), "trestbps": (40, 300), "chol": (50, 800), "fbs": (0, 1), "restecg": (0, 2), "thalach": (40, 250), "exang": (0, 1), "slope": (1, 3), "ca": (0, 3), "thal": (3, 7)}
    for col, (lo, hi) in ranges.items():
        s = pd.to_numeric(df[col], errors="coerce")
        bad = s.notna() & ~s.between(lo, hi)
        out_of_range[col] = int(bad.sum())
    class_counts = df[TARGET].value_counts(dropna=False).sort_index().to_dict()
    return {
      "dataset_name": "UCI Heart Disease – Cleveland subset", "source_url": SOURCE_URL,
      "retrieval_date_utc": datetime.now(timezone.utc).date().isoformat(),
      "licence_terms": "UCI repository dataset page should be checked for current terms before redistribution; dataset is fetched at runtime and is not committed.",
      "rows": int(len(df)), "columns_including_target": int(df.shape[1]),
      "feature_columns": FEATURES, "target_definition": "Original Cleveland num label binarised: 0 -> 0; values 1-4 -> 1.",
      "class_counts": {str(k): int(v) for k,v in class_counts.items()},
      "class_proportions": {str(k): float(v/len(df)) for k,v in class_counts.items()},
      "duplicate_rows": int(df.duplicated().sum()),
      "missing_by_column": {c: int(v) for c,v in df.isna().sum().items()},
      "out_of_range_counts": out_of_range,
      "patient_identifier": None,
      "limitations": ["Small historical cohort (Cleveland subset, at most 303 rows).", "No stable patient identifier is supplied; patient-level splitting cannot be performed.", "Historical benchmark is not representative evidence for a clinic, India, or a deployment population.", "Target is a binary transformation of an ordinal disease-severity field.", "Missing values and measurement/category coding may reflect historical collection practices.", "No causal conclusions or clinical utility claims are supported."]
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data", default=None); a=p.parse_args()
    df=load_local_data(a.data); audit=build_audit(df); REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR/"data_audit.json").write_text(json.dumps(audit,indent=2))
    rows=[]
    for c in df.columns:
        meaning,typ,units=DICTIONARY.get(c,(c,"unknown","unknown"))
        rows.append({"field":c,"meaning":meaning,"type":typ,"units_categories":units,"missing":int(df[c].isna().sum()),"handling":"Median imputation" if c in ["age","trestbps","chol","thalach","oldpeak"] else ("Most-frequent imputation + one-hot encoding" if c in FEATURES else "Binary target; rows with missing target excluded")})
    pd.DataFrame(rows).to_csv(REPORTS_DIR/"data_dictionary.csv",index=False)
    print(json.dumps(audit,indent=2))
if __name__=="__main__": main()
