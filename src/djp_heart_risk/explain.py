from __future__ import annotations
import joblib, numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from .config import *


def main():
    test=pd.read_csv(DATA_DIR/"test_split.csv"); X=test[FEATURES]; y=test[TARGET]
    pipe=joblib.load(MODELS_DIR/"heart_risk_pipeline.joblib")
    # Permutation importance on held-out data is descriptive only; no model selection follows.
    result=permutation_importance(pipe,X,y,n_repeats=30,random_state=SEED,scoring="roc_auc")
    out=pd.DataFrame({"feature":FEATURES,"importance_mean":result.importances_mean,"importance_std":result.importances_std}).sort_values("importance_mean",ascending=False)
    REPORTS_DIR.mkdir(exist_ok=True); out.to_csv(REPORTS_DIR/"permutation_importance.csv",index=False)
    print(out.to_string(index=False))

    # SHAP is computed on a small, fixed sample because the dataset is small and
    # the fitted object is a complete preprocessing+calibration pipeline.
    try:
        import shap
        background = X.sample(n=min(20, len(X)), random_state=SEED)
        explain_rows = X.sample(n=min(40, len(X)), random_state=SEED + 1)
        def predict_positive(values):
            frame = pd.DataFrame(values, columns=FEATURES)
            return pipe.predict_proba(frame)[:, 1]
        explainer = shap.Explainer(predict_positive, background.to_numpy(), feature_names=FEATURES)
        shap_values = explainer(explain_rows.to_numpy(), max_evals=2 * len(FEATURES) + 1)
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = values[:, :, -1]
        shap_summary = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_summary.to_csv(REPORTS_DIR / "shap_summary.csv", index=False)
        print("\nSHAP summary (mean absolute contribution; descriptive only):")
        print(shap_summary.to_string(index=False))
    except Exception as exc:
        print(f"SHAP explanation could not be generated; permutation importance is still available: {exc}")
if __name__=="__main__": main()
