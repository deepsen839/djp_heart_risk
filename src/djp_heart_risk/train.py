from __future__ import annotations
import argparse, json, platform
from datetime import datetime, timezone
import joblib, numpy as np, pandas as pd, sklearn, os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from .config import *
from .data import load_local_data


def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)], remainder="drop")


def candidates():
    models = {
      "dummy_prior": DummyClassifier(strategy="prior"),
      "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
      "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=250, max_depth=2, learning_rate=0.03, subsample=0.85, colsample_bytree=0.85, reg_lambda=2, eval_metric="logloss", random_state=SEED, n_jobs=2)
    except Exception as exc:
        print(f"XGBoost unavailable; justified alternative is ExtraTrees. Reason: {exc}")
        models["extra_trees_alternative"] = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=-1)
    return models


def safe_auc(y, p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan")


def main():
    global SEED
    parser=argparse.ArgumentParser(); parser.add_argument("--data",default=None); parser.add_argument("--seed",type=int,default=SEED); args=parser.parse_args()
    SEED=args.seed
    df=load_local_data(args.data).dropna(subset=[TARGET]).copy()
    X=df[FEATURES]; y=df[TARGET].astype(int)
    # 60/20/20 split: test set is held back until final evaluation.
    X_train, X_temp, y_train, y_temp=train_test_split(X,y,test_size=.40,random_state=SEED,stratify=y)
    X_val, X_test, y_val, y_test=train_test_split(X_temp,y_temp,test_size=.50,random_state=SEED,stratify=y_temp)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    rows=[]; fitted={}
    estimators=candidates()
    for name, estimator in estimators.items():
        pipe=Pipeline([("preprocess",make_preprocessor()),("model",estimator)])
        # CV is restricted to training partition; validation selects the candidate.
        cv_scores=cross_val_score(pipe,X_train,y_train,cv=cv,scoring="roc_auc",n_jobs=1)
        pipe.fit(X_train,y_train)
        val_prob=pipe.predict_proba(X_val)[:,1]
        rows.append({"model":name,"cv_roc_auc_mean":float(np.nanmean(cv_scores)),"cv_roc_auc_std":float(np.nanstd(cv_scores)),"validation_roc_auc":safe_auc(y_val,val_prob),"validation_accuracy":float(accuracy_score(y_val,(val_prob>=.5).astype(int))),"validation_recall":float(recall_score(y_val,(val_prob>=.5).astype(int),zero_division=0)),"validation_precision":float(precision_score(y_val,(val_prob>=.5).astype(int),zero_division=0)),"validation_f1":float(f1_score(y_val,(val_prob>=.5).astype(int),zero_division=0))})
        fitted[name]=pipe
    table=pd.DataFrame(rows).sort_values(["validation_roc_auc","cv_roc_auc_mean"],ascending=False,na_position="last")
    # Select by validation ROC-AUC; deterministic tie-break by CV ROC-AUC. Never use test set.
    selected=str(table.iloc[0]["model"])
    final=Pipeline([("preprocess",make_preprocessor()),("model",estimators[selected].__class__(**estimators[selected].get_params()))])
    # Refit selected configuration on train + validation only after selection.
    X_fit=pd.concat([X_train,X_val]); y_fit=pd.concat([y_train,y_val])
    calibrated = CalibratedClassifierCV(estimator=final, method="sigmoid", cv=5)
    calibrated.fit(X_fit,y_fit)
    final = calibrated
    MODELS_DIR.mkdir(exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True); CONFIGS_DIR.mkdir(exist_ok=True)
    joblib.dump(final,MODELS_DIR/"heart_risk_pipeline.joblib")
    table.to_csv(REPORTS_DIR/"model_comparison.csv",index=False)
    config={"selected_model":selected,"seed":SEED,"split":{"train":0.60,"validation":0.20,"test":0.20},"selection_metric":"validation ROC-AUC; CV ROC-AUC tie-break","probability_calibration":"sigmoid via 5-fold CalibratedClassifierCV on train+validation only","cv":"5-fold stratified on training partition only","test_used_for_selection":False,"train_rows":len(X_train),"validation_rows":len(X_val),"test_rows":len(X_test),"features":FEATURES,"model_version":MODEL_VERSION,"pipeline_version":PIPELINE_VERSION,"python":platform.python_version(),"pandas":pd.__version__,"scikit_learn":sklearn.__version__,"trained_at_utc":datetime.now(timezone.utc).isoformat()}
    (CONFIGS_DIR/"training_config.json").write_text(json.dumps(config,indent=2))
    # Persist untouched test split for evaluation; data is public benchmark only.
    test=pd.concat([X_test,y_test.rename(TARGET)],axis=1); test.to_csv(DATA_DIR/"test_split.csv",index=False)
    # Optional MLflow experiment tracking. Disable by leaving MLFLOW_TRACKING_URI unset.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            import mlflow
            import mlflow.sklearn
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "djp-heart-risk"))
            with mlflow.start_run(run_name=f"{selected}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"):
                mlflow.log_params({"selected_model": selected, "seed": SEED, "cv_folds": 5, "split": "60/20/20", "selection_metric": "validation_roc_auc", "calibration": "sigmoid_cv5"})
                best_row = table.loc[table["model"] == selected].iloc[0]
                for metric_name in ["cv_roc_auc_mean", "cv_roc_auc_std", "validation_roc_auc", "validation_accuracy", "validation_recall", "validation_precision", "validation_f1"]:
                    value = float(best_row[metric_name])
                    if np.isfinite(value): mlflow.log_metric(metric_name, value)
                mlflow.log_artifact(str(REPORTS_DIR / "model_comparison.csv"))
                model_info = mlflow.sklearn.log_model(final, artifact_path="model")
                registered_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "djp-heart-risk")
                mlflow.register_model(model_uri=model_info.model_uri, name=registered_name)
                mlflow.set_tag("model_version", MODEL_VERSION)
                mlflow.set_tag("pipeline_version", PIPELINE_VERSION)
        except Exception as exc:
            print(f"MLflow tracking unavailable; training artifacts were still saved. Reason: {exc}")
    print(table.to_string(index=False)); print(f"Selected: {selected}; saved {MODELS_DIR/'heart_risk_pipeline.joblib'}")
if __name__=="__main__": main()
