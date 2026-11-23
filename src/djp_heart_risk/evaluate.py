from __future__ import annotations
import argparse, json
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from .config import *


def main():
    p=argparse.ArgumentParser(); p.add_argument("--test",default=str(DATA_DIR/"test_split.csv")); a=p.parse_args()
    test=pd.read_csv(a.test); X=test[FEATURES]; y=test[TARGET].astype(int)
    model=joblib.load(MODELS_DIR/"heart_risk_pipeline.joblib"); prob=model.predict_proba(X)[:,1]; pred=(prob>=.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1])
    metrics={"n_test":len(y),"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall_sensitivity":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"roc_auc":float(roc_auc_score(y,prob)) if y.nunique()==2 else None,"brier_score":float(brier_score_loss(y,prob)),"confusion_matrix":{"tn":int(cm[0,0]),"fp":int(cm[0,1]),"fn":int(cm[1,0]),"tp":int(cm[1,1])},"operating_threshold":0.5,"threshold_note":"Demonstration threshold only; not clinically selected or validated.","classification_report":classification_report(y,pred,output_dict=True,zero_division=0)}
    # Descriptive subgroup/error review. Small cells are flagged; no fairness claim is made.
    subgroup = test[["sex", "age", TARGET]].copy()
    subgroup["age_group"] = pd.cut(subgroup["age"], bins=[0, 44, 59, 120], labels=["<=44", "45-59", "60+"], include_lowest=True)
    subgroup["prediction"] = pred
    subgroup["probability"] = prob
    subgroup_rows = []
    for feature in ["sex", "age_group"]:
        for group, part in subgroup.groupby(feature, observed=False, dropna=False):
            if len(part) == 0: continue
            yy=part[TARGET].astype(int); pp=part["prediction"].astype(int)
            subgroup_rows.append({"grouping":feature,"group":str(group),"n":int(len(part)),"positive_labels":int(yy.sum()),"accuracy":float(accuracy_score(yy,pp)),"recall":float(recall_score(yy,pp,zero_division=0)),"precision":float(precision_score(yy,pp,zero_division=0)),"false_negatives":int(((yy==1)&(pp==0)).sum()),"false_positives":int(((yy==0)&(pp==1)).sum()),"interpretation":"Descriptive only; small subgroup sample; not a fairness or clinical safety conclusion." if len(part)<20 else "Descriptive only; not a fairness or clinical safety conclusion."})
    pd.DataFrame(subgroup_rows).to_csv(REPORTS_DIR/"subgroup_error_review.csv",index=False)
    metrics["subgroup_review_file"] = "subgroup_error_review.csv"
    REPORTS_DIR.mkdir(exist_ok=True); (REPORTS_DIR/"evaluation.json").write_text(json.dumps(metrics,indent=2))
    fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(cm,display_labels=["label 0","label 1"]).plot(ax=ax,colorbar=False); fig.tight_layout(); fig.savefig(REPORTS_DIR/"confusion_matrix.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));
    if y.nunique()==2: RocCurveDisplay.from_predictions(y,prob,ax=ax)
    ax.set_title("ROC curve – held-out test"); fig.tight_layout(); fig.savefig(REPORTS_DIR/"roc_curve.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));
    if len(y)>=20 and y.nunique()==2:
        frac,mean=calibration_curve(y,prob,n_bins=min(5,max(2,len(y)//10)),strategy="quantile"); ax.plot(mean,frac,marker="o",label="Model"); ax.plot([0,1],[0,1],"--",label="Ideal"); ax.legend()
        note="Small held-out sample: curve is descriptive and highly uncertain."
    else:
        note="Calibration curve omitted: too few test examples or one class absent. Brier score is still reported with caution."
        ax.text(.5,.5,note,ha="center",va="center",wrap=True); ax.set_axis_off()
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed positive fraction"); ax.set_title("Calibration – held-out test"); fig.tight_layout(); fig.savefig(REPORTS_DIR/"calibration_curve.png",dpi=160); plt.close(fig)
    metrics["calibration_plot_note"]=note; (REPORTS_DIR/"evaluation.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
