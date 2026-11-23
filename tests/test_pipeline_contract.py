import numpy as np
from djp_heart_risk.train import make_preprocessor
from djp_heart_risk.config import FEATURES
import pandas as pd

def test_preprocessor_handles_missing_values_and_unseen_categories():
    rows=[dict(age=50,trestbps=120,chol=200,thalach=150,oldpeak=1,sex=1,cp=2,fbs=0,restecg=0,exang=0,slope=2,ca=np.nan,thal=3),
          dict(age=60,trestbps=130,chol=220,thalach=140,oldpeak=2,sex=0,cp=3,fbs=0,restecg=1,exang=1,slope=1,ca=1,thal=6)]
    X=pd.DataFrame(rows,columns=FEATURES)
    pre=make_preprocessor(); transformed=pre.fit_transform(X)
    assert transformed.shape[0]==2
    assert np.isfinite(transformed).all()
