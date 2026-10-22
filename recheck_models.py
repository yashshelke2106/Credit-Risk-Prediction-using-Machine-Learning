"""Re-run the deployed-model benchmark on CORRECTED data (in-fold imputation, deduped)."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import lightgbm as lgb
import credit_data as cd
RS=cd.RANDOM_STATE; cv=StratifiedKFold(5,shuffle=True,random_state=RS)
X,y,_=cd.load_raw()
Xl,yl=cd.load_clean_legacy()
models={"Logistic Regression":lambda:LogisticRegression(max_iter=2000,class_weight="balanced",random_state=RS),
 "Random Forest":lambda:RandomForestClassifier(n_estimators=300,max_depth=15,class_weight="balanced",random_state=RS,n_jobs=-1),
 "LightGBM":lambda:lgb.LGBMClassifier(n_estimators=100,max_depth=7,learning_rate=0.1,class_weight="balanced",random_state=RS,verbose=-1)}
rows=[]
for label,(XX,yy) in {"Leaky (clean_data.csv + dupes)":(Xl,yl),"Corrected (raw, in-fold, deduped)":(X,y)}.items():
    for m,f in models.items():
        p=cd.make_pipeline(XX,f())
        a=cross_val_score(p,XX,yy,cv=cv,scoring="roc_auc",n_jobs=1)
        o=cross_val_predict(p,XX,yy,cv=cv,method="predict_proba",n_jobs=1)[:,1]
        rows.append({"Data":label,"Model":m,"CV ROC-AUC":a.mean(),"CV Std":a.std(),"KS":cd.ks_statistic(yy,o)})
        print(f"{label:36s} {m:20s} AUC {a.mean():.4f} +/- {a.std():.4f}  KS {rows[-1]['KS']:.4f}")
d=pd.DataFrame(rows); d.to_csv("model_comparison_corrected.csv",index=False)
piv=d.pivot(index="Model",columns="Data",values="CV ROC-AUC")
piv["Delta"]=piv.iloc[:,1]-piv.iloc[:,0]
print("\n",piv.round(4).to_string())
