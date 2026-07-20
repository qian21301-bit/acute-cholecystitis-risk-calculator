#!/usr/bin/env python3
"""Train GBM model for the Acute Cholecystitis Risk Calculator."""
import pandas as pd, numpy as np, json, warnings, re, joblib, os
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
dr = pd.read_excel(os.path.join(BASE, '开发组.xlsx')).drop_duplicates()
dr.columns = [c.replace('\xa0',' ').strip() for c in dr.columns]
d = json.load(open(os.path.join(BASE, '_final_config.json')))
FS = d['features']

X_clean = pd.DataFrame()
for f in FS:
    X_clean[f] = pd.to_numeric(dr[f].astype(str).str.replace(',','.').str.strip(), errors='coerce')
yd = dr['target'].values
Xi = X_clean.fillna(X_clean.median()).values

gbm = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=50, random_state=42)
gbm.fit(Xi, yd)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(gbm, Xi, yd, cv=cv, method='predict_proba')[:, 1]
print(f"OOF AUC: {roc_auc_score(yd, oof):.4f}")

# Save model
joblib.dump(gbm, os.path.join(BASE, 'gbm_model.joblib'))
print("✅ Model saved to gbm_model.joblib")

# Save feature metadata
medians = {f: float(X_clean[f].median()) for f in FS}
meta = {
    'features': FS,
    'medians': medians,
    'threshold': 0.5,
}
json.dump(meta, open(os.path.join(BASE, 'feature_meta.json'), 'w'), indent=2)
print("✅ Feature metadata saved")
