"""
Step 5: Final Out-of-Time Test on 2022 Data
===========================================
- Loads the untouched 2022 dataset based on strict feature alignment
- Applies the original (Option 0) model trained on 2019-2021
- Outputs the final metrics for the deliverable
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 5: OPENING THE 2022 VAULT")
print("=" * 60)

# 1. Load data
print("Loading processed_2022_modeling_ready.csv...")
df_2022 = pd.read_csv('processed_2022_modeling_ready.csv')
print(f"Total 2022 windows: {len(df_2022):,}")

# 2. Extract strict feature order
feature_cols = joblib.load('feature_columns.pkl')
print(f"Aligning strictly to {len(feature_cols)} training features...")

X_test = df_2022[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = df_2022['label']

print(f"\n2022 Label distribution:")
print(f"  Normal (0): {(y_test == 0).sum():,}")
print(f"  Crisis (1): {(y_test == 1).sum():,}")
print(f"  Positive rate: {y_test.mean()*100:.2f}%")

# 3. Load original model
print("\nLoading original 2019-2021 XGBoost model...")
model = joblib.load('xgb_crisis_model.pkl')

print("\n" + "=" * 60)
print("FINAL TEST METRICS (ON UNSEEN 2022 DATA)")
print("=" * 60)

# 4. Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 5. Evaluate
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_proba)
auprc = average_precision_score(y_test, y_proba)

print(f"{'Recall (Sensitivity)':<25} {recall:>10.4f}")
print(f"{'Precision':<25} {precision:>10.4f}")
print(f"{'F1 Score':<25} {f1:>10.4f}")
print(f"{'AUROC':<25} {auroc:>10.4f}")
print(f"{'AUPRC':<25} {auprc:>10.4f}")

print("\nFinal Confusion Matrix (2022 Test):")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:>6,}   FP={cm[0,1]:>6,}")
print(f"  FN={cm[1,0]:>6,}   TP={cm[1,1]:>6,}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Pre-Crisis']))
