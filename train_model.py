"""
Step 3 & 4: Handle Class Imbalance + Train XGBoost
===================================================
- Loads combined 2019-2021 training data
- User-level stratified split (train 80% / val 20%)
- Trains XGBoost with scale_pos_weight (class weights approach)
- Tunes hyperparameters with Optuna (optimizing F1)
- Reports Recall, Precision, F1, AUPRC, AUROC on validation set
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import optuna
import joblib
import warnings
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv('train_2019_2021_combined.csv')
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# ============================================================
# 2. DEFINE FEATURES vs METADATA
# ============================================================
# These columns must NOT be used as features (metadata or leakage)
EXCLUDE_COLS = [
    'author',              # user ID — metadata
    'window_start_time',   # timestamp — metadata
    'window_end_time',     # timestamp — metadata
    'label',               # target variable
    'is_crisis_user',      # LEAKAGE — directly reveals crisis status
    'days_to_crisis',      # LEAKAGE — directly reveals proximity to crisis
    'year',                # metadata (added during merge)
]

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"\nFeature columns: {len(feature_cols)}")
print(f"Excluded columns: {EXCLUDE_COLS}")

X = df[feature_cols]
y = df['label']

# Handle any NaN/inf values
X = X.replace([np.inf, -np.inf], np.nan)
nan_counts = X.isnull().sum()
if nan_counts.sum() > 0:
    print(f"\nWarning: {nan_counts[nan_counts > 0].shape[0]} columns have NaN values. Filling with 0.")
    X = X.fillna(0)

print(f"\nLabel distribution:")
print(f"  Normal (0): {(y == 0).sum():,}")
print(f"  Crisis (1): {(y == 1).sum():,}")
print(f"  Positive rate: {y.mean()*100:.2f}%")

# ============================================================
# 3. USER-LEVEL TRAIN/VALIDATION SPLIT
# ============================================================
print("\n" + "=" * 60)
print("USER-LEVEL TRAIN/VALIDATION SPLIT")
print("=" * 60)

# Get unique users and their crisis status
user_info = df.groupby('author')['label'].max().reset_index()
user_info.columns = ['author', 'has_crisis']

# Stratified split of USERS (not windows)
from sklearn.model_selection import train_test_split

train_users, val_users = train_test_split(
    user_info['author'],
    test_size=0.2,
    random_state=42,
    stratify=user_info['has_crisis']
)

train_users_set = set(train_users)
val_users_set = set(val_users)

# Split data by users
train_mask = df['author'].isin(train_users_set)
val_mask = df['author'].isin(val_users_set)

X_train = X[train_mask].copy()
y_train = y[train_mask].copy()
X_val = X[val_mask].copy()
y_val = y[val_mask].copy()

print(f"Train users: {len(train_users_set):,}")
print(f"Val users:   {len(val_users_set):,}")
print(f"Train windows: {len(X_train):,} (positive: {y_train.sum():,}, rate: {y_train.mean()*100:.2f}%)")
print(f"Val windows:   {len(X_val):,} (positive: {y_val.sum():,}, rate: {y_val.mean()*100:.2f}%)")

# Verify no user leakage
overlap = train_users_set & val_users_set
print(f"User overlap (should be 0): {len(overlap)}")

# ============================================================
# 4. CLASS WEIGHT CALCULATION
# ============================================================
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\nscale_pos_weight: {scale_pos_weight:.1f} (ratio of negatives to positives)")

# ============================================================
# 5. HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (Optuna, 50 trials)")
print("=" * 60)


def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'scale_pos_weight': scale_pos_weight,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': 0,
    }

    # User-level 3-fold CV on training data
    user_train_info = df[train_mask].groupby('author')['label'].max().reset_index()
    user_train_info.columns = ['author', 'has_crisis']
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for fold_train_idx, fold_val_idx in skf.split(
        user_train_info['author'], user_train_info['has_crisis']
    ):
        fold_train_users = set(user_train_info.iloc[fold_train_idx]['author'])
        fold_val_users = set(user_train_info.iloc[fold_val_idx]['author'])
        
        fold_train_mask = df[train_mask]['author'].isin(fold_train_users)
        fold_val_mask = df[train_mask]['author'].isin(fold_val_users)
        
        X_ft = X_train[fold_train_mask.values]
        y_ft = y_train[fold_train_mask.values]
        X_fv = X_train[fold_val_mask.values]
        y_fv = y_train[fold_val_mask.values]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_ft, y_ft, verbose=False)
        
        y_pred = model.predict(X_fv)
        f1 = f1_score(y_fv, y_pred)
        f1_scores.append(f1)
    
    return np.mean(f1_scores)


start_time = time.time()
study = optuna.create_study(direction='maximize', study_name='xgb_crisis_detection')
study.optimize(objective, n_trials=50, show_progress_bar=True)
elapsed = time.time() - start_time

print(f"\nTuning completed in {elapsed/60:.1f} minutes")
print(f"Best F1 (CV mean): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ============================================================
# 6. TRAIN FINAL MODEL WITH BEST PARAMS
# ============================================================
print("\n" + "=" * 60)
print("TRAINING FINAL MODEL")
print("=" * 60)

best_params = study.best_params
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'tree_method': 'hist',
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbosity': 0,
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train, verbose=False)
print("Model trained on full training set.")

# ============================================================
# 7. EVALUATE ON VALIDATION SET
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION SET EVALUATION")
print("=" * 60)

y_val_pred = final_model.predict(X_val)
y_val_proba = final_model.predict_proba(X_val)[:, 1]

# Core metrics
recall = recall_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
auroc = roc_auc_score(y_val, y_val_proba)
auprc = average_precision_score(y_val, y_val_proba)

print(f"\n{'Metric':<25} {'Value':>10}")
print("-" * 37)
print(f"{'Recall (Sensitivity)':<25} {recall:>10.4f}")
print(f"{'Precision':<25} {precision:>10.4f}")
print(f"{'F1 Score':<25} {f1:>10.4f}")
print(f"{'AUROC':<25} {auroc:>10.4f}")
print(f"{'AUPRC':<25} {auprc:>10.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(f"  TN={cm[0,0]:>6,}   FP={cm[0,1]:>6,}")
print(f"  FN={cm[1,0]:>6,}   TP={cm[1,1]:>6,}")
print(f"\n  (FN = missed crisis windows, should be as low as possible)")

print(f"\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Normal', 'Pre-Crisis']))

# ============================================================
# 8. FEATURE IMPORTANCE (Top 20)
# ============================================================
print("\n" + "=" * 60)
print("TOP 20 MOST IMPORTANT FEATURES")
print("=" * 60)

importances = final_model.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

for i, row in feat_imp.head(20).iterrows():
    print(f"  {row['feature']:<40} {row['importance']:.4f}")

# ============================================================
# 9. SAVE MODEL AND ARTIFACTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(final_model, 'xgb_crisis_model.pkl')
feat_imp.to_csv('feature_importance.csv', index=False)

# Save the feature columns list (needed for inference on 2022)
joblib.dump(feature_cols, 'feature_columns.pkl')

# Save validation metrics
metrics = {
    'recall': recall,
    'precision': precision,
    'f1': f1,
    'auroc': auroc,
    'auprc': auprc,
    'best_params': best_params,
    'train_windows': len(X_train),
    'val_windows': len(X_val),
    'train_positive': int(y_train.sum()),
    'val_positive': int(y_val.sum()),
    'scale_pos_weight': scale_pos_weight,
}
joblib.dump(metrics, 'validation_metrics.pkl')

print(f"Saved: xgb_crisis_model.pkl")
print(f"Saved: feature_importance.csv")
print(f"Saved: feature_columns.pkl")
print(f"Saved: validation_metrics.pkl")
print(f"\n✅ STEPS 3 & 4 COMPLETE. Ready for Step 5 (final test on 2022).")
