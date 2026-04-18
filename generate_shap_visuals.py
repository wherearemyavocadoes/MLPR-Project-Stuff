"""
generate_shap_visuals.py
========================
Loads the fully trained 2019-2021 XGBoost Baseline Model.
Analyzes the 2022 Test Dataset using SHAP (SHapley Additive exPlanations).
Generates beautiful, presentation-ready visualizations.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("1. LOADING MODEL AND 2022 TEST DATA")
print("=" * 60)

# Load the exact same ordered features
feature_cols = joblib.load('feature_columns.pkl')

# Load the test data
print("Loading processed_2022_modeling_ready.csv...")
df_2022 = pd.read_csv('processed_2022_modeling_ready.csv')
X_test = df_2022[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# Load the trained Baseline Model
print("Loading original 2019-2021 XGBoost model...")
model = joblib.load('xgb_crisis_model.pkl')

print("\n" + "=" * 60)
print("2. COMPUTING SHAP VALUES (TreeExplainer)")
print("=" * 60)
print("This may take a minute as it analyzes 398 dimensions across 16,000 users...")

# Initialize the TreeExplainer (optimized for tree-based models like XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("SHAP computation complete!")

print("\n" + "=" * 60)
print("3. GENERATING PRESENTATION VISUALS")
print("=" * 60)

# -----------------------------------------------------
# Plot 1: SHAP Summary Plot (Bar Chart)
# Displays the absolute average importance of the Top 20 features
# -----------------------------------------------------
print("Generating SHAP Bar Chart (Top 20 Features)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=20, show=False)
plt.title("Top 20 Most Important Features (SHAP Value Impact)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('presentation_plot_10_shap_bar.png', dpi=300, facecolor='white', bbox_inches='tight')
plt.close()

# -----------------------------------------------------
# Plot 2: SHAP Summary Plot (Dot Plot / Beeswarm)
# Displays HOW features impact the model (High vs Low values)
# -----------------------------------------------------
print("Generating SHAP Beeswarm Plot (Directional Impact)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, max_display=20, show=False)
plt.title("Direct Impact on Crisis Prediction (By Feature Value)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('presentation_plot_11_shap_beeswarm.png', dpi=300, facecolor='white', bbox_inches='tight')
plt.close()

print("\n✅ Success! All visual plots have been generated and saved locally:")
print("  - presentation_plot_10_shap_bar.png")
print("  - presentation_plot_11_shap_beeswarm.png")
