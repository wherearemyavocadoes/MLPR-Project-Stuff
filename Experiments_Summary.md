# Methodological Evaluation: Optimizing Mental Health Crisis Detection

## Overview of the Problem
Mental health crisis detection in social media is characterized by an extreme class imbalance (a 1.38% base rate). Identifying the "needle in the haystack" while mitigating thousands of false alarms represents an ongoing challenge in Data Science. 

This document details the exhaustive methodological experiments applied to to improve upon our initial baseline validation metrics, tracking why each advanced technique ultimately failed to outperform a simple, mathematically weighted baseline.

---

### 1. The Gold Standard: Weighted XGBoost (Baseline)
- **Method:** Passing all 398 raw continuous features (including 384 deep learning text embeddings) directly into an XGBoost classifier. The extreme 1.38% base rate was handled natively via class weighting (`scale_pos_weight = 72`), which severely penalizes the model for missing True Positives.
- **Validation Score:** `Recall: ~50%` | `Precision: ~5%` | `F1 Score: 0.088` | `AUROC: 0.841`
- **Verdict:** The baseline against which all other advanced ML manipulations were tested.

---

### 2. Optimize the Decision Threshold
- **Method:** Adjusting the classification cut-off probability. Instead of a hard 50% threshold, we dynamically shifted the decision boundary to ~0.53 to demand higher confidence before flagging a user.
- **Validation Score:** `F1 Score: 0.094` *(Recall crashed significantly).*
- **Why it was rejected:** In medical/crisis screening, False Negatives (missing a suicidal user) carry catastrophic costs, while False Positives cost very little. Optimizing purely for F1 by shifting the threshold mathematically sacrifices too many human lives (Recall) for a negligible gain in precision.

### 3. Variance Threshold & Feature Selection
- **Method:** Dropped near-constant features and ranked all 398 variables by their XGBoost importance. We deleted the bottom 348 "noisy" dimensions, forcing the model to train exclusively on the Top 50 strongest linguistic predictors.
- **Validation Score:** `Recall: 42.9%` | `Precision: 4.1%` | `F1 Score: 0.074`
- **Why it was rejected:** It made the model worse. This proved that the 348 "low-importance" embeddings weren't actually noise; they contained subtle, non-linear signals that XGBoost relies on to construct complex emotional profiles.

### 4. PCA (Principal Component Analysis)
- **Method:** An attempt to intelligently reduce dimensionality. We compressed the 384 embedding dimensions down to 203 Principal Components (capturing 95% of linguistic variance) to theoretically clean the signal.
- **Validation Score:** `Recall: 40.8%` | `Precision: 4.7%` | `F1 Score: 0.084`
- **Why it was rejected:** The native XGBoost algorithm handled the raw, uncompressed 398 dimensions better than linear PCA compression. 

### 5. Random Undersampling
- **Method:** Deleted 98% of the majority "Normal" class data to force a perfect 50/50 ratio during training, creating a perfectly balanced decision boundary.
- **Validation Score:** `Recall: 87.4%` | `Precision: 3.4%` | `False Alarms: 4,698` 
- **Why it was rejected:** By throwing away 98% of the normal conversations, the model completely "forgot" what normal behavior looked like. It became permanently paranoid, treating mildly negative words as crises and devastating system precision.

### 6. SMOTE (Synthetic Oversampling)
- **Method:** Handled imbalance by mathematically generating fake, synthetic "Crisis" windows. It connects dots between real rare cases in the 398-dimensional space to create new training examples.
- **Validation Score:** `Recall: 12.5%` | `Precision: 6.5%` | `F1 Score: 0.086`
- **Why it was rejected:** Drawing lines between complex human language vectors creates semantic garbage. Interpolating vectors blurred the boundaries of what actually constitutes a crisis, destroying the model's Recall.

### 7. Hybrid Sampling (SMOTE + TomekLinks)
- **Method:** A combination approach that generated synthetic crises while selectively deleting ambiguous, overlapping "normal" posts near the decision boundary.
- **Validation Score:** `Recall: 12.5%` | `F1 Score: 0.086`
- **Why it was rejected:** It performed identically to pure SMOTE. In 400 dimensions, Tomek links failed to identify clear spatial boundaries to clean.

### 8. EasyEnsemble (Balanced Bagging)
- **Method:** An advanced solution to fix Undersampling's terrible precision. We trained 15 distinct undersampled models—each interacting with a radically different scoop of Normal users—and averaged their votes.
- **Validation Score:** `Recall: 86.9%` | `Precision: 4.1%` | `F1 Score: 0.078`
- **Why it was rejected:** It mathematically succeeded in reducing False Positives by ~800, but a 4% operational precision rate alongside incredibly heavy compute overhead is not viable compared to the Baseline.

### 9. Two-Stage Cascade (The Funnel)
- **Method:** Model 1 casts a wide net that catches everything (High Recall, Poor Precision). We then fed exclusively out-of-fold hard cases to a strictly weighted Model 2, whose only job was to filter out the false alarms.
- **Validation Score:** `Recall: 0.0%` | `Precision: 0.0%`
- **Why it was rejected:** A spectacular, highly informative collapse. Stage 2 rejected *everything* (including True Crises). This proved that the False Positives (dramatic normal text) and True Positives (suicidal text) are mathematically indistinguishable when scrutinized in isolation. 

### 10. Deep Learning Anomaly Detection (AutoEncoders)
- **Method:** Built a massive PyTorch Neural Network. We trained it to perfectly memorize "Normal" language, expecting Crises to trigger massive reconstruction errors (anomalies) when fed into the network.
- **Validation Score:** `AUROC: 0.332` *(Worse than random guessing)*. 
- **Why it was rejected:** Our most profound linguistic discovery. "Normal" Reddit text is chaotic, dense, and unpredictable (causing high reconstruction errors). "Crisis" language is actually highly concentrated, repetitive, and predictable (causing *low* reconstruction errors). Crises are not structural anomalies, meaning unsupervised detection mathematically fails.

---

## Conclusion
We systematically exhausted feature selection, threshold manipulation, dimensionality reduction, sampling logic, ensemble methods, cascading architectures, and Deep Learning anomaly heuristics. 

**Conclusion:** We selected the Baseline XGBoost Model with `scale_pos_weight`.

**Why:** Crisis text is a densely patterned but highly ambiguous subset of language. Because human emotion overlaps so heavily (e.g., hyperbole vs. ideation), unsupervised methods and synthetic datasets inherently fail. Supervised classification backed by heavy, mathematically native class weighting is the *only* paradigm capable of aggressively hunting for these specific combinations across 398 dimensions without destroying the clinical mandate of high Recall.
